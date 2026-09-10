import base64
import io

import runpod
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)


BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
ADAPTER_MODEL = "AhmedZaky1/DIMI-Arabic-OCR-V2"

DEFAULT_PROMPT = (
    "استخرج النص العربي والأرقام الموجودة في هذه الصورة بدقة عالية."
)

MAX_NEW_TOKENS = 2048


# ---------------------------------------------------------
# Load models ONCE when the worker starts
# ---------------------------------------------------------

print(f"Loading base model: {BASE_MODEL}")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
)

print(f"Loading LoRA adapter: {ADAPTER_MODEL}")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL,
)

model.eval()

print("DIMI Arabic OCR V2 is ready.")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def decode_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc


def run_ocr(
    image: Image.Image,
    prompt: str,
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Build Qwen vision-language input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to GPU
    inputs = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Remove prompt tokens
    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[:, input_length:]

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(event):

    try:

        job_input = event.get("input", {})

        image_b64 = job_input.get("image")

        if not image_b64:
            return {
                "success": False,
                "error": "Missing input.image",
            }

        prompt = job_input.get(
            "prompt",
            DEFAULT_PROMPT,
        )

        image = decode_image(image_b64)

        text = run_ocr(
            image=image,
            prompt=prompt,
        )

        return {
            "success": True,
            "text": text,
        }

    except Exception as exc:

        print(f"ERROR: {exc}")

        return {
            "success": False,
            "error": str(exc),
        }

runpod.serverless.start(
    {
        "handler": handler,
    }
)