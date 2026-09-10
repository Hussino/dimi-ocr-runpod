import base64
import io
import os

import runpod
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForCausalLM


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
ADAPTER_MODEL = "AhmedZaky1/DIMI-Arabic-OCR-V2"

MAX_NEW_TOKENS = 2048

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
)

print("Loading DIMI LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL,
)

model.eval()

print("DIMI Arabic OCR V2 loaded successfully.")


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


def run_ocr(image: Image.Image, prompt: str) -> str:

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
        truncation=False,
    )

    # Move tensors to GPU
    inputs = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Remove prompt tokens
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(event):

    job_input = event.get("input", {})

    image_b64 = job_input.get("image")

    if not image_b64:
        raise ValueError(
            "Missing input.image. Expected a base64-encoded image."
        )

    prompt = job_input.get(
        "prompt",
        # "استخرج النص العربي والأرقام الموجودة في هذه الصورة بدقة عالية."
        "Extract the Arabic text in the image exactly as it appears, and don't translate it or guess anything"
    )

    image = decode_image(image_b64)

    result = run_ocr(
        image=image,
        prompt=prompt,
    )

    return {
        "success": True,
        "text": result,
    }


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
        }
    )