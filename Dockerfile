FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Match the torch 2.8.0 + CUDA 12.8 stack already in the image.
RUN pip install --no-cache-dir --no-deps \
    torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

COPY handler.py .

CMD ["python", "-u", "handler.py"]