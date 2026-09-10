FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN python -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"

CMD ["python", "-u", "handler.py"]