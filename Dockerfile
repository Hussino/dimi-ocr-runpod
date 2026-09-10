FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]