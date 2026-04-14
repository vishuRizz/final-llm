FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN pip install --upgrade pip==24.3.1

# Install PyTorch separately (important)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py"]