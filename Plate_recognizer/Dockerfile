FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python-is-python3 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]