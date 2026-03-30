# CUDA base image for GPU inference on RunPod
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsndfile1 \
        openssl \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (skip pyaudio - not needed)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir $(grep -v pyaudio requirements.txt)

# Copy application code
COPY . /app

# RunPod exposes this port
EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["/app/app.py"]
