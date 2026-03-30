#!/bin/bash
# RunPod GPU Pod startup script for realtimeWav2lip
# Usage: Deploy a RunPod GPU Pod with PyTorch template, then run this script.
#
# Prerequisites:
#   1. Upload this entire project (including model checkpoints) to the pod
#   2. Run: bash runpod_start.sh

set -e

echo "=== realtimeWav2lip RunPod Setup ==="

# Install system dependencies for OpenCV and audio processing
apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libglib2.0-0 libsndfile1 openssl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (skip pyaudio — not needed, audio comes from browser)
pip install $(grep -v pyaudio requirements.txt)

# Verify model checkpoints exist
if [ ! -f "Wav2Lip/checkpoints/wav2lip_onnx_export.onnx" ]; then
    echo "ERROR: Model checkpoint not found at Wav2Lip/checkpoints/wav2lip_onnx_export.onnx"
    echo "Make sure you uploaded the full project including checkpoints."
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "Checkpoints found:"
ls -lh Wav2Lip/checkpoints/
ls -lh checkpoints/mobilenet.pth 2>/dev/null || echo "(mobilenet.pth will auto-download on first run)"
echo ""
echo "Starting server on port 8080..."
echo "Access via: https://<your-pod-id>-8080.proxy.runpod.net"
echo ""

python app.py
