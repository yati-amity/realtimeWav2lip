# Wav2Lip: Real-Time Lip-Syncing with Browser Audio

## Introduction

Real-time Wav2Lip lip-syncing web application. Upload a face image via the browser, speak into your microphone, and the system generates lip-synced video frames streamed back in real-time via MJPEG.

This is a fork of [devkrish23/realtimeWav2lip](https://github.com/devkrish23/realtimeWav2lip) with significant bug fixes and an audio system rewrite to use browser microphone capture instead of server-side PyAudio.

| Original Paper | Demo | Colab Notebook |
|:-:|:-:|:-:|
| [Paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/Speech-to-Lip/paper.pdf) | [Demo](https://drive.google.com/file/d/1ACp7aDDOgchtABly4usLhmAAOGFpdq_c/view) | [Colab Notebook](https://colab.research.google.com/drive/15jHVLxYJvmptoYmlfpOGbNi0jSZ85hqq#scrollTo=sh72cJ0K-dfb) |

## Quick Start

### 1. Setup Environment
```bash
conda create -n wav2lip python=3.10
conda activate wav2lip
pip install -r requirements.txt
```

### 2. Download Model Weights

**IMPORTANT: The SharePoint download links from the original repo are ALL dead.** Use the links below.

#### Wav2Lip+GAN Model (REQUIRED)

Download from Google Drive and place all files into `Wav2Lip/checkpoints/`:

**[Wav2Lip+GAN (OpenVINO) - Google Drive](https://drive.google.com/drive/folders/193qN6CXkuDorYOHuVj-qDQmI0MLDHlu-)**

This folder contains three files:
| File | Size | Description |
|------|------|-------------|
| `wav2lip_onnx_export.onnx` | 139 MB | ONNX export of Wav2Lip+GAN (default checkpoint) |
| `wav2lip_openvino_model.bin` | 70 MB | OpenVINO IR weights |
| `wav2lip_openvino_model.xml` | 434 KB | OpenVINO IR model definition |

```
Wav2Lip/checkpoints/
  wav2lip_onnx_export.onnx
  wav2lip_openvino_model.bin
  wav2lip_openvino_model.xml
```

#### Face Detection Model (auto-downloaded)

The MobileNet face detection model is **automatically downloaded** at runtime by the `batch_face` library. No manual action needed.

Source URL (for reference): https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth

To pre-download manually:
```bash
wget -O checkpoints/mobilenet.pth https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth
```

### 3. Run the Server
```bash
conda activate wav2lip
python app.py
```

The server starts on **`https://0.0.0.0:8080`** with a self-signed SSL certificate (generated automatically on first run).

### 4. Use the App

1. Open `https://<server-ip>:8080` in your browser
2. **Accept the self-signed certificate warning** (required for HTTPS)
3. Upload a face image (JPG/PNG)
4. Click **Start** and allow microphone access when prompted
5. Speak -- the face will lip-sync to your audio in real-time
6. Click **Stop** to end

> **Note:** HTTPS is required because browsers block microphone access (`getUserMedia`) on non-localhost HTTP pages.

## Model Weights (All Links)

| Model | Description | Link | Status |
|-------|-------------|------|--------|
| Wav2Lip+GAN (OpenVINO) | ONNX + OpenVINO IR files for real-time inference | [Google Drive](https://drive.google.com/drive/folders/193qN6CXkuDorYOHuVj-qDQmI0MLDHlu-) | Working |
| Face Detection (MobileNet) | RetinaFace MobileNet backbone | [GitHub Release](https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth) | Working (auto-downloaded) |
| Wav2Lip | Highly accurate lip-sync (.pth) | ~~[SharePoint](https://iiitaphyd-my.sharepoint.com/...)~~ | DEAD |
| Wav2Lip + GAN | Better visual quality (.pth) | ~~[SharePoint](https://iiitaphyd-my.sharepoint.com/...)~~ | DEAD |
| Expert Discriminator | Discriminator weights | ~~[SharePoint](https://iiitaphyd-my.sharepoint.com/...)~~ | DEAD |
| Visual Quality Discriminator | GAN discriminator weights | ~~[SharePoint](https://iiitaphyd-my.sharepoint.com/...)~~ | DEAD |

## Architecture

```
Browser (remote machine)                    Server (GPU machine)
+-----------------------+                   +------------------------------+
| Upload face image     |--- POST /upload --| Save to assets/              |
|                       |                   |                              |
| <img src=/video_feed> |<-- MJPEG stream --| main() generator yields      |
|                       |                   | JPEG frames continuously     |
| getUserMedia (mic)    |                   |                              |
| ScriptProcessorNode   |-- POST /audio  --| audio_queue (thread-safe)    |
| 16kHz int16 PCM       |   every 0.5s     |                              |
|                       |                   | Wav2LipInference reads queue |
| Start/Stop buttons    |-- POST /requests -| Sets global flag (0/1)       |
+-----------------------+                   +------------------------------+
```

Audio is captured in the browser via Web Audio API, downsampled to 16kHz mono int16 PCM, and streamed to the server in 0.5s chunks. The server runs Wav2Lip inference using OpenVINO and streams back lip-synced MJPEG frames.

## Changes from Upstream

See [CLAUDE.md](CLAUDE.md) for full details. Key changes:

1. **Replaced PyAudio with browser audio capture** -- works across machines, no server mic needed
2. **Fixed streaming bugs** -- `return` vs `yield`, `yield` vs `yield from`, flag passed by value
3. **Fixed PyTorch 2.6 compatibility** -- `weights_only=False` for `torch.load`
4. **Auto-detect model format** -- ONNX/OpenVINO loaded automatically based on file extension
5. **HTTPS support** -- auto-generated self-signed SSL cert for remote `getUserMedia`
6. **AJAX Start/Stop** -- no more page reloads that kill the MJPEG stream

## Python Libraries

- `torch` -- PyTorch deep learning framework (with CUDA for GPU)
- `openvino` -- OpenVINO toolkit for optimized ONNX/IR inference
- `batch-face` -- RetinaFace face detection with auto-download
- `flask` -- Web server
- `librosa` -- Audio mel spectrogram computation
- `opencv-python` -- Image processing
- `numpy`, `pandas`, `tqdm`

> **Note:** `pyaudio` is listed in `requirements.txt` for backward compatibility but is **no longer used** at runtime. Audio comes from the browser via Web Audio API.

## Optimization

We use OpenVINO to optimize the Wav2Lip model inference. The model is exported from PyTorch to ONNX, then optionally converted to OpenVINO IR format. OpenVINO supports inference on CPUs and GPUs.

### Benchmark
| Model | Inference Time |
|-------|---------------|
| Wav2Lip + GAN (PyTorch) | 0.3 sec |
| Wav2Lip + GAN (OpenVINO) | 0.26 sec |

## Known Issues

1. **Single-user only** -- global state means one user at a time
2. **ScriptProcessorNode deprecated** -- should migrate to AudioWorklet
3. **No audio playback** -- lip-synced video is visual only
4. **Dockerfile outdated** -- needs updating to use `app.py` instead of removed `flaskapp_wav2lip.py`

## Implementation Context

See [CLAUDE.md](CLAUDE.md) for comprehensive implementation context including architecture details, data flow, inference pipeline, and all changes from upstream. This file is designed to be read by AI coding agents (Claude Code, etc.) for onboarding.
