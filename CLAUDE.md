# CLAUDE.md - Implementation Context for realtimeWav2lip

## Project Overview

Real-time Wav2Lip lip-syncing web application. A user uploads a face image via a browser, speaks into their microphone, and the system generates lip-synced video frames streamed back in real-time via MJPEG.

**Upstream repo:** https://github.com/devkrish23/realtimeWav2lip
**Fork:** https://github.com/tutor-arc/realtimeWav2lip

## Architecture

```
Browser (remote machine)                    Server (GPU machine)
┌─────────────────────┐                    ┌──────────────────────────────┐
│ Upload face image    │──── POST /upload ──│ Save to assets/              │
│                      │                    │                              │
│ <img src=/video_feed>│◄── MJPEG stream ──│ main() generator yields      │
│                      │                    │ JPEG frames continuously     │
│ getUserMedia (mic)   │                    │                              │
│ ScriptProcessorNode  │── POST /audio ───│ audio_queue (thread-safe)    │
│ 16kHz int16 PCM      │   every 0.5s      │                              │
│                      │                    │ Wav2LipInference reads queue │
│ Start/Stop buttons   │── POST /requests ─│ Sets global flag (0/1)       │
└─────────────────────┘                    └──────────────────────────────┘
```

### Key Data Flow
1. Browser captures mic audio via Web Audio API, downsamples to 16kHz mono, converts to int16 PCM
2. Audio chunks (0.5s each) are POSTed to `/audio` endpoint
3. Server queues chunks in `audio_queue` (max 50, drops oldest on overflow)
4. `inference.py:main()` generator loop reads audio from queue, computes mel spectrogram, runs Wav2Lip model, composites predicted lip region onto face, yields MJPEG frames
5. Flask streams frames via `multipart/x-mixed-replace` response

### HTTPS Requirement
Browsers require a **secure context** (HTTPS) for `navigator.mediaDevices.getUserMedia()` when accessed from non-localhost. The server auto-generates a self-signed SSL certificate at startup (`cert.pem`, `key.pem`). Remote browsers must accept the certificate warning.

## File Structure

### Core Application Files (MODIFIED from upstream)
- **`app.py`** — Flask server entry point. Routes: `/`, `/upload`, `/requests`, `/audio`, `/video_feed`. Manages global `flag` (start/stop), `audio_queue`, SSL cert generation.
- **`inference.py`** — Main inference pipeline. `Wav2LipInference` class loads model + face detector, reads audio from queue, computes mel chunks, runs batched inference. `main()` is a generator that yields MJPEG frames.
- **`templates/index.html`** — Web UI with Bootstrap dashboard, image upload, Start/Stop buttons, MJPEG display, and JavaScript for browser mic capture via Web Audio API.
- **`face_detection/detection/sfd/sfd_detector.py`** — Fixed `torch.load` for PyTorch 2.6 compatibility (`weights_only=False`).

### Model/ML Files (NOT modified)
- **`audio.py`** — Mel spectrogram computation using librosa. `melspectrogram(wav)` expects float32 waveform in [-1,1].
- **`hparams.py`** — Hyperparameters: `sample_rate=16000`, `hop_size=200`, `win_size=800`, `n_fft=800`, `num_mels=80`, `img_size=96`.
- **`models/wav2lip.py`** — Wav2Lip neural network architecture (PyTorch).
- **`models/conv.py`** — Convolution building blocks.
- **`models/syncnet.py`** — SyncNet discriminator (used in training only).
- **`face_detection/`** — RetinaFace face detection (uses `batch_face` library).

### Files NOT used at runtime
- **`flaskapp_wav2lip.py`** — REMOVED. Old Flask entry point (used by Dockerfile). Had bugs: passed `flag` by value, used PyAudio.
- **`hq_wav2lip_train.py`** — Training script, not used for inference.
- **`preprocess.py`** — Data preprocessing, not used for inference.
- **`face_detect.py`** — Standalone face detection script, not used by app.
- **`model_conversion.py`** — PyTorch→ONNX→OpenVINO conversion utility.

### Static Assets
- **`static/`** — Bootstrap admin template (Corona). CSS, JS, fonts, images. Most are unused boilerplate from the template. The only JS files used at runtime are `vendor.bundle.base.js`, `form-browse-file.js`, and the inline `<script>` in `index.html`.

## Model Checkpoints

### Wav2Lip Model (REQUIRED - not in git)

The model files must be placed in `Wav2Lip/checkpoints/`. The ONNX file is the default checkpoint.

**IMPORTANT: The SharePoint download links in the original README are ALL dead.** Use the Google Drive links below.

| File | Size | Source |
|------|------|--------|
| `wav2lip_onnx_export.onnx` | 139MB | Exported from Wav2Lip+GAN .pth via `model_conversion.py` |
| `wav2lip_openvino_model.bin` | 70MB | Converted from ONNX via OpenVINO |
| `wav2lip_openvino_model.xml` | 434KB | Converted from ONNX via OpenVINO |

**Download Wav2Lip+GAN (OpenVINO) from Google Drive:**
https://drive.google.com/drive/folders/193qN6CXkuDorYOHuVj-qDQmI0MLDHlu-

This folder contains the ONNX export and OpenVINO IR files. Download all three files into `Wav2Lip/checkpoints/`.

If you need the original PyTorch `.pth` weights (e.g., `wav2lip_gan.pth`) for re-conversion, look for community mirrors — the official SharePoint links no longer work.

### Face Detection Model (auto-downloaded)

The `mobilenet.pth` face detection model is **automatically downloaded** at runtime by the `batch_face` library when `checkpoints/mobilenet.pth` is not found. It downloads from:
```
https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth
```

The file is cached by `torch.utils.model_zoo` in `~/.cache/torch/hub/checkpoints/`. You do NOT need to manually download this.

If you want to pre-download it manually:
```bash
wget -O checkpoints/mobilenet.pth https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth
```

## Environment Setup

### Conda Environment
```bash
conda create -n wav2lip python=3.10
conda activate wav2lip
pip install -r requirements.txt
```

### Key Dependencies
- `torch` (with CUDA for GPU inference)
- `openvino` — used for ONNX/OpenVINO model inference (works on both CPU and GPU)
- `batch-face` — RetinaFace face detection with auto-download of mobilenet weights
- `flask` — web server
- `librosa` — audio mel spectrogram
- `opencv-python` — image/video processing
- `numpy`, `pandas`, `tqdm`

**Note:** `pyaudio` is listed in `requirements.txt` but is **no longer used** at runtime. Audio comes from the browser via Web Audio API. It can be removed from requirements.

### Running
```bash
conda activate wav2lip
python app.py
```
Server starts on `https://0.0.0.0:8080`. Accept the self-signed cert warning in your browser.

## Changes Made from Upstream

### Bug Fixes
1. **`torch.load` PyTorch 2.6 breaking change** — Added `weights_only=False` in `inference.py` (lines 112, 114-116) and `face_detection/detection/sfd/sfd_detector.py` (line 25). PyTorch 2.6 changed the default to `weights_only=True`.

2. **Flag passed by value** — Upstream passed `flag` (an int) to `main()`, so Start/Stop never propagated. Fixed: `app.py` passes `lambda: flag` callable, `inference.py` calls `get_flag()`.

3. **`update_frames()` used `return` instead of `yield`** — Only the first frame was ever sent. Fixed to `yield`.

4. **`main()` used `yield` instead of `yield from`** — Frames from `update_frames()` were not properly delegated. Fixed to `yield from update_frames(...)`.

5. **Model initialized on page load** — Model loaded immediately when `/video_feed` was requested, before Start was clicked. Fixed: `main()` now shows original face in a loop and only initializes `Wav2LipInference` when the Start flag is set.

### Audio System Rewrite
6. **Replaced PyAudio with browser audio capture** — The original used `pyaudio` to capture from the server's microphone. This doesn't work when the browser is on a different machine. Replaced with:
   - **Browser side** (`index.html`): `getUserMedia` + `ScriptProcessorNode` captures mic, downsamples to 16kHz, converts to int16 PCM, POSTs to `/audio` every 0.5s.
   - **Server side** (`app.py`): New `/audio` POST endpoint receives PCM chunks into thread-safe `audio_queue`.
   - **Inference** (`inference.py`): `get_audio_from_queue()` reads from queue with timeout, falls back to silence.

### ONNX/OpenVINO Loading
7. **Auto-detect model format** — `load_model()` in `inference.py` detects `.onnx`/`.xml` file extensions and uses OpenVINO instead of `torch.load`. Sets `use_openvino=True` flag for inference path.

### Other
8. **Missing mobilenet fallback** — `load_batch_face_model()` falls back to `model_path=None` when `checkpoints/mobilenet.pth` doesn't exist, triggering `batch_face`'s auto-download.
9. **Start/Stop use AJAX** — Buttons use `fetch()` instead of form POST to avoid page reloads that kill the MJPEG stream. Server returns JSON for POST requests.
10. **HTTPS support** — Auto-generates self-signed SSL cert at startup for `getUserMedia` on remote browsers.
11. **Audio int16→float conversion** — `get_mel_chunks()` converts int16 audio to float32 before calling `audio.melspectrogram()` which expects float data.

## Known Issues / Future Work

1. **`ScriptProcessorNode` is deprecated** — Should migrate to `AudioWorklet` for browser audio capture. Works for now but may be removed in future browser versions.

2. **End-to-end testing needed** — The full pipeline (browser mic → server inference → lip-synced frames back to browser) needs testing with actual speech input.

3. **Single-user only** — Global `flag` and `audio_queue` mean only one user can use the app at a time. Would need session-based state for multi-user.

4. **Dockerfile is outdated** — References the removed `flaskapp_wav2lip.py`. Needs updating to use `app.py` and remove PyAudio dependency.

5. **No audio playback** — The lip-synced video is visual only. There's no mechanism to play back the user's audio synchronized with the video on the receiving end.

6. **Static assets bloat** — The Bootstrap Corona admin template includes hundreds of unused files (flag SVGs, sample images, SCSS, gulp tasks, etc.). Could be cleaned up significantly.

## Inference Pipeline Details

### Audio Processing
- Sample rate: 16kHz
- Chunk duration: 0.5 seconds (8000 samples per chunk)
- Input format: int16 PCM from browser, converted to float32 [-1,1] for mel spectrogram
- Mel spectrogram: 80 mel bins, n_fft=800, hop_size=200, win_size=800
- Mel chunks: 16 frames wide, stepped by `80/fps` mel frames per video frame

### Model Inference
- Face detection: RetinaFace with MobileNet backbone (from `batch_face`)
- Face crop: 96x96 pixels, lower half masked (model predicts lip region)
- Batch size: 8 (configurable via `--wav2lip_batch_size`)
- Model input: `[mel_batch, img_batch]` — mel spectrogram chunks + masked face images
- Model output: predicted face crops, composited back onto original frame
- Output: JPEG-encoded frames yielded as MJPEG stream

### OpenVINO vs PyTorch paths
- OpenVINO path: numpy arrays transposed to NCHW, fed to `compiled_model([mel, img])['output']`
- PyTorch path: torch tensors on CUDA, `model(mel, img)` with `torch.no_grad()`
- Default: OpenVINO (since checkpoint is .onnx). Uses GPU device when CUDA available.
