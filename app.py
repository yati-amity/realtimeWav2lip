from flask import Flask, Response, render_template, redirect, request, jsonify
import os
import subprocess
from inference import main
import time
import threading
import queue
import numpy as np
import librosa

app=Flask(__name__) 

app.config['IMAGE_DIR'] = './assets/uploaded_images/' 
app.config['Filename'] = 'Elon_Musk.jpg'  # Default face image

# Ensure required directories exist
os.makedirs('./assets/uploaded_images/', exist_ok=True)
os.makedirs('./temp/', exist_ok=True)

# Thread-safe audio queue: browser sends PCM chunks here, inference reads from it
audio_queue = queue.Queue(maxsize=50)

def remove_files_in_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file and remove it
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload():

    # clear images folder
    remove_files_in_directory(app.config['IMAGE_DIR'])

    # Check if the POST request has the file part
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'

    # Save the file to the asset folder
    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['IMAGE_DIR'], file.filename))

    # Return JSON for AJAX requests, redirect for form submissions
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({"status": "ok", "filename": file.filename})
    return redirect("/")

global flag
flag = 0

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global flag
    try:    
        if request.method == 'POST':
            if request.form.get('start') == 'Start':
                flag = 1
            elif  request.form.get('stop') == 'Stop':
                flag = 0
                # Clear audio queue on stop
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
            elif  request.form.get('clear') == 'clear':
                flag = 0
            print(f"Flag value {flag}")
            return jsonify({"status": "ok", "flag": flag})
        elif request.method=='GET':
            return render_template('index.html')
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

    return render_template("index.html")

# Endpoint to receive audio from browser
@app.route('/audio', methods=['POST'])
def receive_audio():
    try:
        raw_data = request.get_data()
        # Browser sends 16-bit PCM at 16kHz, mono
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        if len(audio_data) > 0:
            try:
                audio_queue.put_nowait(audio_data)
            except queue.Full:
                # Drop oldest chunk if queue is full
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                audio_queue.put_nowait(audio_data)
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Audio receive error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


# Endpoint to receive an audio file, decode it, and feed chunks into the audio queue
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    global flag
    try:
        if 'audio' not in request.files:
            return jsonify({"status": "error", "message": "No audio file"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Clear old uploaded audio files, then save new one
        ext = os.path.splitext(audio_file.filename)[1] or '.wav'
        playback_dir = os.path.join('static', 'uploaded_audio')
        os.makedirs(playback_dir, exist_ok=True)
        for old_file in os.listdir(playback_dir):
            os.remove(os.path.join(playback_dir, old_file))
        playback_filename = 'current_audio' + ext
        playback_path = os.path.join(playback_dir, playback_filename)
        audio_file.save(playback_path)

        wav, sr = librosa.load(playback_path, sr=16000, mono=True)

        # Convert float32 [-1,1] to int16 PCM (same format as browser mic)
        wav_int16 = (wav * 32767).astype(np.int16)

        # Clear any existing audio in the queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        # Split into 0.5s chunks (8000 samples at 16kHz) and queue them
        chunk_size = 8000
        num_chunks = 0
        for i in range(0, len(wav_int16), chunk_size):
            chunk = wav_int16[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad the last chunk with silence
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                audio_queue.put_nowait(chunk)
            num_chunks += 1

        # Auto-start lip sync
        flag = 1

        duration = len(wav_int16) / 16000
        audio_url = '/' + playback_path.replace(os.sep, '/')
        print(f"Audio file uploaded: {audio_file.filename}, duration: {duration:.1f}s, chunks: {num_chunks}")
        return jsonify({"status": "ok", "duration": duration, "chunks": num_chunks, "audio_url": audio_url})
    except Exception as e:
        print(f"Audio file upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    global flag
    try:    
        if app.config['Filename']!='':        
            response = Response(main(os.path.join(app.config['IMAGE_DIR'], app.config['Filename']), lambda: flag, audio_queue),
                               mimetype='multipart/x-mixed-replace; boundary=frame')
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
    except Exception as e:
        print(e)
    return ""

def generate_self_signed_cert(cert_file='cert.pem', key_file='key.pem'):
    """Generate a self-signed SSL certificate if it doesn't exist.
    Required for getUserMedia (mic access) from remote browsers over HTTPS."""
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Generating self-signed SSL certificate for HTTPS...")
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
            '-keyout', key_file, '-out', cert_file,
            '-days', '365', '-nodes', '-subj', '/CN=wav2lip-server'
        ], check=True, capture_output=True)
        print(f"SSL certificate generated: {cert_file}, {key_file}")

if __name__=="__main__":
    # HTTPS is required for getUserMedia (mic access) from remote browsers
    generate_self_signed_cert()
    ssl_context = ('cert.pem', 'key.pem')
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
