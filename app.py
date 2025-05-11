import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
import librosa
import numpy as np
import os
import tempfile

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

NOTE_FREQS = {
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20,
    'F1': 43.65, 'F#1': 46.25, 'G1': 49.00, 'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41,
    'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81,
    'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.26,
    'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51
}

def get_closest_note(freq):
    return min(NOTE_FREQS, key=lambda note: abs(NOTE_FREQS[note] - freq))

# REST endpoint (fallback)
@app.route('/api/detect-note', methods=['POST'])
def detect_note():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        file.save(temp_webm.name)
        webm_path = temp_webm.name

    try:
        y, sr = librosa.load(webm_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        index = magnitudes.argmax()
        freq = pitches[index // pitches.shape[1], index % pitches.shape[1]]
        os.remove(webm_path)
    except Exception as e:
        os.remove(webm_path)
        return {"error": f"Failed to process audio: {str(e)}"}, 500

    if freq > 0:
        note = get_closest_note(freq)
        return {'note': note, 'frequency': round(float(freq), 2)}
    else:
        return {'note': 'No note detected'}

# WebSocket: ping test
@socketio.on('ping')
def handle_ping(data):
    print("📡 Ping received from client:", data)
    socketio.emit('pong', {'message': 'Pong from server'})

# WebSocket: real-time audio chunk
@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            y, sr = librosa.load(tmp.name, sr=None)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            index = magnitudes.argmax()
            freq = pitches[index // pitches.shape[1], index % pitches.shape[1]]
        except Exception as e:
            print("WebSocket audio error:", e)
            freq = 0
        finally:
            os.remove(tmp.name)

    if freq > 0:
        note = get_closest_note(freq)
        socketio.emit('note_detected', {
            'note': note,
            'frequency': round(float(freq), 2)
        })
    else:
        socketio.emit('note_detected', {'note': 'No note detected'})

if __name__ == '__main__':
    import eventlet
    import eventlet.wsgi
    eventlet.monkey_patch()
    socketio.run(app, host='0.0.0.0', port=5000)



