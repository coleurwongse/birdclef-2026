import pyaudio
import tensorflow as tf
import numpy as np
import librosa

model = tf.keras.models.load_model('audio_classification_model.h5')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                input=True, frames_per_buffer=1024)

frames = []
for i in range(0, int(44100 / 1024 * 5)): # Record for 5 seconds
    data = stream.read(1024)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()
audio_data = np.frombuffer(frames, dtype=np.float32)
spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio_data, sr=16000, n_mels=128)).T