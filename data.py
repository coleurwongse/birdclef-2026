import os
import librosa
import numpy as np
import pandas as pd

BASE = 'birdclef-2026'

train_csv = pd.read_csv(os.path.join(BASE, 'train.csv'))
for index, row in train_csv.iterrows():
    audio_path = os.path.join(BASE, 'train_audio', row['filename'])
    label = row['primary_label']
    y, sr = librosa.load(audio_path)
    spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    spectrogram = spectrogram.T
    folder, filename = row['filename'].split('/')
    os.makedirs(f"mel-spectrograms/{folder}", exist_ok=True)
    filename = filename.replace('.ogg', '.npy')
    np.save(os.path.join('mel-spectrograms', folder, filename), spectrogram)
    if index % 100 == 0:
        print(index)