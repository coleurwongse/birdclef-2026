import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import pandas as pd
import sklearn

BASE = 'birdclef-2026'
audio_data = []
labels = []

class_csv = pd.read_csv(os.path.join(BASE, 'taxonomy.csv'))
classes = class_csv['primary_label'].tolist()

train_csv = pd.read_csv(os.path.join(BASE, 'train.csv'))
for index, row in train_csv.iterrows():
    filename = row['filename'].replace('.ogg', '.npy')
    audio_path = os.path.join('mel-spectrograms', filename)
    label = row['primary_label']
    spectrogram = np.load(os.path.join('mel-spectrograms', filename))
    frame_length = 128
    num_frames = spectrogram.shape[1]
    parts = []
    for i in range(0, num_frames, frame_length):
        part = spectrogram[:, i:i + frame_length]
        if part.shape[1] < frame_length:
            pad_width = frame_length - part.shape[1]
            part = np.pad(part, ((0, 0), (0, pad_width)), mode='constant')
        audio_data.append(part)
    labels.append(label)
    if index % 100 == 0:
        print(index)

label_encoder = sklearn.preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(audio_data, encoded_labels, test_size=0.2, random_state=42)

# Ensure all spectrograms have the same shape
max_length = max([spec.shape[0] for spec in audio_data])
X_train = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_train]
X_test = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_test]

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

model = tf.keras.models.Sequential([
  layers.Conv2D(8, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(512, activation='relu'),
  layers.Dense(len(classes), activation='softmax', name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 25
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

test_accuracy=model.evaluate(X_test, y_test,verbose=0)
print(test_accuracy[1])

model.save('audio_classification_model.h5')