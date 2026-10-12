import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, Input, ReLU, MaxPool2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
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
    frame_length = 216
    num_frames = spectrogram.shape[1]
    parts = []
    for i in range(0, num_frames, frame_length):
        part = spectrogram[i:i + frame_length]
        if len(part) == frame_length:
            audio_data.append(part)
            labels.append(label)
    if index % 100 == 0:
        print(index)

label_encoder = sklearn.preprocessing.LabelEncoder()
minmax_scaler = sklearn.preprocessing.MinMaxScaler()
audio_data = np.array(audio_data)
n_samples, n_timesteps, n_features = audio_data.shape
audio_data = audio_data.reshape(-1, n_features)
audio_data = minmax_scaler.fit_transform(audio_data)
audio_data = audio_data.reshape(n_samples, n_timesteps, n_features)
encoded_labels = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(audio_data, encoded_labels, test_size=0.2, random_state=42)

# Ensure all spectrograms have the same shape
max_length = max([spec.shape[0] for spec in audio_data])
X_train = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_train]
X_test = [np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant') for spec in X_test]

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 10
subtract_pixel_mean = True
depth = 29

img_height = 224
img_width = 224

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate:', lr)
    return lr


def densenet(input_shape, n_classes, filters=32):
    # batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x

    def dense_block(x, repetition):

        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x

    def transition_layer(x):

        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AveragePooling2D(2, strides=2, padding='same')(x)
        return x

    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

checkpoint = ModelCheckpoint(filepath='resnet.keras',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]

model = densenet(input_shape=(216, 128, 1), depth=depth, num_classes=num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks)
