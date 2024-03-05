import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.layers import Reshape


def normalize_spectrogram(spectrogram, max_value=10000) -> np.array:
    spectrogram = np.clip(spectrogram, 0, max_value)
    spectrogram = spectrogram / max_value
    return spectrogram


def adjust_spectrogram(spectrogram, target_shape):
    height, width = target_shape
    current_height, current_width = spectrogram.shape

    if current_height < height:
        pad_top = (height - current_height) // 2
        pad_bottom = height - current_height - pad_top
        spectrogram = np.pad(spectrogram, ((pad_top, pad_bottom), (0, 0)), mode='edge')
    elif current_height > height:
        crop_top = (current_height - height) // 2
        crop_bottom = current_height - height - crop_top
        spectrogram = spectrogram[crop_top:current_height - crop_bottom, :]

    if current_width < width:
        pad_left = (width - current_width) // 2
        pad_right = width - current_width - pad_left
        spectrogram = np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode='edge')
    elif current_width > width:
        crop_left = (current_width - width) // 2
        crop_right = current_width - width - crop_left
        spectrogram = spectrogram[:, crop_left:current_width - crop_right]

    return spectrogram


real_data = np.load('dataset/real.npz')
fake_data = np.load('dataset/fake.npz')

real_labels = np.array([1] * 750)
fake_labels = np.array([0] * 750)

x_real = np.array([normalize_spectrogram(s) for s in real_data['x']])
x_fake = np.array([normalize_spectrogram(s) for s in fake_data['x']])

x = np.concatenate((x_real, x_fake))
y = np.concatenate((real_labels, fake_labels))

x_train, x_test = np.concatenate((x_real[:600], x_fake[:600])), np.concatenate((x_real[600:], x_fake[600:]))
y_train, y_test = np.concatenate((real_labels[:600], fake_labels[:600])), np.concatenate((real_labels[600:], fake_labels[600:]))


# Build the model
model = Sequential([
    Reshape((1998, 101, 1), input_shape=(1998, 101)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

model.save('spectrogram.h5')
