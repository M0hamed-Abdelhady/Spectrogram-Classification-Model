import numpy as np
import tensorflow as tf


def normalize(spectrogram, max_value=10000) -> np.array:
    spectrogram = np.clip(spectrogram, 0, max_value)
    spectrogram = spectrogram / max_value
    return spectrogram


def adjust(spectrogram, target_shape=(1998, 101)) -> np.array:
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


def make_prediction(spectrogram):
    spectrogram = adjust(spectrogram)
    spectrogram = normalize(spectrogram)
    spectrogram = tf.reshape(spectrogram, (1, 1998, 101))
    model = tf.keras.models.load_model("AudioSpectrogram.h5")
    prediction = model.predict(spectrogram)
    return prediction
