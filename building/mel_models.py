"""
mel_models.py — CNN models operating on log-mel spectrograms.

All models use the same mel preprocessing as utils.py (16 kHz, 80 bins,
184 frames) and output Dense(n_classes, activation="sigmoid") for
multi-label classification, compiled with binary cross-entropy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import keras
else:
    import tensorflow as tf
    keras = tf.keras

from keras import layers, Model

# Must match utils.py
NUM_MEL_BINS = 80
TARGET_FRAMES = 184  # time frames
MEL_INPUT_SHAPE = (TARGET_FRAMES, NUM_MEL_BINS, 1)


def _compile(model: Model) -> Model:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model

CONV_FILTER_SIZE = 3
N_CHANNELS = 4
HIDDEN_SIZE = 8

def build_cnn2d(n_classes: int, input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE) -> Model:
    """Standard 2-D CNN on log-mel spectrogram."""
    target_frame,num_mel_bins, _ = input_shape
    end_of_conv1_s1 = (target_frame - CONV_FILTER_SIZE + 1) // 2
    end_of_conv2_s1 = (end_of_conv1_s1 - CONV_FILTER_SIZE + 1) // 2
    end_of_conv1_s2 = (num_mel_bins - CONV_FILTER_SIZE + 1) // 2
    end_of_conv2_s2 = (end_of_conv1_s2 - CONV_FILTER_SIZE + 1) // 2

    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(N_CHANNELS, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(N_CHANNELS, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Reshape((end_of_conv2_s2, end_of_conv2_s1, N_CHANNELS))(x)
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return _compile(Model(inp, out, name="cnn2d_mel"))

