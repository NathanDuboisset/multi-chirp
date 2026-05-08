"""
mel_models.py — CNN models operating on log-mel spectrograms.

All models use the same mel preprocessing as utils.py (16 kHz, 80 bins,
184 frames) and output Dense(n_classes, activation="sigmoid") for
multi-label classification, compiled with binary cross-entropy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
N_CHANNELS = 16
HIDDEN_SIZE = 64


def build_cnn2d(
    n_classes: int, input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE
) -> Model:
    """Standard 2-D CNN on log-mel spectrogram."""
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.BatchNormalization(name="input_norm")(inp)
    x = layers.Conv2D(N_CHANNELS, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(N_CHANNELS, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return _compile(Model(inp, out, name="cnn2d_mel"))


def get_mel_model(name: str) -> Callable[[int], Model]:
    if name == "mel_cnn":
        return build_cnn2d
    else:
        raise ValueError(f"Unknown model: {name}")
