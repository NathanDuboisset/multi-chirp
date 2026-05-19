"""Lightweight 1-D CNN on raw waveform."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import TARGET_AUDIO_LEN, compile_model

CONV1_FILTERS = 4
CONV1_KERNEL = 3
POOL_SIZE = 2
POOL_STRIDES = 2
CONV2_FILTERS = 8
CONV2_KERNEL = 3
DENSE_HIDDEN = 64


def build(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Conv1D(CONV1_FILTERS, CONV1_KERNEL, activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDES)(x)
    x = layers.Conv1D(CONV2_FILTERS, CONV2_KERNEL, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(DENSE_HIDDEN, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="cnn1d"))
