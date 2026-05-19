"""TinyChirp-style 2-D CNNs on log-mel spectrogram."""

from __future__ import annotations

from keras import Model, layers

from building.models._common import MEL_INPUT_SHAPE, compile_model

N_CHANNELS = 4
HIDDEN_SIZE = 8
DROPOUT = 0.3
CONV_KERNEL_SIZE = (3, 3)
CONV_POOL_SIZE = (2, 2)


def build(n_classes: int, input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE) -> Model:
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(N_CHANNELS, CONV_KERNEL_SIZE, activation="relu")(inp)
    x = layers.AveragePooling2D(CONV_POOL_SIZE)(x)
    x = layers.Conv2D(N_CHANNELS, CONV_KERNEL_SIZE, activation="relu")(x)
    x = layers.AveragePooling2D(CONV_POOL_SIZE)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="cnn2d_mel"))



N_CHANNELS_2 = 8
N_PATTERNS_2 = 8
PATTERN_TIME_2 = 15
POOL_MODE_2 = "max"  # "max" or "avg"
CONV_POOL_SIZE_2 = (2, 1)
SPATIAL_DROPOUT_2 = 0.1


def build_2(n_classes: int, input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE) -> Model:
    freq_after_pool = (input_shape[1] - CONV_KERNEL_SIZE[1] + 1) // CONV_POOL_SIZE_2[1]
    freq_after_pool = (freq_after_pool - CONV_KERNEL_SIZE[1] + 1) // CONV_POOL_SIZE[1]

    inp = layers.Input(shape=input_shape, name="mel_spectrogram")
    x = layers.Conv2D(N_CHANNELS_2, CONV_KERNEL_SIZE, activation="relu")(inp)
    x = layers.MaxPooling2D(CONV_POOL_SIZE_2)(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT_2)(x)
    x = layers.Conv2D(N_CHANNELS_2, CONV_KERNEL_SIZE, activation="relu")(x)
    x = layers.MaxPooling2D(CONV_POOL_SIZE)(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT_2)(x)

    usefreq = freq_after_pool //2

    x = layers.Conv2D(
        N_PATTERNS_2,
        (PATTERN_TIME_2, usefreq),
        activation="relu",
        padding="valid",
        name="chirp_patterns",
    )(x)

    if POOL_MODE_2 == "max":
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="cnn2d_mel_2"))
