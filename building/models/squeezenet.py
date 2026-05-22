"""SqueezeNet 1.0 (TinyChirp paper): 2-D on log-mel, 1-D on raw waveform.

References:
    Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and < 0.5 MB model size", arXiv:1602.07360.
    Huang et al., "TinyChirp: Bird Song Recognition Using TinyML Models on
    Low-power Wireless Acoustic Sensors", IEEE IS2 2024 (arXiv:2407.21453).

SqueezeNet-Mel: standard SqueezeNet 1.0 with input (184, 80, 1) and n_classes
output channels at conv10. Parameter count: ~727K (matches paper Table II).

SqueezeNet-Time: same backbone with Conv2D -> Conv1D and filter counts scaled
by 0.3 (~70% reduction, per paper Table III). Parameter count: ~31K.
"""

from __future__ import annotations

from math import floor

from keras import Model, layers

from building.models._common import MEL_INPUT_SHAPE, TARGET_AUDIO_LEN, compile_model

# SqueezeNet 1.0 fire module filter counts (s, e1, e3) per the Iandola paper.
_FIRES = [
    (16, 64, 64),   # fire2
    (16, 64, 64),   # fire3
    (32, 128, 128), # fire4
    (32, 128, 128), # fire5
    (48, 192, 192), # fire6
    (48, 192, 192), # fire7
    (64, 256, 256), # fire8
    (64, 256, 256), # fire9
]
_STEM_FILTERS = 96
_STEM_KERNEL = 7
_STEM_STRIDES = 2
_POOL_KERNEL = 3
_POOL_STRIDES = 2
_DROPOUT = 0.5
_TIME_SCALE = 0.3  # 70% reduction for the time-domain variant.


def _fire_2d(x, squeeze: int, expand1: int, expand3: int, name: str):
    s = layers.Conv2D(
        squeeze, (1, 1), activation="relu", padding="same", name=f"{name}_squeeze"
    )(x)
    e1 = layers.Conv2D(
        expand1, (1, 1), activation="relu", padding="same", name=f"{name}_expand1x1"
    )(s)
    e3 = layers.Conv2D(
        expand3, (3, 3), activation="relu", padding="same", name=f"{name}_expand3x3"
    )(s)
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([e1, e3])


def _fire_1d(x, squeeze: int, expand1: int, expand3: int, name: str):
    s = layers.Conv1D(
        squeeze, 1, activation="relu", padding="same", name=f"{name}_squeeze"
    )(x)
    e1 = layers.Conv1D(
        expand1, 1, activation="relu", padding="same", name=f"{name}_expand1"
    )(s)
    e3 = layers.Conv1D(
        expand3, 3, activation="relu", padding="same", name=f"{name}_expand3"
    )(s)
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([e1, e3])


def build_mel(
    n_classes: int,
    input_shape: tuple[int, int, int] = MEL_INPUT_SHAPE,
    dropout: float = _DROPOUT,
) -> Model:
    inp = layers.Input(shape=input_shape, name="mel_spectrogram")

    x = layers.Conv2D(
        _STEM_FILTERS,
        (_STEM_KERNEL, _STEM_KERNEL),
        strides=(_STEM_STRIDES, _STEM_STRIDES),
        padding="same",
        activation="relu",
        name="conv1",
    )(inp)
    x = layers.MaxPooling2D(
        (_POOL_KERNEL, _POOL_KERNEL), strides=(_POOL_STRIDES, _POOL_STRIDES),
        padding="same", name="pool1",
    )(x)

    # SqueezeNet 1.0 places max-pools after fire4 and fire8.
    for idx, (s, e1, e3) in enumerate(_FIRES, start=2):
        x = _fire_2d(x, s, e1, e3, name=f"fire{idx}")
        if idx in (4, 8):
            x = layers.MaxPooling2D(
                (_POOL_KERNEL, _POOL_KERNEL),
                strides=(_POOL_STRIDES, _POOL_STRIDES),
                padding="same",
                name=f"pool{idx}",
            )(x)

    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout9")(x)

    x = layers.Conv2D(
        n_classes, (1, 1), padding="same", activation="relu", name="conv10",
    )(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Activation("sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="squeezenet_mel"))


def build_time(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    dropout: float = _DROPOUT,
) -> Model:
    def scale(n: int) -> int:
        return max(1, floor(_TIME_SCALE * n))

    inp = layers.Input(shape=(input_len, 1), name="audio")

    x = layers.Conv1D(
        scale(_STEM_FILTERS),
        _STEM_KERNEL,
        strides=_STEM_STRIDES,
        padding="same",
        activation="relu",
        name="conv1",
    )(inp)
    x = layers.MaxPooling1D(
        _POOL_KERNEL, strides=_POOL_STRIDES, padding="same", name="pool1",
    )(x)

    for idx, (s, e1, e3) in enumerate(_FIRES, start=2):
        x = _fire_1d(x, scale(s), scale(e1), scale(e3), name=f"fire{idx}")
        if idx in (4, 8):
            x = layers.MaxPooling1D(
                _POOL_KERNEL, strides=_POOL_STRIDES, padding="same",
                name=f"pool{idx}",
            )(x)

    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout9")(x)

    x = layers.Conv1D(
        n_classes, 1, padding="same", activation="relu", name="conv10",
    )(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    out = layers.Activation("sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="squeezenet_time"))
