"""Audio augmentations for bird-call training, built on `audiomentations`."""

from __future__ import annotations

import numpy as np
import pyrootutils
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AirAbsorption,
    ClippingDistortion,
    Compose,
    Gain,
    Mp3Compression,
    PolarityInversion,
    Shift,
    SomeOf,
    TimeMask,
)

from .download import TARGET_SAMPLE_RATE

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)
AUDIOSET_DIR = ROOT / "raw_dataset" / "audioset"

P_SHIFT = 0.5
P_GAIN = 0.5
P_POLARITY = 0.5
P_NOISE = 0.5
P_DROPOUT = 0.25
P_CHANNEL = 0.25
P_DISTORTION = 0.10

MIN_BACKGROUND_SNR_DB = 8.0
MAX_BACKGROUND_SNR_DB = 25.0

SHIFT_FRACTION = 0.1
GAIN_DB = 4.0
GAUSSIAN_NOISE_AMPLITUDE = (1e-4, 3e-3)
TIME_MASK_BAND = (0.02, 0.08)
AIR_DISTANCE_M = (1.0, 10.0)
MP3_BITRATE = (64, 160)
CLIPPING_PERCENTILE = (0, 5)


def _has_wavs(path: Path) -> bool:
    if not path.exists():
        return False
    return next(path.rglob("*.wav"), None) is not None


noise_choices: list = [
    AddGaussianNoise(
        min_amplitude=GAUSSIAN_NOISE_AMPLITUDE[0],
        max_amplitude=GAUSSIAN_NOISE_AMPLITUDE[1],
        p=1.0,
    ),
]
if _has_wavs(AUDIOSET_DIR):
    noise_choices.append(
        AddBackgroundNoise(
            sounds_path=str(AUDIOSET_DIR),
            min_snr_db=MIN_BACKGROUND_SNR_DB,
            max_snr_db=MAX_BACKGROUND_SNR_DB,
            noise_rms="relative",
            p=1.0,
        )
    )

augmenter = Compose(
    [
        Shift(
            min_shift=-SHIFT_FRACTION,
            max_shift=SHIFT_FRACTION,
            shift_unit="fraction",
            rollover=True,
            p=P_SHIFT,
        ),
        Gain(min_gain_db=-GAIN_DB, max_gain_db=GAIN_DB, p=P_GAIN),
        PolarityInversion(p=P_POLARITY),
        SomeOf((1, 2), noise_choices, p=P_NOISE),
        TimeMask(
            min_band_part=TIME_MASK_BAND[0],
            max_band_part=TIME_MASK_BAND[1],
            fade_duration=0.005,
            p=P_DROPOUT,
        ),
        SomeOf(
            (1, 1),
            [
                AirAbsorption(
                    min_distance=AIR_DISTANCE_M[0],
                    max_distance=AIR_DISTANCE_M[1],
                    p=1.0,
                ),
                Mp3Compression(
                    min_bitrate=MP3_BITRATE[0],
                    max_bitrate=MP3_BITRATE[1],
                    backend="pydub",
                    p=1.0,
                ),
            ],
            p=P_CHANNEL,
        ),
        ClippingDistortion(
            min_percentile_threshold=CLIPPING_PERCENTILE[0],
            max_percentile_threshold=CLIPPING_PERCENTILE[1],
            p=P_DISTORTION,
        ),
    ],
    shuffle=False,
)


def augment(audio: np.ndarray) -> np.ndarray:
    return augmenter(samples=audio.astype(np.float32, copy=False), sample_rate=TARGET_SAMPLE_RATE)


def augment_tf(audio):
    """tf.data-compatible wrapper: same augment pipeline, applied per sample."""
    import tensorflow as tf

    def _np_augment(arr: np.ndarray) -> np.ndarray:
        return augment(arr).astype(np.float32, copy=False)

    out = tf.numpy_function(_np_augment, [audio], tf.float32)
    out.set_shape(audio.shape)
    return out
