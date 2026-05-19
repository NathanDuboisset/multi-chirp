"""Raw-audio dataset loaders and length-fixing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import tensorflow as tf

from .constants import SAMPLE_RATE, SEED, TARGET_AUDIO_LEN_MEL, TARGET_AUDIO_LEN_TIME
from .paths import DATASET_ROOT

if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras


def make_audio_datasets(
    root: Path = DATASET_ROOT,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 32,
    seed: int = SEED,
    class_names: list[str] | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    train_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "training",
        labels="inferred",
        class_names=class_names,
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "validation",
        labels="inferred",
        class_names=class_names,
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "testing",
        labels="inferred",
        class_names=class_names,
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=False,
    )

    label_names = np.array(train_ds_raw.class_names)
    return train_ds_raw, val_ds_raw, test_ds_raw, label_names


def fix_audio_length_time(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[:, :TARGET_AUDIO_LEN_TIME]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_TIME - tf.shape(audio)[1])
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    audio = tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_TIME])
    return tf.expand_dims(audio, axis=-1)


def fix_audio_length_mel(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[:, :TARGET_AUDIO_LEN_MEL]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_MEL - tf.shape(audio)[1])
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    return tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_MEL])


def time_to_features(audio: tf.Tensor, label: tf.Tensor):
    return fix_audio_length_time(audio), label


def make_time_datasets(
    root: Path = DATASET_ROOT,
    batch_size: int = 32,
    seed: int = SEED,
    class_names: list[str] | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root,
        sample_rate=SAMPLE_RATE,
        batch_size=batch_size,
        seed=seed,
        class_names=class_names,
    )
    train_ds = train_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    val_ds = val_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    test_ds = test_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)
    return train_ds, val_ds, test_ds, label_names
