"""Mel spectrogram: Rust-compatible filterbank + log-mel + dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from .audio_io import fix_audio_length_mel, make_audio_datasets
from .constants import (
    FFT_LENGTH_MEL,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HERTZ,
    NUM_MEL_BINS_MEL,
    SAMPLE_RATE,
    SEED,
    TARGET_FRAMES_MEL,
    UPPER_EDGE_HERTZ,
)
from .paths import DATASET_ROOT


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_rust_mel_matrix(
    num_mel_bins: int,
    fft_length_mel: int,
    frame_length: int = FRAME_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    lower_edge_hz: float = LOWER_EDGE_HERTZ,
    upper_edge_hz: float = UPPER_EDGE_HERTZ,
):
    fft_bins = fft_length_mel // 2

    mel_edges = np.zeros(num_mel_bins + 2, dtype=np.int32)
    low_mel = hz_to_mel(lower_edge_hz)
    high_mel = hz_to_mel(upper_edge_hz)
    for i in range(num_mel_bins + 2):
        frac = i / (num_mel_bins + 1)
        mel = low_mel + frac * (high_mel - low_mel)
        hz = mel_to_hz(mel)
        bin_idx = int(((frame_length + 1.0) * hz) / sample_rate)
        mel_edges[i] = min(bin_idx, fft_bins - 1)

    rust_mel_matrix_np = np.zeros((fft_bins, num_mel_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        left = mel_edges[m]
        center = mel_edges[m + 1]
        right = mel_edges[m + 2]
        for k in range(left, center):
            rust_mel_matrix_np[k, m] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            rust_mel_matrix_np[k, m] = (right - k) / max(right - center, 1)

    return fft_bins, tf.constant(rust_mel_matrix_np, dtype=tf.float32)


_RUST_FFT_BINS, RUST_MEL_MATRIX = build_rust_mel_matrix(
    num_mel_bins=NUM_MEL_BINS_MEL,
    fft_length_mel=FFT_LENGTH_MEL,
)


def create_log_mel_spectrogram(audio: tf.Tensor) -> tf.Tensor:
    stfts = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH_MEL,
        window_fn=lambda n, dtype: tf.signal.hann_window(n, periodic=False, dtype=dtype),
    )
    spectrograms = tf.abs(stfts)[..., :_RUST_FFT_BINS]
    mel_spectrograms = tf.tensordot(spectrograms, RUST_MEL_MATRIX, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate([NUM_MEL_BINS_MEL]))
    return tf.math.log(mel_spectrograms + 1e-6)


def make_mel_datasets(
    root: Path = DATASET_ROOT,
    batch_size: int = 32,
    seed: int = SEED,
    num_mel_bins: int = NUM_MEL_BINS_MEL,
    target_frames: int = TARGET_FRAMES_MEL,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    if num_mel_bins != NUM_MEL_BINS_MEL or target_frames != TARGET_FRAMES_MEL:
        raise ValueError(
            "make_mel_datasets currently requires num_mel_bins == NUM_MEL_BINS_MEL "
            "and target_frames == TARGET_FRAMES_MEL to stay in sync with Rust."
        )

    def _mel_to_features(audio: tf.Tensor, label: tf.Tensor):
        audio_fixed = fix_audio_length_mel(audio)
        spec = create_log_mel_spectrogram(audio_fixed)
        spec = tf.ensure_shape(spec, [None, target_frames, num_mel_bins])
        return tf.expand_dims(spec, axis=-1), label

    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root,
        sample_rate=SAMPLE_RATE,
        batch_size=batch_size,
        seed=seed,
        class_names=None,
    )
    train_ds = train_raw.map(_mel_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.map(_mel_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_raw.map(_mel_to_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, label_names
