from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Callable, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras

# GLOBAL CONFIGURATION / PATHS

SEED = 3407


def set_global_seed(seed: int = SEED) -> None:
    """Set Python, NumPy and TensorFlow RNG seeds."""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_tf_runtime() -> None:
    """Configure TensorFlow runtime flags and GPU memory growth."""

    # Disable XLA / JIT and oneDNN for deterministic, CPU-only behaviour
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Limit GPU memory growth if GPUs are available (mirrors cnn_time.ipynb)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Best-effort; don't crash if this fails on some platforms.
                pass


# Repository paths
REPO_ROOT = Path.cwd().parent
MODELS_DIR = REPO_ROOT / "models"
SRC_DIR = REPO_ROOT / "src"
DATASET_ROOT = REPO_ROOT / "dataset"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPaths:
    """Convenience bundle for per-model output paths."""

    out_tflite: Path
    out_audio_rs: Path


def get_paths(model_stem: str) -> ModelPaths:
    """Return standard output paths for a given model name (without extension)."""

    out_tflite = MODELS_DIR / f"{model_stem}.tflite"
    out_audio_rs = SRC_DIR / "audio_sample.rs"
    return ModelPaths(out_tflite=out_tflite, out_audio_rs=out_audio_rs)


# AUDIO GEOMETRY

SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 3.0
FRAME_LENGTH = 1024
FRAME_STEP = 256

# Full clip = 3 s at 16 kHz = 48000 samples. Source of truth for every
TARGET_AUDIO_LEN = int(SAMPLE_RATE * CLIP_DURATION_SEC)
# Back-compat aliases — both equal TARGET_AUDIO_LEN now.
TARGET_AUDIO_LEN_TIME = TARGET_AUDIO_LEN
TARGET_AUDIO_LEN_MEL = TARGET_AUDIO_LEN

# CNN mel
FFT_LENGTH_MEL = FRAME_LENGTH
NUM_MEL_BINS_MEL = 80
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 8000.0
# Number of STFT frames produced by create_log_mel_spectrogram on
# TARGET_AUDIO_LEN samples: 1 + (L - FRAME_LENGTH) // FRAME_STEP.
TARGET_FRAMES_MEL = 1 + (TARGET_AUDIO_LEN - FRAME_LENGTH) // FRAME_STEP
TARGET_FRAMES_TIME = TARGET_FRAMES_MEL  # kept for any legacy reference


# DATASET HELPERS
def make_audio_datasets(
    root: Path = DATASET_ROOT,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 32,
    seed: int = SEED,
    class_names: list[str] | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Create raw train/val/test datasets from the TinyChirp directory layout."""

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
    """Match cnn_time.ipynb: crop/pad to TARGET_AUDIO_LEN_TIME and add channel."""

    # audio: [batch, time, 1] from keras audio_dataset_from_directory
    audio = tf.squeeze(audio, axis=-1)  # [batch, time]
    audio = audio[:, :TARGET_AUDIO_LEN_TIME]
    current_len = tf.shape(audio)[1]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_TIME - current_len)
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    audio = tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_TIME])
    audio = tf.expand_dims(audio, axis=-1)  # [batch, time, 1]
    return audio


def time_to_features(audio: tf.Tensor, label: tf.Tensor):
    audio = fix_audio_length_time(audio)
    return audio, label


def make_time_datasets(
    root: Path = DATASET_ROOT,
    batch_size: int = 32,
    seed: int = SEED,
    class_names: list[str] | None = None,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    np.ndarray,
]:
    """Return (train_ds, val_ds, test_ds, label_names, steps_per_epoch, val_steps, test_steps)
    for the CNN-time model.
    """

    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root,
        sample_rate=SAMPLE_RATE,
        batch_size=batch_size,
        seed=seed,
        class_names=class_names,
    )

    # Compute finite cardinalities before repeating.
    train_ds = train_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)
    val_ds = val_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)
    test_ds = test_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)

    return train_ds, val_ds, test_ds, label_names


# MEL HELPERS
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
    """Build the Rust-compatible mel filterbank matrix and return (fft_bins, matrix)."""

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

    rust_mel_matrix = tf.constant(rust_mel_matrix_np, dtype=tf.float32)
    return fft_bins, rust_mel_matrix


_RUST_FFT_BINS, RUST_MEL_MATRIX = build_rust_mel_matrix(
    num_mel_bins=NUM_MEL_BINS_MEL,
    fft_length_mel=FFT_LENGTH_MEL,
)


def fix_audio_length_mel(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[:, :TARGET_AUDIO_LEN_MEL]
    current_len = tf.shape(audio)[1]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_MEL - current_len)
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    audio = tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_MEL])
    return audio


def create_log_mel_spectrogram(audio: tf.Tensor) -> tf.Tensor:
    stfts = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH_MEL,
        window_fn=lambda n, dtype: tf.signal.hann_window(
            n, periodic=False, dtype=dtype
        ),
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
    """Return (train_ds, val_ds, test_ds, label_names) for CNN-mel model.

    `num_mel_bins` and `target_frames` are threaded through the tf.data
    pipeline for shape checking, but must currently match the Rust-side
    configuration (NUM_MEL_BINS_MEL / TARGET_FRAMES_MEL).
    """

    if num_mel_bins != NUM_MEL_BINS_MEL or target_frames != TARGET_FRAMES_MEL:
        raise ValueError(
            "make_mel_datasets currently requires num_mel_bins == NUM_MEL_BINS_MEL "
            "and target_frames == TARGET_FRAMES_MEL to stay in sync with Rust."
        )

    def _mel_to_features(audio: tf.Tensor, label: tf.Tensor):
        audio_fixed = fix_audio_length_mel(audio)
        spec = create_log_mel_spectrogram(audio_fixed)
        spec = tf.ensure_shape(spec, [None, target_frames, num_mel_bins])
        spec = tf.expand_dims(spec, axis=-1)
        return spec, label

    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root,
        sample_rate=SAMPLE_RATE,
        batch_size=batch_size,
        seed=seed,
        class_names=None,
    )

    train_ds = train_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    test_ds = test_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names


# TFLite export helpers


def build_representative_batches(
    dataset: tf.data.Dataset,
    target_len: int,
    take: int = 100,
) -> List[np.ndarray]:
    """Collect a small set of representative samples for quantization."""

    batches: List[np.ndarray] = []
    for x_batch, _ in dataset.unbatch().take(take):
        sample = x_batch.numpy().astype(np.float32)
        sample = np.reshape(sample, (1, target_len, 1))
        batches.append(sample)
    return batches


def representative_dataset_from_batches(
    batches: Iterable[np.ndarray],
) -> Callable[[], Iterable[List[np.ndarray]]]:
    """Create a TFLite representative_dataset generator from prebuilt batches."""

    def gen():
        for sample in batches:
            yield [sample]

    return gen


def export_int8_tflite_from_saved_model(
    saved_model_dir: str,
    out_tflite: Path,
    rep_batches: Iterable[np.ndarray],
) -> None:
    """Export an INT8 TFLite model from a SavedModel with representative data."""

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_from_batches(rep_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    out_tflite.write_bytes(tflite_bytes)


def export_keras_model_to_int8_tflite(
    model: keras.Model,
    rep_batches: Iterable[np.ndarray],
    out_tflite: Path,
    tmp_dir: str = "temp_saved_model",
) -> None:
    """High-level helper that mirrors the export flow used in the notebooks."""

    model.export(tmp_dir)
    export_int8_tflite_from_saved_model(tmp_dir, out_tflite, rep_batches)


# Rust audio_sample.rs helpers
TestClip = Tuple[str, np.ndarray, str]


def collect_test_clips_for_rs(
    root: Path,
    sample_rate: int,
    target_len: int,
    num_per_label: int = 2,
) -> List[TestClip]:
    """Collect a small, fixed set of test clips for Rust."""

    raw_sample_ds = keras.utils.audio_dataset_from_directory(
        root,
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=1,
        shuffle=False,
    )

    clips_by_label: dict[int, List[np.ndarray]] = {}
    for audio_batch, label_batch in raw_sample_ds.unbatch():
        label_idx = int(label_batch.numpy())
        if label_idx not in clips_by_label:
            clips_by_label[label_idx] = []
        if len(clips_by_label[label_idx]) < num_per_label:
            fixed = (
                fix_audio_length_time(tf.expand_dims(audio_batch, 0))[0]
                .numpy()
                .astype(np.float32)
            )
            clips_by_label[label_idx].append(fixed)
        if (
            all(len(v) >= num_per_label for v in clips_by_label.values())
            and len(clips_by_label) >= 2
        ):
            break

    if len(clips_by_label) < 2:
        raise RuntimeError("Expected at least two labels in testing dataset.")

    # Assumes binary classification with labels 0 and 1, matching existing notebooks.
    ordered: List[TestClip] = []
    for i in range(num_per_label):
        for label_idx in sorted(clips_by_label.keys()):
            label_name = "target" if label_idx == 1 else "non_target"
            audio = clips_by_label[label_idx][i]
            rel_path = f"dataset/testing/{label_name}/sample_{i + 1}.wav"
            ordered.append((label_name, audio, rel_path))

    return ordered


def write_audio_sample_rs(
    out_path: Path,
    clips: List[TestClip],
    sample_rate: int,
    generator_name: str = "building_tensorflow/cnn_time.ipynb",
) -> None:
    """Write a Rust audio_sample.rs file compatible with the TinyChirp runner."""

    rs: List[str] = []
    rs.append(f"// Generated by {generator_name}\n")
    rs.append(f"pub const SAMPLE_RATE: usize = {sample_rate};\n\n")
    rs.append("pub struct TestClip {\n")
    rs.append("    pub expected_label: &'static str,\n")
    rs.append("    pub source_file: &'static str,\n")
    rs.append("    pub audio: &'static [f32],\n")
    rs.append("}\n\n")

    for i, (_label, audio, _rel_path) in enumerate(clips, 1):
        audio_vals = ", ".join(f"{float(v):.8f}" for v in audio)
        rs.append(f"pub const CLIP_{i}: &[f32] = &[{audio_vals}];\n\n")

    rs.append(f"pub const TEST_CLIPS: [TestClip; {len(clips)}] = [\n")
    for i, (label, _audio, rel_path) in enumerate(clips, 1):
        rs.append("    TestClip {\n")
        rs.append(f'        expected_label: "{label}",\n')
        rs.append(f'        source_file: "{rel_path}",\n')
        rs.append(f"        audio: CLIP_{i},\n")
        rs.append("    },\n")
    rs.append("];\n")

    out_path.write_text("".join(rs), encoding="utf-8")
