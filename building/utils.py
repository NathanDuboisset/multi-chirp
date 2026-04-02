from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Any, Callable, Iterable, List, Tuple, TYPE_CHECKING, cast

import numpy as np
import tensorflow as tf
from tqdm import tqdm
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
FRAME_LENGTH = 1024
FRAME_STEP = 256

# CNN TIME GEOMETRY
TARGET_FRAMES_TIME = 184
TARGET_AUDIO_LEN_TIME = (TARGET_FRAMES_TIME - 1) * FRAME_STEP + FRAME_LENGTH


# CNN mel
FFT_LENGTH_MEL = FRAME_LENGTH
NUM_MEL_BINS_MEL = 80 
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 8000.0
TARGET_FRAMES_MEL = 184 
TARGET_AUDIO_LEN_MEL = (TARGET_FRAMES_MEL - 1) * FRAME_STEP + FRAME_LENGTH

# DATASET HELPERS
def make_audio_datasets(
    root: Path = DATASET_ROOT,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 32,
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Create raw train/val/test datasets from the TinyChirp directory layout."""

    train_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "training",
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "validation",
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "testing",
        labels="inferred",
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
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])
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
        root=root, sample_rate=SAMPLE_RATE, batch_size=batch_size, seed=seed
    )

    # Compute finite cardinalities before repeating.
    train_ds = (
        train_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(2)
    )
    val_ds = (
        val_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(2)
    )
    test_ds = (
        test_raw.map(time_to_features, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(2)
    )

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
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])
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
        root=root, sample_rate=SAMPLE_RATE, batch_size=batch_size, seed=seed
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


# METRICS 


def plot_training_history(
    history: object,
    title: str | None = None,
) -> None:
    """Plot training/validation loss and accuracy side-by-side.

    Works with the return value of `tf.keras.Model.fit(...)` (Keras `History`)
    or with a raw `history.history` dict.
    """

    # Defer plotting dependency; most of this module is "TF only".
    import matplotlib.pyplot as plt

    if hasattr(history, "history"):
        history_dict = cast(dict[str, Any], history.history)
    else:
        history_dict = cast(dict[str, Any], history)

    if not isinstance(history_dict, dict):
        raise ValueError("`history` must be a Keras History object or a dict-like `history.history`.")

    loss_key = "loss"
    val_loss_key = "val_loss"
    if loss_key not in history_dict or val_loss_key not in history_dict:
        raise ValueError(
            "History is missing `loss` and/or `val_loss`. "
            f"Keys: {sorted(history_dict.keys())}"
        )

    # Keras can report accuracy under different names depending on label encoding.
    accuracy_key = next(
        (k for k in ("accuracy", "sparse_categorical_accuracy", "categorical_accuracy") if k in history_dict),
        None,
    )
    if accuracy_key is None:
        raise ValueError(
            "Could not find an accuracy key in history. "
            f"Keys: {sorted(history_dict.keys())}"
        )

    val_accuracy_key = f"val_{accuracy_key}"
    if val_accuracy_key not in history_dict:
        raise ValueError(
            f"Missing `{val_accuracy_key}` in history. Keys: {sorted(history_dict.keys())}"
        )

    def _to_1d(x: Any) -> np.ndarray:
        arr = np.asarray(x).reshape(-1)
        return arr

    train_loss = _to_1d(history_dict[loss_key])
    val_loss = _to_1d(history_dict[val_loss_key])
    train_acc = _to_1d(history_dict[accuracy_key])
    val_acc = _to_1d(history_dict[val_accuracy_key])

    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    if title is None:
        title = "Training History"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def evaluate_binary_classifier(
    model: "keras.Model",
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    display: bool = True,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy, precision, recall, F2, AUC and ROC curve for a model.

    Assumes a binary classification task with labels in {0, 1}.
    The model is expected to output either:
      * a single logit/probability per example (shape [..., 1]), or
      * two logits/probabilities for classes 0 and 1 (shape [..., 2]).

    If `test_dataset` is provided, the function will evaluate both:
      - Training: prints accuracy/precision/recall/F2 and (when `display=True`) AUC + ROC plot
      - Test: prints accuracy/precision/recall/F2 (and does not plot ROC by default)
    """

    def _evaluate_one(ds: tf.data.Dataset, compute_roc: bool, find_best_f2_threshold: bool = False) -> dict:
        y_true_list: list[np.ndarray] = []
        y_score_list: list[np.ndarray] = []

        for i, (x_batch, y_batch) in tqdm(enumerate(ds)):
            preds = model(x_batch, training=False)
            preds_np = preds.numpy()
            y_np = y_batch.numpy()

            # Flatten labels to 1D
            y_np = np.reshape(y_np, (-1,))

            # Convert model outputs to probability of the positive class.
            if preds_np.shape[-1] == 1:
                # Single logit or probability.
                probs = tf.math.sigmoid(preds_np).numpy()
                probs = np.reshape(probs, (-1,))
            elif preds_np.shape[-1] == 2:
                # Two-class logits/probabilities; take class-1 probability.
                probs = tf.nn.softmax(preds_np, axis=-1).numpy()[..., 1]
                probs = np.reshape(probs, (-1,))
            else:
                raise ValueError(
                    "evaluate_binary_classifier expects model outputs of shape "
                    "[..., 1] or [..., 2] for binary classification."
                )

            y_true_list.append(y_np)
            y_score_list.append(probs)

        if not y_true_list:
            raise ValueError("Dataset appears to be empty; no batches were processed.")

        y_true = np.concatenate(y_true_list, axis=0)
        y_score = np.concatenate(y_score_list, axis=0)

        # Select the threshold that maximises F2 by sweeping all unique score values.
        if find_best_f2_threshold:
            best_thr = threshold
            best_f2_thr = -1.0
            beta = 2.0
            for thr in np.unique(y_score):
                y_pred_thr = (y_score >= thr).astype(np.int32)
                tp_thr = np.sum((y_true == 1) & (y_pred_thr == 1))
                fp_thr = np.sum((y_true == 0) & (y_pred_thr == 1))
                fn_thr = np.sum((y_true == 1) & (y_pred_thr == 0))
                prec_thr = tp_thr / (tp_thr + fp_thr) if (tp_thr + fp_thr) > 0 else 0.0
                rec_thr = tp_thr / (tp_thr + fn_thr) if (tp_thr + fn_thr) > 0 else 0.0
                denom_thr = (beta**2) * prec_thr + rec_thr
                f2_thr = (1.0 + beta**2) * prec_thr * rec_thr / denom_thr if denom_thr > 0 else 0.0
                if f2_thr > best_f2_thr:
                    best_f2_thr = f2_thr
                    best_thr = thr
            applied_threshold = float(best_thr)
        else:
            applied_threshold = threshold

        # Predicted labels via threshold
        y_pred = (y_score >= applied_threshold).astype(np.int32)

        # Confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        beta = 2.0
        denom = (beta * beta) * precision + recall
        f2 = (1.0 + beta * beta) * precision * recall / denom if denom > 0 else 0.0

        out: dict[str, float | None | np.ndarray] = {
            "threshold": applied_threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f2": float(f2),
        }

        if not compute_roc:
            out.update(
                {
                    "auc": None,
                    "roc_fpr": None,
                    "roc_tpr": None,
                    "roc_thresholds": None,
                }
            )
            return out

        # ROC curve and AUC
        # Sort by decreasing score.
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        # Unique thresholds (including edge cases)
        thresholds_arr = np.r_[np.inf, np.unique(y_score_sorted)[::-1], -np.inf]

        tpr_list = []
        fpr_list = []

        P = np.sum(y_true_sorted == 1)
        N = np.sum(y_true_sorted == 0)

        for thr in thresholds_arr:
            y_pred_thr = (y_score_sorted >= thr).astype(np.int32)
            tp_thr = np.sum((y_true_sorted == 1) & (y_pred_thr == 1))
            fp_thr = np.sum((y_true_sorted == 0) & (y_pred_thr == 1))

            tpr = tp_thr / P if P > 0 else 0.0
            fpr = fp_thr / N if N > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)

        # Sort by FPR for a well-defined curve, then integrate.
        order_roc = np.argsort(fpr_arr)
        fpr_arr = fpr_arr[order_roc]
        tpr_arr = tpr_arr[order_roc]

        auc = np.trapz(tpr_arr, fpr_arr)
        out.update(
            {
                "auc": float(auc),
                "roc_fpr": fpr_arr,
                "roc_tpr": tpr_arr,
                "roc_thresholds": thresholds_arr[order_roc],
            }
        )
        return out

    # Evaluate training.
    # Always compute ROC/AUC for training so callers can plot from returned arrays.
    train_metrics = _evaluate_one(train_dataset, compute_roc=True)
    # For the test set, sweep all unique score values and pick the threshold that
    # maximises the F2 score rather than relying on the fixed default.
    test_metrics = _evaluate_one(test_dataset, compute_roc=False, find_best_f2_threshold=True)

    if display:

        print("=== Binary classifier metrics : TEST SET===")
        print(f"Threshold: {test_metrics['threshold']:.4f}  (best F2 threshold)")
        print(f"Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall   : {test_metrics['recall']:.4f}")
        print(f"F2 score : {test_metrics['f2']:.4f}")

        print("=== Binary classifier metrics : TRAIN SET===")
        print(f"AUC      : {train_metrics['auc']:.4f}")

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(train_metrics["roc_fpr"], train_metrics["roc_tpr"], label=f"AUC = {train_metrics['auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    return {"train": train_metrics, "test": test_metrics}




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
            fixed = fix_audio_length_time(
                tf.expand_dims(audio_batch, 0)
            )[0].numpy().astype(np.float32)  # ty:ignore[not-subscriptable]
            clips_by_label[label_idx].append(fixed)
        if all(len(v) >= num_per_label for v in clips_by_label.values()) and len(
            clips_by_label
        ) >= 2:
            break

    if len(clips_by_label) < 2:
        raise RuntimeError("Expected at least two labels in testing dataset.")

    # Assumes binary classification with labels 0 and 1, matching existing notebooks.
    ordered: List[TestClip] = []
    for i in range(num_per_label):
        for label_idx in sorted(clips_by_label.keys()):
            label_name = "target" if label_idx == 1 else "non_target"
            audio = clips_by_label[label_idx][i]
            rel_path = f"dataset/testing/{label_name}/sample_{i+1}.wav"
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
        rs.append(f"        expected_label: \"{label}\",\n")
        rs.append(f"        source_file: \"{rel_path}\",\n")
        rs.append(f"        audio: CLIP_{i},\n")
        rs.append("    },\n")
    rs.append("];\n")

    out_path.write_text("".join(rs), encoding="utf-8")

