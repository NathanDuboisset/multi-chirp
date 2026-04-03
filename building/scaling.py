from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras


DatasetTriplet = Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]


@dataclass
class ScalingConfig:
    dataset_root: Path
    build_model_fn: Callable[[Tuple[int, ...], int], "keras.Model"]
    make_datasets_fn: Callable[[Path, List[str], int], DatasetTriplet]
    max_n: int
    n_samples: int
    epochs: int = 10
    batch_size: int = 32
    seed: int = 42
    verbose: int = 0
    threshold: float = 0.5
    baseline_targets: int | None = None
    neg_pos_ratio: float = 1.0


def _get_all_classes(dataset_root: Path) -> List[str]:
    training = dataset_root / "training"
    if not training.exists():
        raise ValueError(f"training directory not found: {training}")
    names = sorted([p.name for p in training.iterdir() if p.is_dir()])
    if not names:
        raise ValueError(f"no class folders under {training}")
    return names


def _sample_binary_indices(
    y_bin: np.ndarray,
    neg_pos_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y_bin == 1.0)[0]
    neg_idx = np.where(y_bin == 0.0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return pos_idx, np.array([], dtype=int)
    n_neg = int(len(pos_idx) * neg_pos_ratio)
    n_neg = min(n_neg, len(neg_idx))
    if n_neg == 0:
        return pos_idx, np.array([], dtype=int)
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    return pos_idx, sampled_neg


def _dataset_to_arrays(ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for x_batch, y_batch in ds:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    if not xs:
        return np.empty((0,)), np.empty((0,))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


def _build_binary_dataset(
    ds: tf.data.Dataset,
    target_index: int,
    neg_pos_ratio: float,
    batch_size: int,
    rng: np.random.Generator,
) -> tf.data.Dataset:
    x, y_int = _dataset_to_arrays(ds)
    y_int = np.asarray(y_int).reshape(-1)
    y_bin = (y_int == target_index).astype("float32")
    pos_idx, neg_idx = _sample_binary_indices(y_bin, neg_pos_ratio, rng)
    keep = np.concatenate([pos_idx, neg_idx])
    if keep.size == 0:
        raise ValueError("no samples left after positive/negative sampling")
    rng.shuffle(keep)
    x_sel = x[keep]
    y_sel = y_bin[keep]
    ds_bin = tf.data.Dataset.from_tensor_slices((x_sel, y_sel))
    return ds_bin.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_multiclass_datasets(
    make_datasets_fn: Callable[[Path, List[str], int], DatasetTriplet],
    dataset_root: Path,
    class_names: List[str],
    batch_size: int,
) -> Tuple[DatasetTriplet, np.ndarray]:
    train_ds, val_ds, test_ds = make_datasets_fn(dataset_root, class_names, batch_size)
    # audio_dataset_from_directory with labels="inferred" yields integer labels.
    return (train_ds, val_ds, test_ds), np.array(class_names)


def _infer_input_shape(train_ds: tf.data.Dataset) -> Tuple[int, ...]:
    for x_batch, _ in train_ds.take(1):
        return tuple(x_batch.shape[1:])
    raise ValueError("empty training dataset")


def _per_class_positive_accuracy(
    model: "keras.Model",
    ds: tf.data.Dataset,
    threshold: float,
    n_classes: int,
) -> float:
    correct_counts = np.zeros(n_classes, dtype=np.int64)
    total_counts = np.zeros(n_classes, dtype=np.int64)
    for x_batch, y_batch in ds:
        y = y_batch.numpy().reshape(-1)
        preds = model(x_batch, training=False).numpy()
        if preds.shape[-1] != n_classes:
            raise ValueError(f"expected {n_classes} outputs, got {preds.shape[-1]}")
        for c in range(n_classes):
            mask = (y == c)
            if not np.any(mask):
                continue
            probs_c = preds[mask, c]
            correct = np.sum(probs_c >= threshold)
            correct_counts[c] += int(correct)
            total_counts[c] += int(mask.sum())
    per_class = []
    for c in range(n_classes):
        if total_counts[c] == 0:
            continue
        per_class.append(correct_counts[c] / total_counts[c])
    if not per_class:
        return 0.0
    return float(np.mean(per_class))


def train_and_evaluate_scaling(
    dataset_root: Path,
    build_model_fn: Callable[[Tuple[int, ...], int], "keras.Model"],
    make_datasets_fn: Callable[[Path, List[str], int], DatasetTriplet],
    max_n: int,
    n_samples: int,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    verbose: int = 0,
    threshold: float = 0.5,
    baseline_targets: int | None = None,
    neg_pos_ratio: float = 1.0,
) -> Tuple["DataFrame", float]:

    rng = np.random.default_rng(seed)
    all_classes = _get_all_classes(dataset_root)
    if max_n < 2:
        raise ValueError("max_n must be at least 2")
    if max_n > len(all_classes):
        raise ValueError(f"max_n={max_n} but only {len(all_classes)} classes are available")

    # Binary baseline: target vs all others, positives-only accuracy, computed once.
    if baseline_targets is None or baseline_targets >= len(all_classes):
        baseline_targets_list = all_classes
    else:
        baseline_targets_list = list(rng.choice(all_classes, size=baseline_targets, replace=False))

    _, _, test_full = make_datasets_fn(dataset_root, all_classes, batch_size)
    baseline_accs: List[float] = []

    for target_name in baseline_targets_list:
        target_index = all_classes.index(target_name)
        train_full, val_full, test_full_local = make_datasets_fn(dataset_root, all_classes, batch_size)

        train_bin = _build_binary_dataset(train_full, target_index, neg_pos_ratio, batch_size, rng)
        val_bin = _build_binary_dataset(val_full, target_index, neg_pos_ratio, batch_size, rng)
        test_bin = _build_binary_dataset(test_full_local, target_index, neg_pos_ratio, batch_size, rng)

        input_shape = _infer_input_shape(train_bin)
        model = build_model_fn(input_shape, 1)

        early = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
        model.fit(
            train_bin,
            validation_data=val_bin,
            epochs=epochs,
            verbose=verbose,
            callbacks=[early],
        )

        y_true = []
        y_score = []
        for x_batch, y_batch in test_bin:
            preds = model(x_batch, training=False).numpy().reshape(-1)
            labels = y_batch.numpy().reshape(-1)
            y_true.append(labels)
            y_score.append(preds)
        if not y_true:
            continue
        y_true_arr = np.concatenate(y_true, axis=0)
        y_score_arr = np.concatenate(y_score, axis=0)
        pos_mask = (y_true_arr == 1.0)
        if not np.any(pos_mask):
            continue
        pos_scores = y_score_arr[pos_mask]
        acc_pos = float(np.mean(pos_scores >= threshold))
        baseline_accs.append(acc_pos)

    baseline_mean = float(np.mean(baseline_accs)) if baseline_accs else 0.0

    # Multi-class scaling.
    rows: List[dict] = []
    for k in range(2, max_n + 1):
        run_scores: List[float] = []
        for _ in range(n_samples):
            subset = list(rng.choice(all_classes, size=k, replace=False))
            (train_ds, val_ds, test_ds), label_names = _build_multiclass_datasets(
                make_datasets_fn=make_datasets_fn,
                dataset_root=dataset_root,
                class_names=subset,
                batch_size=batch_size,
            )

            input_shape = _infer_input_shape(train_ds)
            model = build_model_fn(input_shape, k)

            early = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=verbose,
                callbacks=[early],
            )

            score = _per_class_positive_accuracy(
                model=model,
                ds=test_ds,
                threshold=threshold,
                n_classes=k,
            )
            run_scores.append(score)

        if not run_scores:
            rows.append(
                {
                    "n_classes": k,
                    "per_class_acc_mean": 0.0,
                    "per_class_acc_std": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "n_classes": k,
                    "per_class_acc_mean": float(np.mean(run_scores)),
                    "per_class_acc_std": float(np.std(run_scores)),
                }
            )

    df = DataFrame(rows)
    return df, baseline_mean


def plot_scaling(df: "DataFrame", baseline: float | None = None) -> None:
    import matplotlib.pyplot as plt

    x = df["n_classes"].to_numpy()
    y = df["per_class_acc_mean"].to_numpy()
    yerr = df["per_class_acc_std"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label="Per-class accuracy")
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    if baseline is not None:
        plt.axhline(baseline, color="red", linestyle="--", label="Binary baseline")

    plt.xlabel("Number of classes")
    plt.ylabel("Per-class accuracy (positives only)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

