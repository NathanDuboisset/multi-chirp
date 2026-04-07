from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
import pyrootutils
import tensorflow as tf

from building.time_models import TARGET_AUDIO_LEN, build_cnn1d
from building.utils import make_time_datasets

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)


class ScalingRunConfig(BaseModel):
    collection: str
    build_model: str
    epochs: int
    patience: int
    batch_size: int
    seed: int
    threshold: float
    models_dir: Path
    results_file: Path


class DatasetArrays(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


FIT_VERBOSE = 2
SHUFFLE_BUFFER_CAP = 1024


def _configure_tf_for_long_runs() -> None:
    # Disable XLA JIT to reduce long-run memory growth in notebook loops.
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass


_configure_tf_for_long_runs()


def _dataset_to_arrays(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_batch, y_batch in ds:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    if not xs:
        return np.empty((0, TARGET_AUDIO_LEN, 1), dtype=np.float32), np.empty((0,), dtype=np.int32)
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int32)
    return x, y


def load_full_arrays(collection: str, batch_size: int = 32, seed: int = 42) -> DatasetArrays:
    dataset_root = ROOT / "datasets" / collection
    train_ds, val_ds, test_ds, label_names = make_time_datasets(
        root=dataset_root,
        batch_size=batch_size,
        seed=seed,
        class_names=None,
    )
    x_train, y_train = _dataset_to_arrays(train_ds)
    x_val, y_val = _dataset_to_arrays(val_ds)
    x_test, y_test = _dataset_to_arrays(test_ds)
    return DatasetArrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=label_names.tolist(),
    )


def _model_factory(name: str) -> Callable[[int], tf.keras.Model]:
    if name == "cnn1d":
        return lambda n_classes: build_cnn1d(n_classes=n_classes, input_len=TARGET_AUDIO_LEN)
    raise ValueError(f"Unknown model: {name}")


def _make_early_stopping(patience: int) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def _build_binary_dataset(
    x: np.ndarray,
    y: np.ndarray,
    target_idx: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[tf.data.Dataset, dict[str, int]]:
    y_bin = (y == target_idx).astype(np.float32)
    pos_idx = np.where(y_bin == 1.0)[0]
    neg_idx = np.where(y_bin == 0.0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Binary dataset requires both positive and negative samples.")

    n_neg = min(len(neg_idx), len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(keep)

    x_sel = x[keep]
    y_sel = y_bin[keep]
    ds = tf.data.Dataset.from_tensor_slices((x_sel, y_sel))
    if shuffle:
        buffer_size = min(len(x_sel), SHUFFLE_BUFFER_CAP)
        ds = ds.shuffle(buffer_size=buffer_size, seed=int(rng.integers(0, 2**31 - 1)))
    meta = {
        "target_count": int(len(pos_idx)),
        "non_target_count": int(n_neg),
    }
    return ds.batch(batch_size).prefetch(1), meta


def _build_k_plus_non_target_dataset(
    x: np.ndarray,
    y: np.ndarray,
    chosen_idxs: Sequence[int],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[tf.data.Dataset, dict[str, int | float | bool]]:
    chosen_idxs = list(chosen_idxs)
    remaining_idxs = [i for i in np.unique(y).tolist() if i not in chosen_idxs]
    if not remaining_idxs:
        raise ValueError("No remaining classes available for non-target sampling.")

    chosen_masks = [np.where(y == c)[0] for c in chosen_idxs]
    if any(len(idx) == 0 for idx in chosen_masks):
        raise ValueError("A selected class has no samples.")

    non_target_pool = np.where(np.isin(y, remaining_idxs))[0]
    if len(non_target_pool) == 0:
        raise ValueError("Non-target pool is empty.")

    is_training = bool(shuffle)
    min_count = min(len(idx) for idx in chosen_masks)
    max_count = max(len(idx) for idx in chosen_masks)
    n_target_each = min_count
    if is_training:
        # Keep chosen classes balanced while forcing non-target to be >= 50% when possible.
        if len(non_target_pool) >= len(chosen_idxs):
            n_target_each = min(min_count, len(non_target_pool) // len(chosen_idxs))
        if n_target_each < 1:
            raise ValueError("Cannot build balanced training set with current class counts.")

    target_total = n_target_each * len(chosen_idxs)
    if is_training:
        # User rule: non-target count must match per-class chosen count.
        non_target_n = min(n_target_each, len(non_target_pool))
    else:
        # Keep previous eval behavior: non-target capped by largest chosen class.
        non_target_n = min(max_count, len(non_target_pool))
    non_target_idx = rng.choice(non_target_pool, size=non_target_n, replace=False)

    label_map = {class_idx: local_idx for local_idx, class_idx in enumerate(chosen_idxs)}
    non_target_label = len(chosen_idxs)

    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for class_idx in chosen_idxs:
        idx = np.where(y == class_idx)[0]
        if is_training and len(idx) > n_target_each:
            idx = rng.choice(idx, size=n_target_each, replace=False)
        parts_x.append(x[idx])
        parts_y.append(np.full(shape=(len(idx),), fill_value=label_map[class_idx], dtype=np.int32))

    parts_x.append(x[non_target_idx])
    parts_y.append(np.full(shape=(len(non_target_idx),), fill_value=non_target_label, dtype=np.int32))

    x_all = np.concatenate(parts_x, axis=0)
    y_int = np.concatenate(parts_y, axis=0)
    order = np.arange(len(x_all))
    rng.shuffle(order)
    x_all = x_all[order]
    y_int = y_int[order]

    ds = tf.data.Dataset.from_tensor_slices((x_all, y_int))
    if shuffle:
        buffer_size = min(len(x_all), SHUFFLE_BUFFER_CAP)
        ds = ds.shuffle(buffer_size=buffer_size, seed=int(rng.integers(0, 2**31 - 1)))
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda xb, yb: (xb, tf.one_hot(yb, depth=non_target_label + 1, dtype=tf.float32)),
        num_parallel_calls=1,
    )
    meta = {
        "chosen_count_each": int(n_target_each),
        "n_chosen_classes": int(len(chosen_idxs)),
        "target_total": int(target_total),
        "non_target_count": int(non_target_n),
        "non_target_ratio": float(non_target_n / max(1, non_target_n + target_total)),
        "non_target_matches_chosen_each": bool(non_target_n == n_target_each),
    }
    return ds.prefetch(1), meta


def _cleanup_after_fit() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def _positive_recall(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float) -> float:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    for x_batch, y_batch in ds:
        preds = model(x_batch, training=False).numpy().reshape(-1)
        y_true.append(y_batch.numpy().reshape(-1))
        y_score.append(preds)
    y_true_arr = np.concatenate(y_true, axis=0)
    y_score_arr = np.concatenate(y_score, axis=0)
    pos_mask = y_true_arr == 1.0
    if not np.any(pos_mask):
        return 0.0
    return float(np.mean(y_score_arr[pos_mask] >= threshold))


def _positive_precision(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float) -> float:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    for x_batch, y_batch in ds:
        preds = model(x_batch, training=False).numpy().reshape(-1)
        y_true.append(y_batch.numpy().reshape(-1))
        y_score.append(preds)
    y_true_arr = np.concatenate(y_true, axis=0)
    y_score_arr = np.concatenate(y_score, axis=0)
    pred_pos = y_score_arr >= threshold
    if not np.any(pred_pos):
        return 0.0
    tp = np.sum((y_true_arr == 1.0) & pred_pos)
    return float(tp / np.sum(pred_pos))


def _macro_per_class_recall(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float) -> tuple[float, float]:
    recalls: list[float] = []
    for x_batch, y_batch in ds:
        preds = model(x_batch, training=False).numpy()
        y_true = y_batch.numpy()
        for c in range(y_true.shape[1]):
            true_c = y_true[:, c] >= 0.5
            if not np.any(true_c):
                continue
            pred_c = preds[:, c] >= threshold
            recalls.append(float(np.mean(pred_c[true_c])))
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.std(recalls))


def _macro_per_class_precision(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float) -> tuple[float, float]:
    precisions: list[float] = []
    for x_batch, y_batch in ds:
        preds = model(x_batch, training=False).numpy()
        y_true = y_batch.numpy()
        for c in range(y_true.shape[1]):
            pred_c = preds[:, c] >= threshold
            if not np.any(pred_c):
                continue
            true_c = y_true[:, c] >= 0.5
            precisions.append(float(np.mean(true_c[pred_c])))
    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.std(precisions))


def _top1_accuracy(model: tf.keras.Model, ds: tf.data.Dataset) -> float:
    correct = 0
    total = 0
    for x_batch, y_batch in ds:
        preds = model(x_batch, training=False).numpy()
        y_true = np.argmax(y_batch.numpy(), axis=1)
        y_pred = np.argmax(preds, axis=1)
        correct += int(np.sum(y_true == y_pred))
        total += int(len(y_true))
    if total == 0:
        return 0.0
    return float(correct / total)


def load_results(results_file: Path) -> list[dict]:
    if not results_file.exists():
        return []
    rows: list[dict] = []
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_experiments(
    arrays: DatasetArrays,
    config: ScalingRunConfig,
    n_samples: int = 3,
    k_values: Iterable[int] = range(2, 10),
    *,
    run_baseline: bool = True,
    run_scaling: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(config.seed)
    model_builder = _model_factory(config.build_model)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_results(config.results_file)
    produced: list[dict] = []
    n_classes_total = len(arrays.class_names)

    def _is_done(**keys: object) -> bool:
        return any(all(r.get(k) == v for k, v in keys.items()) for r in existing)

    def _save(row: dict) -> None:
        with config.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        existing.append(row)
        produced.append(row)

    if run_baseline:
        for target_idx, target_name in enumerate(arrays.class_names):
            if _is_done(run_type="baseline", target_class=target_name):
                continue
            model = history = train_ds = val_ds = test_ds = train_meta = None
            try:
                print(f"[baseline] target={target_name}")
                train_ds, train_meta = _build_binary_dataset(
                    arrays.x_train, arrays.y_train, target_idx, config.batch_size, rng, True
                )
                val_ds, _ = _build_binary_dataset(
                    arrays.x_val, arrays.y_val, target_idx, config.batch_size, rng, False
                )
                test_ds, _ = _build_binary_dataset(
                    arrays.x_test, arrays.y_test, target_idx, config.batch_size, rng, False
                )
                train_total = train_meta["target_count"] + train_meta["non_target_count"]
                print(
                    f"[baseline data] target={train_meta['target_count']} non_target={train_meta['non_target_count']} "
                    f"non_target_ratio={train_meta['non_target_count'] / max(1, train_total):.2f}"
                )
                model = model_builder(1)
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=config.epochs,
                    verbose=FIT_VERBOSE,
                    callbacks=[_make_early_stopping(config.patience)],
                )
                model_path = config.models_dir / f"baseline_{target_name}.keras"
                model.save(model_path)
                recall = _positive_recall(model, test_ds, config.threshold)
                precision = _positive_precision(model, test_ds, config.threshold)
                _save({
                    "run_type": "baseline",
                    "collection": config.collection,
                    "build_model": config.build_model,
                    "target_class": target_name,
                    "target_idx": target_idx,
                    "recall": recall,
                    "precision": precision,
                    "epochs_trained": int(len(history.history.get("loss", []))),
                    "model_path": str(model_path),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": config.model_dump(mode="json"),
                })
            finally:
                del model, history, train_ds, val_ds, test_ds, train_meta
                _cleanup_after_fit()

    if run_scaling:
        for k in k_values:
            if k < 2 or k >= n_classes_total:
                continue
            for sample_idx in range(n_samples):
                if _is_done(run_type="scaling", k=k, sample_idx=sample_idx):
                    continue
                model = history = train_ds = val_ds = test_ds = train_meta = None
                try:
                    print(f"[scaling] k={k} sample={sample_idx + 1}/{n_samples}")
                    chosen_idxs = rng.choice(np.arange(n_classes_total), size=k, replace=False).tolist()
                    chosen_names = [arrays.class_names[i] for i in chosen_idxs]
                    train_ds, train_meta = _build_k_plus_non_target_dataset(
                        arrays.x_train, arrays.y_train, chosen_idxs, config.batch_size, rng, True
                    )
                    val_ds, _ = _build_k_plus_non_target_dataset(
                        arrays.x_val, arrays.y_val, chosen_idxs, config.batch_size, rng, False
                    )
                    test_ds, _ = _build_k_plus_non_target_dataset(
                        arrays.x_test, arrays.y_test, chosen_idxs, config.batch_size, rng, False
                    )
                    print(
                        f"[scaling data] chosen_each={train_meta['chosen_count_each']} "
                        f"target_total={train_meta['target_total']} non_target={train_meta['non_target_count']} "
                        f"non_target_ratio={train_meta['non_target_ratio']:.2f} "
                        f"match_rule={train_meta['non_target_matches_chosen_each']}"
                    )
                    model = model_builder(k + 1)
                    history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=config.epochs,
                        verbose=FIT_VERBOSE,
                        callbacks=[_make_early_stopping(config.patience)],
                    )
                    model_path = config.models_dir / f"scaling_k{k}_s{sample_idx}.keras"
                    model.save(model_path)
                    recall_mean, recall_std = _macro_per_class_recall(model, test_ds, config.threshold)
                    precision_mean, precision_std = _macro_per_class_precision(model, test_ds, config.threshold)
                    top1_acc = _top1_accuracy(model, test_ds)
                    _save({
                        "run_type": "scaling",
                        "collection": config.collection,
                        "build_model": config.build_model,
                        "k": k,
                        "sample_idx": sample_idx,
                        "chosen_class_indices": chosen_idxs,
                        "chosen_classes": chosen_names,
                        "n_labels": k + 1,
                        "per_class_recall_mean": recall_mean,
                        "per_class_recall_std": recall_std,
                        "per_class_precision_mean": precision_mean,
                        "per_class_precision_std": precision_std,
                        "top1_accuracy": top1_acc,
                        "epochs_trained": int(len(history.history.get("loss", []))),
                        "model_path": str(model_path),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config": config.model_dump(mode="json"),
                    })
                finally:
                    del model, history, train_ds, val_ds, test_ds, train_meta
                    _cleanup_after_fit()

    return produced


def summarize_results(results_file: Path) -> tuple[dict[str, float], pd.DataFrame]:
    rows = load_results(results_file)
    if not rows:
        return {"recall": 0.0, "precision": 0.0}, pd.DataFrame(
            columns=["k", "recall_mean", "recall_std", "precision_mean", "precision_std", "top1_acc_mean", "top1_acc_std"]
        )

    baseline_vals = [r["recall"] for r in rows if r.get("run_type") == "baseline" and "recall" in r]
    baseline_precision_vals = [r["precision"] for r in rows if r.get("run_type") == "baseline" and "precision" in r]
    baseline = {
        "recall": float(np.mean(baseline_vals)) if baseline_vals else 0.0,
        "precision": float(np.mean(baseline_precision_vals)) if baseline_precision_vals else 0.0,
    }

    scaling_rows = [r for r in rows if r.get("run_type") == "scaling"]
    if not scaling_rows:
        return baseline, pd.DataFrame(
            columns=["k", "recall_mean", "recall_std", "precision_mean", "precision_std", "top1_acc_mean", "top1_acc_std"]
        )

    df = pd.DataFrame(scaling_rows)
    if "per_class_precision_mean" not in df.columns:
        df["per_class_precision_mean"] = np.nan
    if "top1_accuracy" not in df.columns:
        df["top1_accuracy"] = np.nan
    summary = (
        df.groupby("k", as_index=False)
        .agg(
            recall_mean=("per_class_recall_mean", "mean"),
            recall_std=("per_class_recall_mean", "std"),
            precision_mean=("per_class_precision_mean", "mean"),
            precision_std=("per_class_precision_mean", "std"),
            top1_acc_mean=("top1_accuracy", "mean"),
            top1_acc_std=("top1_accuracy", "std"),
        )
        .fillna(0.0)
        .sort_values("k")
    )
    return baseline, summary


def plot_summary(summary_df: pd.DataFrame, baseline: dict[str, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("No scaling runs found in results file.")
        return

    x = summary_df["k"].to_numpy()
    rec = summary_df["recall_mean"].to_numpy()
    rec_std = summary_df["recall_std"].to_numpy()
    prec = summary_df["precision_mean"].to_numpy()
    prec_std = summary_df["precision_std"].to_numpy()
    top1 = summary_df["top1_acc_mean"].to_numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(x, rec, marker="o", label="Macro recall")
    plt.fill_between(x, rec - rec_std, rec + rec_std, alpha=0.2)
    plt.plot(x, prec, marker="s", label="Macro precision")
    plt.fill_between(x, prec - prec_std, prec + prec_std, alpha=0.2)
    plt.plot(x, top1, marker="^", label="Top-1 accuracy")
    if baseline is not None:
        plt.axhline(y=baseline.get("recall", 0.0), linestyle="--", label="Baseline recall")
        plt.axhline(y=baseline.get("precision", 0.0), linestyle=":", label="Baseline precision")
    plt.xlabel("k chosen classes (+ non-target)")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
