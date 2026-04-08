from __future__ import annotations

from datetime import datetime, timezone
import gc
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Union, Annotated
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
import pyrootutils
import tensorflow as tf

from building.mel_models import MEL_INPUT_SHAPE, build_cnn2d
from building.time_models import TARGET_AUDIO_LEN, build_cnn1d
from building.utils import make_mel_datasets, make_time_datasets

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
    input_repr: Literal["time", "mel"] = "time"


class DatasetArrays(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


class BaseRunResult(BaseModel):
    collection: str
    build_model: str
    epochs_trained: int
    model_path: str
    timestamp: str
    config: dict[str, Any]


class BaselineRunResult(BaseRunResult):
    run_type: Literal["baseline"]
    target_class: str
    target_idx: int
    recall: float
    precision: float


class ScalingResult(BaseRunResult):
    run_type: Literal["scaling"]
    k: int
    sample_idx: int
    chosen_class_indices: list[int]
    chosen_classes: list[str]
    n_labels: int
    per_class_recall_mean: float
    per_class_recall_std: float
    per_class_precision_mean: float
    per_class_precision_std: float
    top1_accuracy: float


RunResult = Annotated[
    Union[BaselineRunResult, ScalingResult], Field(discriminator="run_type")
]
_run_result_adapter = TypeAdapter(RunResult)


class BaselineSummary(BaseModel):
    recall: float
    precision: float


FIT_VERBOSE = 2
SHUFFLE_BUFFER_CAP = 1024


def _configure_tf_for_long_runs() -> None:
    # Disable XLA JIT to reduce long-run memory growth in notebook loops.
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    # Avoid pre-allocating all GPU VRAM, which can make desktop sessions unstable.
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


_configure_tf_for_long_runs()


def _empty_input_shape(input_repr: Literal["time", "mel"]) -> tuple[int, ...]:
    if input_repr == "time":
        return (TARGET_AUDIO_LEN, 1)
    return MEL_INPUT_SHAPE


def _dataset_to_arrays(
    ds: tf.data.Dataset, input_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_batch, y_batch in ds:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    if not xs:
        return np.empty((0, *input_shape), dtype=np.float32), np.empty(
            (0,), dtype=np.int32
        )
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int32)
    return x, y


def load_full_arrays(
    collection: str,
    batch_size: int = 32,
    seed: int = 42,
    input_repr: Literal["time", "mel"] = "time",
) -> DatasetArrays:
    dataset_root = ROOT / "datasets" / collection
    if input_repr == "mel":
        train_ds, val_ds, test_ds, label_names = make_mel_datasets(
            root=dataset_root,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        train_ds, val_ds, test_ds, label_names = make_time_datasets(
            root=dataset_root,
            batch_size=batch_size,
            seed=seed,
            class_names=None,
        )
    input_shape = _empty_input_shape(input_repr)
    x_train, y_train = _dataset_to_arrays(train_ds, input_shape)
    x_val, y_val = _dataset_to_arrays(val_ds, input_shape)
    x_test, y_test = _dataset_to_arrays(test_ds, input_shape)
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
        return lambda n_classes: build_cnn1d(
            n_classes=n_classes, input_len=TARGET_AUDIO_LEN
        )
    if name == "cnn2d":
        return lambda n_classes: build_cnn2d(n_classes=n_classes)
    raise ValueError(f"Unknown model: {name}")


def _make_early_stopping(patience: int) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


@dataclass
class DatasetMeta:
    n_each: int
    n_non_target: int
    n_classes: int  # chosen + 1 non-target


def _build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    chosen_idxs: list[int],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    remaining = [i for i in np.unique(y).tolist() if i not in chosen_idxs]
    if not remaining:
        raise ValueError("No remaining classes for non-target sampling.")

    counts = [len(np.where(y == c)[0]) for c in chosen_idxs]
    if any(n == 0 for n in counts):
        raise ValueError("A chosen class has no samples.")
    non_target_pool = np.where(np.isin(y, remaining))[0]

    n_each = min(counts)
    if shuffle:
        if len(non_target_pool) >= len(chosen_idxs):
            n_each = min(n_each, len(non_target_pool) // len(chosen_idxs))
        if n_each < 1:
            raise ValueError("Cannot build balanced training set.")
        n_non_target = min(n_each, len(non_target_pool))
    else:
        n_non_target = min(max(counts), len(non_target_pool))

    non_target_idx = rng.choice(non_target_pool, size=n_non_target, replace=False)
    non_target_label = len(chosen_idxs)

    parts_x, parts_y = [], []
    for local_label, c in enumerate(chosen_idxs):
        idx = np.where(y == c)[0]
        if shuffle and len(idx) > n_each:
            idx = rng.choice(idx, size=n_each, replace=False)
        parts_x.append(x[idx])
        parts_y.append(np.full(len(idx), local_label, dtype=np.int32))

    parts_x.append(x[non_target_idx])
    parts_y.append(np.full(n_non_target, non_target_label, dtype=np.int32))

    x_all = np.concatenate(parts_x)
    y_all = np.concatenate(parts_y)
    order = rng.permutation(len(x_all))

    ds = tf.data.Dataset.from_tensor_slices((x_all[order], y_all[order]))
    if shuffle:
        ds = ds.shuffle(
            min(len(x_all), SHUFFLE_BUFFER_CAP), seed=int(rng.integers(0, 2**31 - 1))
        )
    ds = (
        ds.batch(batch_size)
        .map(
            lambda xb, yb: (xb, tf.one_hot(yb, non_target_label + 1, dtype=tf.float32)),
            num_parallel_calls=1,
        )
        .prefetch(1)
    )
    return ds, DatasetMeta(
        n_each=int(n_each),
        n_non_target=int(n_non_target),
        n_classes=non_target_label + 1,
    )


def _collect_predictions(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for xb, yb in ds:
        y_true.append(yb.numpy())
        y_pred.append(model(xb, training=False).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float]:
    recalls, precisions = [], []
    for c in range(y_true.shape[1]):
        true_c = y_true[:, c] >= 0.5
        pred_c = y_pred[:, c] >= threshold
        if true_c.any():
            recalls.append(float(np.mean(pred_c[true_c])))
        if pred_c.any():
            precisions.append(float(np.mean(true_c[pred_c])))
    return {
        "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recall_std": float(np.std(recalls)) if recalls else 0.0,
        "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
        "precision_std": float(np.std(precisions)) if precisions else 0.0,
        "top1_accuracy": float(np.mean(np.argmax(y_true, 1) == np.argmax(y_pred, 1))),
    }


def _cleanup_after_fit() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def load_results(results_file: Path) -> list[RunResult]:
    if not results_file.exists():
        return []
    with results_file.open("r", encoding="utf-8") as f:
        return [_run_result_adapter.validate_json(line) for line in f if line.strip()]


def run_experiments(
    arrays: DatasetArrays,
    config: ScalingRunConfig,
    n_samples: int = 3,
    k_values: Iterable[int] = range(2, 10),
    *,
    run_baseline: bool = True,
    run_scaling: bool = True,
) -> list[RunResult]:
    rng = np.random.default_rng(config.seed)
    model_builder = _model_factory(config.build_model)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_results(config.results_file)
    produced: list[RunResult] = []

    def _is_done(run_type: Literal["baseline", "scaling"], **keys: object) -> bool:
        return any(
            r.run_type == run_type
            and all(getattr(r, k, None) == v for k, v in keys.items())
            for r in existing
        )

    def _save(row: RunResult) -> None:
        with config.results_file.open("a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")
        existing.append(row)
        produced.append(row)

    def _fit_eval_cleanup(
        chosen_idxs: list[int],
        model_name: str,
        run_type: Literal["baseline", "scaling"],
        extra_info: dict[str, Any],
    ) -> None:
        model = history = None
        try:
            train_ds, meta = _build_dataset(
                arrays.x_train,
                arrays.y_train,
                chosen_idxs,
                config.batch_size,
                rng,
                shuffle=True,
            )
            val_ds, _ = _build_dataset(
                arrays.x_val,
                arrays.y_val,
                chosen_idxs,
                config.batch_size,
                rng,
                shuffle=False,
            )
            test_ds, _ = _build_dataset(
                arrays.x_test,
                arrays.y_test,
                chosen_idxs,
                config.batch_size,
                rng,
                shuffle=False,
            )

            print(
                f"[{run_type}] n_each={meta.n_each} n_non_target={meta.n_non_target} n_classes={meta.n_classes}"
            )

            model = model_builder(meta.n_classes)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.epochs,
                verbose=FIT_VERBOSE,
                callbacks=[_make_early_stopping(config.patience)],
            )

            y_true, y_pred = _collect_predictions(model, test_ds)
            m = _compute_metrics(y_true, y_pred, config.threshold)

            model_path = (
                config.models_dir
                / ("scaling_" + str(len(chosen_idxs)))
                / f"{model_name}.keras"
            )
            model.save(model_path)

            common: dict[str, Any] = {
                "collection": config.collection,
                "build_model": config.build_model,
                "epochs_trained": len(history.history.get("loss", [])),
                "model_path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config.model_dump(mode="json"),
                **extra_info,
            }
            if run_type == "baseline":
                res = BaselineRunResult(
                    run_type="baseline",
                    recall=m["recall_mean"],
                    precision=m["precision_mean"],
                    **common,
                )  # type: ignore
            else:
                res = ScalingResult(
                    run_type="scaling",
                    per_class_recall_mean=m["recall_mean"],
                    per_class_recall_std=m["recall_std"],
                    per_class_precision_mean=m["precision_mean"],
                    per_class_precision_std=m["precision_std"],
                    top1_accuracy=m["top1_accuracy"],
                    **common,  # type: ignore
                )
            _save(res)
        finally:
            del model, history
            _cleanup_after_fit()

    if run_baseline:
        baseline_dir = config.models_dir / "scaling_1"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for target_idx, target_name in enumerate(arrays.class_names):
            if _is_done("baseline", target_class=target_name):
                continue
            print(f"[baseline] target={target_name}")
            _fit_eval_cleanup(
                [target_idx],
                f"sample_{target_idx}",
                "baseline",
                {"target_class": target_name, "target_idx": target_idx},
            )

    if run_scaling:
        n_classes_total = len(arrays.class_names)
        for k in k_values:
            if k < 2 or k >= n_classes_total:
                continue
            scaling_dir = config.models_dir / f"scaling_{k}"
            scaling_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(n_samples):
                if _is_done("scaling", k=k, sample_idx=sample_idx):
                    continue
                chosen_idxs = rng.choice(
                    n_classes_total, size=k, replace=False
                ).tolist()
                chosen_names = [arrays.class_names[i] for i in chosen_idxs]
                print(f"[scaling] k={k} sample={sample_idx + 1}/{n_samples}")
                sample_name = chosen_idxs.join("_")
                _fit_eval_cleanup(
                    chosen_idxs,
                    f"sample_{sample_name}",
                    "scaling",
                    {
                        "k": k,
                        "sample_idx": sample_idx,
                        "chosen_class_indices": chosen_idxs,
                        "chosen_classes": chosen_names,
                        "n_labels": k + 1,
                    },
                )

    return produced


def summarize_results(results_file: Path) -> tuple[BaselineSummary, pd.DataFrame]:
    rows = load_results(results_file)
    if not rows:
        return BaselineSummary(recall=0.0, precision=0.0), pd.DataFrame(
            columns=[
                "k",
                "recall_mean",
                "recall_std",
                "precision_mean",
                "precision_std",
                "top1_acc_mean",
                "top1_acc_std",
            ]
        )

    baseline_vals = [r.recall for r in rows if isinstance(r, BaselineRunResult)]
    baseline_precision_vals = [
        r.precision for r in rows if isinstance(r, BaselineRunResult)
    ]
    baseline = BaselineSummary(
        recall=float(np.mean(baseline_vals)) if baseline_vals else 0.0,
        precision=float(np.mean(baseline_precision_vals))
        if baseline_precision_vals
        else 0.0,
    )

    scaling_rows = [r.model_dump() for r in rows if isinstance(r, ScalingResult)]
    if not scaling_rows:
        return baseline, pd.DataFrame(
            columns=[
                "k",
                "recall_mean",
                "recall_std",
                "precision_mean",
                "precision_std",
                "top1_acc_mean",
                "top1_acc_std",
            ]
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


def plot_summary(
    summary_df: pd.DataFrame, baseline: BaselineSummary | None = None
) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("No scaling runs found in results file.")
        return

    x = summary_df["k"].to_numpy()
    x_full = (
        np.array([1, *x]) if baseline is not None else x
    )  # prepend k=1 for baseline point

    # Random-chance baselines as a function of k: 1/(k+1) classes total
    x_rand = np.linspace(x_full.min(), x_full.max(), 300)
    rand_curve = 1.0 / (x_rand + 1)

    metrics = [
        {
            "title": "Macro Recall",
            "mean": summary_df["recall_mean"].to_numpy(),
            "std": summary_df["recall_std"].to_numpy(),
            "baseline_val": baseline.recall if baseline else None,
            "color": "steelblue",
            "marker": "o",
        },
        {
            "title": "Macro Precision",
            "mean": summary_df["precision_mean"].to_numpy(),
            "std": summary_df["precision_std"].to_numpy(),
            "baseline_val": baseline.precision if baseline else None,
            "color": "darkorange",
            "marker": "s",
        },
        {
            "title": "Top-1 Accuracy",
            "mean": summary_df["top1_acc_mean"].to_numpy(),
            "std": summary_df["top1_acc_std"].to_numpy(),
            "baseline_val": None,  # top-1 not tracked in BaselineSummary; extend if needed
            "color": "seagreen",
            "marker": "^",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Scaling results by k", fontsize=13, fontweight="bold")

    for ax, m in zip(axes, metrics):
        mean, std = m["mean"], m["std"]

        # Random-chance reference curve
        ax.plot(
            x_rand,
            rand_curve,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Random (1 / (k+1))",
        )

        # Baseline as explicit k=1 point
        if m["baseline_val"] is not None:
            ax.scatter(
                [1],
                [m["baseline_val"]],
                color=m["color"],
                marker="*",
                s=120,
                zorder=5,
                label="Baseline (k=1)",
            )

        # Scaling curve
        ax.plot(x, mean, marker=m["marker"], color=m["color"], label=m["title"])
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=m["color"])

        ax.set_title(m["title"], fontsize=11)
        ax.set_xlabel("k  (target classes + 1 non-target)")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
