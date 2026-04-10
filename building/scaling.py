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

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

FIT_VERBOSE = 2
SHUFFLE_BUFFER_CAP = 1024


def configure_tf_for_long_runs() -> None:
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


configure_tf_for_long_runs()


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


class RunMetrics(BaseModel):
    recall_mean: float
    recall_std: float
    precision_mean: float
    precision_std: float
    f1_mean: float
    f1_std: float
    top1_accuracy: float
    loss: float


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
    metrics: RunMetrics


class ScalingResult(BaseRunResult):
    run_type: Literal["scaling"]
    k: int
    sample_idx: int
    chosen_class_indices: list[int]
    chosen_classes: list[str]
    n_labels: int
    metrics: RunMetrics


RunResult = Annotated[
    Union[BaselineRunResult, ScalingResult], Field(discriminator="run_type")
]
_run_result_adapter = TypeAdapter(RunResult)


class BaselineSummary(BaseModel):
    recall: float
    precision: float
    f1: float
    top1_accuracy: float
    loss: float


@dataclass
class DatasetMeta:
    n_each: int
    n_non_target: int
    n_classes: int


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
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).reshape(-1).astype(np.int32),
    )


def load_full_arrays(
    collection: str,
    batch_size: int = 32,
    seed: int = 42,
    input_repr: Literal["time", "mel"] = "time",
) -> DatasetArrays:
    from building.mel_models import MEL_INPUT_SHAPE, build_cnn2d  # noqa: F401
    from building.time_models import TARGET_AUDIO_LEN
    from building.utils import make_mel_datasets, make_time_datasets

    dataset_root = ROOT / "datasets" / collection
    if input_repr == "mel":
        from building.mel_models import MEL_INPUT_SHAPE

        train_ds, val_ds, test_ds, label_names = make_mel_datasets(
            root=dataset_root, batch_size=batch_size, seed=seed
        )
        input_shape = MEL_INPUT_SHAPE
    else:
        train_ds, val_ds, test_ds, label_names = make_time_datasets(
            root=dataset_root, batch_size=batch_size, seed=seed, class_names=None
        )
        input_shape = (TARGET_AUDIO_LEN, 1)

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


def _model_factory(
    name: str, input_repr: Literal["time", "mel"] = "time"
) -> Callable[[int], tf.keras.Model]:
    if name == "cnn1d":
        from building.time_models import TARGET_AUDIO_LEN, build_cnn1d

        return lambda n_classes: build_cnn1d(
            n_classes=n_classes, input_len=TARGET_AUDIO_LEN
        )
    if name == "cnn2d":
        from building.mel_models import build_cnn2d

        return lambda n_classes: build_cnn2d(n_classes=n_classes)
    raise ValueError(f"Unknown model: {name}")


def _build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    chosen_idxs: list[int],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
    non_target_idx: int | None = None,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    remaining = [i for i in np.unique(y).tolist() if i not in chosen_idxs]
    if not remaining:
        raise ValueError("No remaining classes for non-target sampling.")

    # Use the dedicated non_target class exclusively when available.
    if non_target_idx is not None and non_target_idx in remaining:
        non_target_pool = np.where(y == non_target_idx)[0]
    else:
        non_target_pool = np.where(np.isin(y, remaining))[0]

    counts = [len(np.where(y == c)[0]) for c in chosen_idxs]
    if any(n == 0 for n in counts):
        raise ValueError("A chosen class has no samples.")

    n_each = min(counts)
    if shuffle:
        if len(non_target_pool) >= len(chosen_idxs):
            n_each = min(n_each, len(non_target_pool) // len(chosen_idxs))
        if n_each < 1:
            raise ValueError("Cannot build balanced training set.")
        n_non_target = min(n_each, len(non_target_pool))
    else:
        n_non_target = min(max(counts), len(non_target_pool))

    non_target_sample = rng.choice(non_target_pool, size=n_non_target, replace=False)
    non_target_label = len(chosen_idxs)

    parts_x, parts_y = [], []
    for local_label, c in enumerate(chosen_idxs):
        idx = np.where(y == c)[0]
        if shuffle and len(idx) > n_each:
            idx = rng.choice(idx, size=n_each, replace=False)
        parts_x.append(x[idx])
        parts_y.append(np.full(len(idx), local_label, dtype=np.int32))
    parts_x.append(x[non_target_sample])
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
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, loss: float
) -> RunMetrics:
    recalls, precisions, f1s = [], [], []
    for c in range(y_true.shape[1]):
        true_c = y_true[:, c] >= 0.5
        pred_c = y_pred[:, c] >= threshold
        rec = float(np.mean(pred_c[true_c])) if true_c.any() else None
        prec = float(np.mean(true_c[pred_c])) if pred_c.any() else None
        if rec is not None:
            recalls.append(rec)
        if prec is not None:
            precisions.append(prec)
        if rec is not None and prec is not None and (rec + prec) > 0:
            f1s.append(2 * rec * prec / (rec + prec))
    return RunMetrics(
        recall_mean=float(np.mean(recalls)) if recalls else 0.0,
        recall_std=float(np.std(recalls)) if recalls else 0.0,
        precision_mean=float(np.mean(precisions)) if precisions else 0.0,
        precision_std=float(np.std(precisions)) if precisions else 0.0,
        f1_mean=float(np.mean(f1s)) if f1s else 0.0,
        f1_std=float(np.std(f1s)) if f1s else 0.0,
        top1_accuracy=float(np.mean(np.argmax(y_true, 1) == np.argmax(y_pred, 1))),
        loss=loss,
    )


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
    model_builder = _model_factory(config.build_model, config.input_repr)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_file.parent.mkdir(parents=True, exist_ok=True)

    non_target_idx = (
        arrays.class_names.index("non_target")
        if "non_target" in arrays.class_names
        else None
    )
    existing = load_results(config.results_file)
    produced: list[RunResult] = []

    def is_done(run_type: Literal["baseline", "scaling"], **keys: object) -> bool:
        return any(
            r.run_type == run_type
            and all(getattr(r, k, None) == v for k, v in keys.items())
            for r in existing
        )

    def fit_eval(
        chosen_idxs: list[int],
        model_name: str,
        run_type: Literal["baseline", "scaling"],
        extra: dict[str, Any],
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
                non_target_idx=non_target_idx,
            )
            val_ds, _ = _build_dataset(
                arrays.x_val,
                arrays.y_val,
                chosen_idxs,
                config.batch_size,
                rng,
                shuffle=False,
                non_target_idx=non_target_idx,
            )
            test_ds, _ = _build_dataset(
                arrays.x_test,
                arrays.y_test,
                chosen_idxs,
                config.batch_size,
                rng,
                shuffle=False,
                non_target_idx=non_target_idx,
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
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=config.patience,
                        restore_best_weights=True,
                    )
                ],
            )
            test_loss = float(model.evaluate(test_ds, verbose=0)[0])  # ty:ignore[not-subscriptable]
            y_true, y_pred = _collect_predictions(model, test_ds)
            m = _compute_metrics(y_true, y_pred, config.threshold, test_loss)

            model_path = (
                config.models_dir
                / f"scaling_{len(chosen_idxs)}"
                / f"{model_name}.keras"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)

            common: dict[str, Any] = {
                "collection": config.collection,
                "build_model": config.build_model,
                "epochs_trained": len(history.history.get("loss", [])),
                "model_path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config.model_dump(mode="json"),
                **extra,
            }
            res_cls = BaselineRunResult if run_type == "baseline" else ScalingResult
            res = res_cls(run_type=run_type, metrics=m, **common)  # ty:ignore[invalid-argument-type]
            with config.results_file.open("a", encoding="utf-8") as f:
                f.write(res.model_dump_json() + "\n")
            existing.append(res)
            produced.append(res)
        finally:
            del model, history
            tf.keras.backend.clear_session()
            gc.collect()

    if run_baseline:
        species_classes = [
            (i, n) for i, n in enumerate(arrays.class_names) if n != "non_target"
        ]
        for target_idx, target_name in species_classes:
            if is_done("baseline", target_class=target_name):
                continue
            print(f"[baseline] target={target_name}")
            fit_eval(
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
            for sample_idx in range(n_samples):
                if is_done("scaling", k=k, sample_idx=sample_idx):
                    continue
                chosen_idxs = rng.choice(
                    n_classes_total, size=k, replace=False
                ).tolist()
                chosen_names = [arrays.class_names[i] for i in chosen_idxs]
                print(f"[scaling] k={k} sample={sample_idx + 1}/{n_samples}")
                fit_eval(
                    chosen_idxs,
                    f"sample_{'_'.join(str(i) for i in chosen_idxs)}",
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
    empty_df = pd.DataFrame(
        columns=[
            "k",
            "recall_mean",
            "recall_std",
            "precision_mean",
            "precision_std",
            "f1_mean",
            "f1_std",
            "top1_acc_mean",
            "top1_acc_std",
            "loss_mean",
            "loss_std",
        ]
    )
    if not rows:
        return BaselineSummary(
            recall=0.0, precision=0.0, f1=0.0, top1_accuracy=0.0, loss=0.0
        ), empty_df

    bl = [r for r in rows if isinstance(r, BaselineRunResult)]
    baseline = BaselineSummary(
        recall=float(np.mean([r.metrics.recall_mean for r in bl])) if bl else 0.0,
        precision=float(np.mean([r.metrics.precision_mean for r in bl])) if bl else 0.0,
        f1=float(np.mean([r.metrics.f1_mean for r in bl])) if bl else 0.0,
        top1_accuracy=float(np.mean([r.metrics.top1_accuracy for r in bl]))
        if bl
        else 0.0,
        loss=float(np.mean([r.metrics.loss for r in bl])) if bl else 0.0,
    )

    sc = [r for r in rows if isinstance(r, ScalingResult)]
    if not sc:
        return baseline, empty_df

    df = pd.DataFrame(
        [
            {
                "k": r.k,
                "recall_mean": r.metrics.recall_mean,
                "precision_mean": r.metrics.precision_mean,
                "f1_mean": r.metrics.f1_mean,
                "top1_accuracy": r.metrics.top1_accuracy,
                "loss": r.metrics.loss,
            }
            for r in sc
        ]
    )
    summary = (
        df.groupby("k", as_index=False)
        .agg(
            recall_mean=("recall_mean", "mean"),
            recall_std=("recall_mean", "std"),
            precision_mean=("precision_mean", "mean"),
            precision_std=("precision_mean", "std"),
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            top1_acc_mean=("top1_accuracy", "mean"),
            top1_acc_std=("top1_accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
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
        print("No scaling runs to plot.")
        return

    x = summary_df["k"].to_numpy()
    x_rand = np.linspace(x.min() if baseline is None else 1, x.max(), 300)
    rand_curve = 1.0 / (x_rand + 1)

    # (title, col_mean, col_std, baseline_val, color, marker, show_rand_curve)
    metrics = [
        (
            "Macro Recall",
            "recall_mean",
            "recall_std",
            baseline.recall if baseline else None,
            "steelblue",
            "o",
            True,
        ),
        (
            "Macro Precision",
            "precision_mean",
            "precision_std",
            baseline.precision if baseline else None,
            "darkorange",
            "s",
            True,
        ),
        (
            "Macro F1",
            "f1_mean",
            "f1_std",
            baseline.f1 if baseline else None,
            "mediumpurple",
            "D",
            True,
        ),
        (
            "Top-1 Accuracy",
            "top1_acc_mean",
            "top1_acc_std",
            baseline.top1_accuracy if baseline else None,
            "seagreen",
            "^",
            True,
        ),
        (
            "Test Loss",
            "loss_mean",
            "loss_std",
            baseline.loss if baseline else None,
            "crimson",
            "v",
            False,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scaling results by k", fontsize=13, fontweight="bold")
    flat_axes = axes.flatten()
    flat_axes[-1].set_visible(False)

    for ax, (title, col_mean, col_std, bl_val, color, marker, show_rand) in zip(
        flat_axes, metrics
    ):
        mean = summary_df[col_mean].to_numpy()
        std = summary_df[col_std].to_numpy()

        if show_rand:
            ax.plot(
                x_rand,
                rand_curve,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label="Random (1/(k+1))",
            )
        if bl_val is not None:
            ax.scatter(
                [1],
                [bl_val],
                color=color,
                marker="*",
                s=120,
                zorder=5,
                label=f"Baseline ({bl_val:.3f})",
            )
        ax.plot(x, mean, marker=marker, color=color, label=title)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k  (target classes + 1 non-target)")
        ax.set_ylabel("Loss" if not show_rand else "Score")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
