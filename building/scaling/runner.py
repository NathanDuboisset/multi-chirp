from __future__ import annotations

from datetime import datetime, timezone
import gc
from pathlib import Path
from typing import Any, Iterable, Literal, Union, Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter
import tensorflow as tf

from building.training import (
    DatasetCatalog,
    RunMetrics,
    FIT_VERBOSE,
    build_dataset_from_catalog,
    collect_predictions,
    compute_metrics,
    model_factory,
)

NON_TARGET_NAME = "non_target"

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
    augment: bool = True
    # fold this many unchosen target species into the "other" class so its
    # diversity stays comparable across k values
    extras_into_other: int = 4
    # ReduceLROnPlateau sits inside EarlyStopping(patience)
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_min: float = 1e-5


class BaseRunResult(BaseModel):
    collection: str
    build_model: str
    epochs_trained: int
    model_path: str
    timestamp: str
    config: dict[str, Any]


class QuantStats(BaseModel):
    model_size_kb: float
    arena_size_kb: float | None = None
    flops_mflops: float
    input_dtype: str
    output_dtype: str
    input_shape: list[int]
    tflite_path: str


class BaselineRunResult(BaseRunResult):
    run_type: Literal["baseline"]
    target_class: str
    target_idx: int
    metrics: RunMetrics
    # optional so old rows keep parsing
    label_names: list[str] | None = None
    float_metrics: RunMetrics | None = None
    quant_stats: QuantStats | None = None
    tflite_path: str | None = None
    npz_path: str | None = None


class ScalingResult(BaseRunResult):
    run_type: Literal["scaling"]
    k: int
    sample_idx: int
    chosen_class_indices: list[int]
    chosen_classes: list[str]
    n_labels: int
    extra_other_indices: list[int] = []
    extra_other_classes: list[str] = []
    metrics: RunMetrics
    # optional so old rows keep parsing
    label_names: list[str] | None = None
    float_metrics: RunMetrics | None = None
    quant_stats: QuantStats | None = None
    tflite_path: str | None = None
    npz_path: str | None = None


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


def load_results(results_file: Path) -> list[RunResult]:
    if not results_file.exists():
        return []
    with results_file.open("r", encoding="utf-8") as f:
        return [_run_result_adapter.validate_json(line) for line in f if line.strip()]


def run_experiments(
    catalog: DatasetCatalog,
    config: ScalingRunConfig,
    n_samples: int = 3,
    k_values: Iterable[int] = range(2, 6),
    *,
    run_baseline: bool = True,
    run_scaling: bool = True,
) -> list[RunResult]:
    from building.models import input_repr_for

    rng = np.random.default_rng(config.seed)
    model_builder = model_factory(config.build_model)
    input_repr: Literal["time", "mel"] = input_repr_for(config.build_model)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_file.parent.mkdir(parents=True, exist_ok=True)

    if "non_target" not in catalog.class_names:
        raise ValueError("Dataset must contain a 'non_target' class.")
    non_target_idx = catalog.class_names.index("non_target")
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
        extra_other_idxs: list[int] | None = None,
    ) -> None:
        tf.keras.backend.clear_session()
        gc.collect()
        model = history = None
        train_ds = val_ds = test_ds = None
        try:
            train_ds, meta = build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="train",
                input_repr=input_repr,
                augment=config.augment,
                extra_other_idxs=extra_other_idxs,
            )
            val_ds, _ = build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="val",
                input_repr=input_repr,
                augment=False,
                extra_other_idxs=extra_other_idxs,
            )
            test_ds, _ = build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="test",
                input_repr=input_repr,
                augment=False,
                extra_other_idxs=extra_other_idxs,
            )
            print(
                f"[{run_type}] training_samples={meta.epoch_samples} "
                f"n_classes={meta.n_classes} class_weights={meta.class_weights}"
            )

            model = model_builder(meta.n_classes)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.epochs,
                verbose=FIT_VERBOSE,
                callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=config.lr_factor,
                        patience=config.lr_patience,
                        min_lr=config.lr_min,
                        verbose=1,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=config.patience,
                        restore_best_weights=True,
                    ),
                ],
            )
            test_loss = float(model.evaluate(test_ds, verbose=0)[0])  # ty:ignore[not-subscriptable]

            # chosen classes first, then "non_target" (folds non_target + extra_other_idxs)
            label_names = [catalog.class_names[i] for i in chosen_idxs] + [NON_TARGET_NAME]

            model_path = (
                config.models_dir
                / f"scaling_{len(chosen_idxs)}"
                / f"{model_name}.keras"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)

            from building.models import model_eval as M
            from building.models.bake import bake_model
            from building import results_io as R

            tflite_path = model_path.with_suffix(".tflite")
            tflite_stats = bake_model(
                model, val_ds, tflite_path, n_representative=100, verbose=False
            )
            cmp = M.compare_float_vs_quantized(
                model, tflite_path, test_ds, label_names,
                threshold=config.threshold, threshold_mode="fixed",
                non_target_names=[NON_TARGET_NAME],
            )
            y_true_float = cmp.float_eval.y_true
            y_pred_float = cmp.float_eval.y_score
            y_pred_quant = cmp.quant_eval.y_score

            # quantized BCE is the canonical "loss" (deployment reality)
            quant_loss = float(
                tf.keras.losses.BinaryCrossentropy()(y_true_float, y_pred_quant).numpy()
            )

            m = compute_metrics(
                y_true_float, y_pred_quant, config.threshold, quant_loss,
                class_names=label_names, non_target_names=[NON_TARGET_NAME],
            )
            float_m = compute_metrics(
                y_true_float, y_pred_float, config.threshold, test_loss,
                class_names=label_names, non_target_names=[NON_TARGET_NAME],
            )

            npz_dir = config.results_file.parent / "arrays"
            npz_dir.mkdir(parents=True, exist_ok=True)
            npz_key = (
                f"baseline_{extra.get('target_idx')}"
                if run_type == "baseline"
                else f"scaling_k{extra.get('k')}_s{extra.get('sample_idx')}_{'_'.join(str(i) for i in chosen_idxs)}"
            )
            npz_path = npz_dir / f"{npz_key}.npz"
            np.savez_compressed(
                npz_path,
                **R.build_arrays_npz(
                    float_eval=cmp.float_eval, quant_eval=cmp.quant_eval
                ),
            )

            quant_stats = QuantStats(
                model_size_kb=float(tflite_stats.model_size_kb),
                arena_size_kb=(
                    None if tflite_stats.arena_size_kb is None
                    else float(tflite_stats.arena_size_kb)
                ),
                flops_mflops=float(tflite_stats.flops_mflops),
                input_dtype=tflite_stats.input_dtype,
                output_dtype=tflite_stats.output_dtype,
                input_shape=list(tflite_stats.input_shape),
                tflite_path=str(tflite_path),
            )

            common: dict[str, Any] = {
                "collection": config.collection,
                "build_model": config.build_model,
                "epochs_trained": len(history.history.get("loss", [])),
                "model_path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config.model_dump(mode="json"),
                "label_names": label_names,
                "float_metrics": float_m,
                "quant_stats": quant_stats,
                "tflite_path": str(tflite_path),
                "npz_path": str(npz_path),
                **extra,
            }
            res_cls = BaselineRunResult if run_type == "baseline" else ScalingResult
            res = res_cls(run_type=run_type, metrics=m, **common)  # ty:ignore[invalid-argument-type]
            with config.results_file.open("a", encoding="utf-8") as f:
                f.write(res.model_dump_json() + "\n")
            existing.append(res)
            produced.append(res)
        finally:
            # drop dataset refs before clear_session so iterator/shuffle-buffer
            # state isn't pinned across session teardown
            del model, history, train_ds, val_ds, test_ds
            tf.keras.backend.clear_session()
            gc.collect()

    if run_baseline:
        species_classes = [
            (i, n) for i, n in enumerate(catalog.class_names) if n != "non_target"
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
        target_idxs_pool = [
            i for i, n in enumerate(catalog.class_names) if n != "non_target"
        ]
        for k in k_values:
            if k < 2 or k > len(target_idxs_pool):
                continue
            for sample_idx in range(n_samples):
                if is_done("scaling", k=k, sample_idx=sample_idx):
                    continue
                chosen_idxs = rng.choice(
                    target_idxs_pool, size=k, replace=False
                ).tolist()
                unchosen = [i for i in target_idxs_pool if i not in chosen_idxs]
                n_extras = min(config.extras_into_other, len(unchosen))
                extra_other_idxs = (
                    rng.choice(unchosen, size=n_extras, replace=False).tolist()
                    if n_extras > 0
                    else []
                )
                chosen_names = [catalog.class_names[i] for i in chosen_idxs]
                extra_names = [catalog.class_names[i] for i in extra_other_idxs]
                print(
                    f"[scaling] k={k} sample={sample_idx + 1}/{n_samples} "
                    f"chosen={chosen_names} extras_into_other={extra_names}"
                )
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
                        "extra_other_indices": extra_other_idxs,
                        "extra_other_classes": extra_names,
                    },
                    extra_other_idxs=extra_other_idxs,
                )

    return produced


def print_baselines(catalog: DatasetCatalog, results_file: Path) -> None:
    try:
        df = pd.read_json(results_file, lines=True)
    except Exception as e:
        print(f"Could not read results file: {e}")
        return

    baseline_df = df[df["run_type"] == "baseline"].copy()
    class_names = list(catalog.class_names)

    max_cls_len = max((len(str(cls)) for cls in class_names), default=0)
    col_width = max(max_cls_len + 2, 15)

    header = f"{'Target Class':<{col_width}} | {'Precision':>9} | {'Recall':>9} | {'Epochs':>6} | {'Timestamp'}"
    print(header)
    print("-" * len(header))

    for cls in class_names:
        cls_str = f"'{cls}'"
        matches = baseline_df[baseline_df["target_class"] == cls]

        if matches.empty:
            continue

        row = matches.iloc[-1]
        metrics = row["metrics"]

        prec = metrics["precision_mean"]
        prec_std = metrics["precision_std"]

        rec = metrics["recall_mean"]
        rec_std = metrics["recall_std"]

        f1 = metrics["f1_mean"]
        f1_std = metrics["f1_std"]

        epochs = row["epochs_trained"]
        timestamp = row["timestamp"]
        print(
            f"{cls_str:<{col_width}} | {prec:>9.4f} ± {prec_std:>9.4f} | {rec:>9.4f} ± {rec_std:>9.4f} | {f1:>9.4f} ± {f1_std:>9.4f} | {epochs:>6.0f} "
        )


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
