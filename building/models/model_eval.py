"""Unified evaluation for float Keras models and INT8 TFLite flatbuffers.

Both backends present the same `evaluate(...)` surface so a notebook can drop
in either one without touching the metric code. The output (`EvalResult`)
carries everything the geographic_task / scaling notebooks already report —
per-class precision/recall/F1/F2 at a threshold, macro/micro/top-1, a
confusion matrix, an ROC/PR sweep for threshold selection — plus the size /
arena / MFLOPs numbers that only make sense for the baked TFLite version.

`compare_float_vs_quantized` then runs both side-by-side on the same test
loader and produces the float→INT8 drop table and bar chart, matching the
"perf vs quantization" figure in 2407.21453.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import keras
    import matplotlib.figure
    import pandas as pd
else:
    keras = tf.keras

from building.models.bake import TFLiteStats, analyze_tflite

Backend = Literal["keras", "tflite"]
ThresholdMode = Literal["fixed", "best_f1", "best_f2"]


# Prediction backends


def _predict_keras(
    model: "keras.Model", ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    t0 = time.perf_counter()
    n = 0
    for batch in ds:
        xb, yb = batch[0], batch[1]
        probs = model(xb, training=False).numpy()
        y_score.append(probs)
        y_true.append(yb.numpy())
        n += probs.shape[0]
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n) * 1000.0 if n else 0.0
    return np.concatenate(y_true), np.concatenate(y_score), avg_ms


def _predict_tflite(
    tflite_path: Path, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    if in_scale == 0 or out_scale == 0:
        raise ValueError(
            f"{tflite_path} has zero quantization scale — not an INT8 model?"
        )
    qmin, qmax = (-128, 127) if inp["dtype"] == np.int8 else (0, 255)
    in_rank = len(inp["shape"])

    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    times: list[float] = []
    for batch in ds:
        xb, yb = batch[0].numpy(), batch[1].numpy()
        for i in range(xb.shape[0]):
            x = xb[i]
            if x.ndim != in_rank - 1:
                raise ValueError(
                    f"Sample rank {x.ndim} does not match TFLite input rank "
                    f"{in_rank - 1} (input shape {inp['shape']})."
                )
            xq = np.clip(np.round(x / in_scale) + in_zp, qmin, qmax).astype(
                inp["dtype"]
            )[None, ...]
            interp.set_tensor(inp["index"], xq)
            t0 = time.perf_counter()
            interp.invoke()
            times.append(time.perf_counter() - t0)
            raw = interp.get_tensor(out["index"]).astype(np.float32)
            probs = (raw - out_zp) * out_scale
            y_score.append(probs.reshape(-1))
            y_true.append(yb[i])
    avg_ms = float(np.mean(times) * 1000.0) if times else 0.0
    return np.asarray(y_true), np.asarray(y_score), avg_ms


# Metrics


class PerClassMetrics(BaseModel):
    name: str
    threshold: float
    support: int
    precision: float
    recall: float
    f1: float
    f2: float
    auc: float | None = None


class EvalResult(BaseModel):
    """Per-backend evaluation summary + raw scores. Pydantic over dataclass so
    autoreload picks up new fields without a kernel restart."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Backend
    label_names: list[str]
    per_class: dict[str, PerClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_f2: float
    macro_auc: float | None
    top1_accuracy: float
    subset_accuracy: float  # all-classes-correct rate under each class's threshold
    confusion: np.ndarray  # [true_top1, pred_top1]
    avg_inference_ms: float
    threshold_mode: ThresholdMode
    non_target_names: tuple[str, ...] = ()
    # Macros computed over target species only (label_names minus non_target_names).
    macro_precision_targets: float | None = None
    macro_recall_targets: float | None = None
    macro_f1_targets: float | None = None
    macro_f2_targets: float | None = None
    macro_auc_targets: float | None = None
    tflite_stats: TFLiteStats | None = None
    y_true: np.ndarray | None = Field(default=None, repr=False)
    y_score: np.ndarray | None = Field(default=None, repr=False)


def macro_targets(
    per_class: dict[str, PerClassMetrics],
    non_target_names: Iterable[str],
) -> dict[str, float | None]:
    """Mean of per-class metrics restricted to target classes (not in non_target_names).

    Returns {'precision', 'recall', 'f1', 'f2', 'auc'}. AUC is None if any
    included class has AUC None (e.g. constant labels).
    """
    excluded = set(non_target_names)
    targets = [m for name, m in per_class.items() if name not in excluded]
    if not targets:
        return {"precision": None, "recall": None, "f1": None, "f2": None, "auc": None}
    aucs = [m.auc for m in targets]
    return {
        "precision": float(np.mean([m.precision for m in targets])),
        "recall": float(np.mean([m.recall for m in targets])),
        "f1": float(np.mean([m.f1 for m in targets])),
        "f2": float(np.mean([m.f2 for m in targets])),
        "auc": (
            float(np.mean(aucs)) if all(a is not None for a in aucs) else None
        ),
    }


def threshold_curve(
    y_true_c: np.ndarray,
    y_score_c: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-class metric-vs-threshold sweep.

    Returns dict of arrays of shape (len(thresholds),) for:
    threshold, precision, recall, f1, f2, accuracy.
    """
    yt = (np.asarray(y_true_c) >= 0.5).astype(int)
    ys = np.asarray(y_score_c)
    thr = np.asarray(thresholds, dtype=float)

    n = yt.size
    precs = np.empty_like(thr)
    recs = np.empty_like(thr)
    f1s = np.empty_like(thr)
    f2s = np.empty_like(thr)
    accs = np.empty_like(thr)
    pos = int(yt.sum())
    neg = n - pos
    for i, t in enumerate(thr):
        yp = (ys >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = pos - tp
        tn = neg - fp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        d2 = 4 * p + r
        f2 = 5 * p * r / d2 if d2 else 0.0
        precs[i] = p
        recs[i] = r
        f1s[i] = f1
        f2s[i] = f2
        accs[i] = (tp + tn) / n if n else 0.0
    return {
        "threshold": thr,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
        "f2": f2s,
        "accuracy": accs,
    }


def roc_arrays(
    y_true_c: np.ndarray, y_score_c: np.ndarray
) -> dict[str, np.ndarray | float | None]:
    """Per-class ROC: {'fpr', 'tpr', 'thresholds', 'auc'}.

    AUC is None when the class is degenerate (all-positive or all-negative).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    yt = (np.asarray(y_true_c) >= 0.5).astype(int)
    ys = np.asarray(y_score_c)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return {
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "thresholds": np.array([np.inf, -np.inf]),
            "auc": None,
        }
    fpr, tpr, thr = roc_curve(yt, ys)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc": float(roc_auc_score(yt, ys)),
    }


def _best_threshold(
    y_true_c: np.ndarray, y_score_c: np.ndarray, beta: float
) -> tuple[float, float]:
    """Return (threshold, Fbeta) maximising Fbeta on this class's PR curve."""
    from sklearn.metrics import precision_recall_curve

    if y_true_c.sum() == 0:
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(y_true_c, y_score_c)
    prec, rec = prec[:-1], rec[:-1]
    if thr.size == 0:
        return 0.5, 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    fb = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
    i = int(np.argmax(fb))
    return float(thr[i]), float(fb[i])


def _per_class_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
) -> tuple[dict[str, PerClassMetrics], float | None]:
    """Per-class precision/recall/F1/F2/AUC + macro AUC."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, PerClassMetrics] = {}
    aucs: list[float] = []
    for c, name in enumerate(label_names):
        yt = (y_true[:, c] >= 0.5).astype(int)
        ys = y_score[:, c]
        thr = float(thresholds[c])
        yp = (ys >= thr).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        denom2 = 4 * prec + rec
        f2 = 5 * prec * rec / denom2 if denom2 else 0.0
        auc: float | None = None
        if yt.sum() and yt.sum() < yt.size:
            auc = float(roc_auc_score(yt, ys))
            aucs.append(auc)
        out[name] = PerClassMetrics(
            name=name,
            threshold=thr,
            support=int(yt.sum()),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            f2=float(f2),
            auc=auc,
        )
    macro_auc = float(np.mean(aucs)) if aucs else None
    return out, macro_auc


def _resolve_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    threshold: float | np.ndarray,
    mode: ThresholdMode,
) -> np.ndarray:
    """Return a per-class threshold vector for `mode`.

    `fixed`     — broadcast `threshold` to all classes.
    `best_f1/2` — sweep each class's PR curve on the *provided* y_true/y_score
                  and pick its argmax-Fbeta. Pass the *threshold-tuning* split
                  here (train or val), not the test set.
    """
    if mode == "fixed":
        if isinstance(threshold, (int, float)):
            return np.full(len(label_names), float(threshold))
        thr = np.asarray(threshold, dtype=float)
        if thr.shape != (len(label_names),):
            raise ValueError(
                f"threshold array shape {thr.shape} != ({len(label_names)},)"
            )
        return thr

    beta = 1.0 if mode == "best_f1" else 2.0
    thrs = np.empty(len(label_names))
    for c in range(len(label_names)):
        thrs[c], _ = _best_threshold(
            (y_true[:, c] >= 0.5).astype(int), y_score[:, c], beta
        )
    return thrs


def _confusion_top1(y_true: np.ndarray, y_score: np.ndarray, k: int) -> np.ndarray:
    yt = np.argmax(y_true, axis=1)
    yp = np.argmax(y_score, axis=1)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm


def _eval_from_scores(
    *,
    backend: Backend,
    label_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    avg_inference_ms: float,
    threshold_mode: ThresholdMode,
    non_target_names: Iterable[str] = (),
    tflite_stats: TFLiteStats | None = None,
) -> EvalResult:
    """Wrap pre-computed predictions in an EvalResult.

    Shared between `evaluate()` and the cascade orchestration in
    `eval_result_from_predictions()`.
    """
    per_class, macro_auc = _per_class_metrics(
        y_true, y_score, label_names, thresholds
    )

    precisions = [m.precision for m in per_class.values()]
    recalls = [m.recall for m in per_class.values()]
    f1s = [m.f1 for m in per_class.values()]
    f2s = [m.f2 for m in per_class.values()]

    yt_top1 = np.argmax(y_true, axis=1)
    yp_top1 = np.argmax(y_score, axis=1)
    top1 = float(np.mean(yt_top1 == yp_top1))

    yt_bin = (y_true >= 0.5).astype(int)
    yp_bin = (y_score >= thresholds[None, :]).astype(int)
    subset = float(np.mean(np.all(yt_bin == yp_bin, axis=1)))

    cm = _confusion_top1(y_true, y_score, len(label_names))

    nt_tuple = tuple(non_target_names)
    if nt_tuple:
        mt = macro_targets(per_class, nt_tuple)
    else:
        mt = {"precision": None, "recall": None, "f1": None, "f2": None, "auc": None}

    return EvalResult(
        backend=backend,
        label_names=list(label_names),
        per_class=per_class,
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        macro_f2=float(np.mean(f2s)),
        macro_auc=macro_auc,
        top1_accuracy=top1,
        subset_accuracy=subset,
        confusion=cm,
        avg_inference_ms=avg_inference_ms,
        threshold_mode=threshold_mode,
        non_target_names=nt_tuple,
        macro_precision_targets=mt["precision"],
        macro_recall_targets=mt["recall"],
        macro_f1_targets=mt["f1"],
        macro_f2_targets=mt["f2"],
        macro_auc_targets=mt["auc"],
        tflite_stats=tflite_stats,
        y_true=y_true,
        y_score=y_score,
    )


def predict_via_tflite(
    tflite_path: Path | str, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, float]:
    """Public wrapper around the internal int8 inference loop.

    Used by cascade composition: get the same dequantized probabilities the
    evaluator sees, but without running the full evaluate() pipeline.
    """
    return _predict_tflite(Path(tflite_path), ds)


def eval_result_from_predictions(
    *,
    backend: Backend,
    label_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    threshold_mode: ThresholdMode = "fixed",
    threshold_tuning: tuple[np.ndarray, np.ndarray] | None = None,
    non_target_names: Iterable[str] = (),
    avg_inference_ms: float = 0.0,
    tflite_stats: TFLiteStats | None = None,
) -> EvalResult:
    """Build an EvalResult from already-computed predictions.

    Lets the cascade orchestrate stage-1 + stage-2 inference in numpy then
    reuse the same metric machinery as the single-model path.
    """
    if threshold_mode == "fixed":
        thresholds = _resolve_thresholds(
            y_true, y_score, label_names, threshold, "fixed"
        )
    else:
        if threshold_tuning is None:
            tune_y_true, tune_y_score = y_true, y_score
        else:
            tune_y_true, tune_y_score = threshold_tuning
        thresholds = _resolve_thresholds(
            tune_y_true, tune_y_score, label_names, threshold, threshold_mode
        )
    return _eval_from_scores(
        backend=backend,
        label_names=list(label_names),
        y_true=np.asarray(y_true),
        y_score=np.asarray(y_score),
        thresholds=thresholds,
        avg_inference_ms=avg_inference_ms,
        threshold_mode=threshold_mode,
        non_target_names=non_target_names,
        tflite_stats=tflite_stats,
    )


def evaluate(
    model_or_path: "keras.Model | str | Path",
    test_ds: tf.data.Dataset,
    label_names: list[str],
    *,
    threshold: float | np.ndarray = 0.5,
    threshold_mode: ThresholdMode = "fixed",
    threshold_tuning_ds: tf.data.Dataset | None = None,
    non_target_names: Iterable[str] = (),
) -> EvalResult:
    """Run a model end-to-end and return a fully-populated `EvalResult`.

    The first argument can be:
      * a `tf.keras.Model` (float inference via `model(x, training=False)`)
      * a `Path | str` to a `.tflite` flatbuffer (INT8 inference)

    Thresholds:
      * `threshold_mode='fixed'` uses `threshold` (scalar or per-class array).
      * `threshold_mode='best_f1' | 'best_f2'` derives thresholds from
        `threshold_tuning_ds` if given, otherwise from the test set (a known
        peek; warn-worthy if you care about strict generalization).

    `non_target_names` populates the `macro_*_targets` fields on the result
    by averaging only over species not in that set.
    """
    is_tflite = isinstance(model_or_path, (str, Path))
    backend: Backend = "tflite" if is_tflite else "keras"
    if is_tflite:
        tflite_path = Path(model_or_path)
        y_true, y_score, avg_ms = _predict_tflite(tflite_path, test_ds)
        tflite_stats: TFLiteStats | None = analyze_tflite(tflite_path)
    else:
        y_true, y_score, avg_ms = _predict_keras(model_or_path, test_ds)
        tflite_stats = None

    if threshold_mode == "fixed":
        thresholds = _resolve_thresholds(
            y_true, y_score, label_names, threshold, "fixed"
        )
    else:
        if threshold_tuning_ds is None:
            tune_y_true, tune_y_score = y_true, y_score
        elif is_tflite:
            tune_y_true, tune_y_score, _ = _predict_tflite(
                tflite_path, threshold_tuning_ds
            )
        else:
            tune_y_true, tune_y_score, _ = _predict_keras(
                model_or_path, threshold_tuning_ds
            )
        thresholds = _resolve_thresholds(
            tune_y_true, tune_y_score, label_names, threshold, threshold_mode
        )

    return _eval_from_scores(
        backend=backend,
        label_names=list(label_names),
        y_true=y_true,
        y_score=y_score,
        thresholds=thresholds,
        avg_inference_ms=avg_ms,
        threshold_mode=threshold_mode,
        non_target_names=non_target_names,
        tflite_stats=tflite_stats,
    )


# Reporting helpers


def per_class_table(result: EvalResult) -> "pd.DataFrame":
    import pandas as pd

    rows = [
        {
            "class": m.name,
            "support": m.support,
            "threshold": round(m.threshold, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "f2": round(m.f2, 4),
            "auc": None if m.auc is None else round(m.auc, 4),
        }
        for m in result.per_class.values()
    ]
    return pd.DataFrame(rows)


def summary(result: EvalResult) -> dict[str, Any]:
    s: dict[str, Any] = {
        "backend": result.backend,
        "threshold_mode": result.threshold_mode,
        "top1_accuracy": result.top1_accuracy,
        "subset_accuracy": result.subset_accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "macro_f2": result.macro_f2,
        "macro_auc": result.macro_auc,
        "avg_inference_ms": result.avg_inference_ms,
    }
    if result.non_target_names:
        s["non_target_names"] = list(result.non_target_names)
        s["macro_precision_targets"] = result.macro_precision_targets
        s["macro_recall_targets"] = result.macro_recall_targets
        s["macro_f1_targets"] = result.macro_f1_targets
        s["macro_f2_targets"] = result.macro_f2_targets
        s["macro_auc_targets"] = result.macro_auc_targets
    if result.tflite_stats is not None:
        s["model_size_kb"] = result.tflite_stats.model_size_kb
        s["arena_size_kb"] = result.tflite_stats.arena_size_kb
        s["flops_mflops"] = result.tflite_stats.flops_mflops
    return s


def print_summary(result: EvalResult) -> None:
    print(f"=== {result.backend.upper()} evaluation ===")
    print(f"  threshold mode    : {result.threshold_mode}")
    print(f"  top-1 accuracy    : {result.top1_accuracy:.4f}")
    print(f"  subset accuracy   : {result.subset_accuracy:.4f}")
    print(f"  macro precision   : {result.macro_precision:.4f}")
    print(f"  macro recall      : {result.macro_recall:.4f}")
    print(f"  macro F1          : {result.macro_f1:.4f}")
    print(f"  macro F2          : {result.macro_f2:.4f}")
    if result.macro_auc is not None:
        print(f"  macro AUC         : {result.macro_auc:.4f}")
    print(f"  avg inference (ms): {result.avg_inference_ms:.3f}")
    if result.tflite_stats is not None:
        ts = result.tflite_stats
        print(f"  flash (weights)   : {ts.model_size_kb:.1f} KB")
        if ts.arena_size_kb is not None:
            print(f"  arena (activ.)    : {ts.arena_size_kb:.1f} KB")
        print(f"  est. MFLOPs       : {ts.flops_mflops:.3f}")


# Threshold-selection plots


def _display_classes(
    result: EvalResult, classes: Iterable[str] | None, target_only: bool
) -> list[str]:
    """Resolve which classes to plot — explicit list wins, then target_only."""
    if classes is not None:
        return list(classes)
    excluded = set(result.non_target_names) if target_only else set()
    return [n for n in result.label_names if n not in excluded]


def plot_threshold_sweep(
    result: EvalResult,
    beta: float = 2.0,
    classes: Iterable[str] | None = None,
    target_only: bool = True,
) -> "matplotlib.figure.Figure":
    """Plot Fbeta vs threshold per class, marking the argmax.

    By default skips classes in `result.non_target_names`. Pass `target_only=False`
    to include every class, or `classes=[...]` for an explicit subset.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    target = _display_classes(result, classes, target_only)
    fig, ax = plt.subplots(figsize=(8, 5))
    b2 = beta * beta
    for c, name in enumerate(result.label_names):
        if name not in target:
            continue
        yt = (result.y_true[:, c] >= 0.5).astype(int)
        if yt.sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(yt, result.y_score[:, c])
        prec, rec = prec[:-1], rec[:-1]
        denom = b2 * prec + rec
        fb = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        ax.plot(thr, fb, label=name, linewidth=1.2)
        i = int(np.argmax(fb))
        ax.scatter([thr[i]], [fb[i]], s=20)
    ax.set_xlabel("threshold")
    ax.set_ylabel(f"F{int(beta) if beta == int(beta) else beta}")
    ax.set_title("Per-class threshold sweep")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_roc(
    result: EvalResult,
    classes: Iterable[str] | None = None,
    target_only: bool = True,
) -> "matplotlib.figure.Figure":
    """One ROC per class, log-x to highlight the low-FPR regime.

    Skips classes in `result.non_target_names` by default.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    target = _display_classes(result, classes, target_only)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, name in enumerate(result.label_names):
        if name not in target:
            continue
        yt = (result.y_true[:, c] >= 0.5).astype(int)
        if yt.sum() == 0 or yt.sum() == yt.size:
            continue
        fpr, tpr, _ = roc_curve(yt, result.y_score[:, c])
        auc = result.per_class[name].auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})" if auc else name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC per class")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_metric_sweep_panel(
    result: EvalResult,
    classes: Iterable[str] | None = None,
    target_only: bool = True,
    n_thresholds: int = 101,
) -> "matplotlib.figure.Figure":
    """Five-panel figure (accuracy / precision / recall / F1 / F2 vs threshold).

    Matches Fig 5 of arXiv:2407.21453 — one line per class on each panel.
    Skips classes in `result.non_target_names` by default.
    """
    import matplotlib.pyplot as plt

    target = _display_classes(result, classes, target_only)
    thr = np.linspace(0.0, 1.0, n_thresholds)
    metrics = ("accuracy", "precision", "recall", "f1", "f2")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.4), sharey=True)
    for c, name in enumerate(result.label_names):
        if name not in target:
            continue
        yt = (result.y_true[:, c] >= 0.5).astype(int)
        if yt.sum() == 0:
            continue
        curve = threshold_curve(yt, result.y_score[:, c], thr)
        for ax, m in zip(axes, metrics):
            ax.plot(thr, curve[m], label=name, linewidth=1.0)
    for ax, m in zip(axes, metrics):
        ax.set_title(m)
        ax.set_xlabel("t")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Per-class metrics vs threshold")
    fig.tight_layout()
    return fig


# Float vs quantized comparison


class QuantComparison(BaseModel):
    float_eval: EvalResult
    quant_eval: EvalResult


def compare_float_vs_quantized(
    float_model: "keras.Model",
    tflite_path: Path | str,
    test_ds: tf.data.Dataset,
    label_names: list[str],
    *,
    threshold: float | np.ndarray = 0.5,
    threshold_mode: ThresholdMode = "fixed",
    threshold_tuning_ds: tf.data.Dataset | None = None,
    non_target_names: Iterable[str] = (),
) -> QuantComparison:
    """Evaluate float Keras + INT8 TFLite on the same loader.

    Both runs share the *same* `threshold_mode`. With `best_f1`/`best_f2`,
    each backend tunes its own thresholds on `threshold_tuning_ds`, so the
    drop reflects what you'd actually deploy (each model at its own optimum)
    rather than penalizing INT8 with a float-tuned cutoff.

    `non_target_names` populates target-only macro fields on both results.
    """
    f_eval = evaluate(
        float_model,
        test_ds,
        label_names,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_tuning_ds=threshold_tuning_ds,
        non_target_names=non_target_names,
    )
    q_eval = evaluate(
        Path(tflite_path),
        test_ds,
        label_names,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_tuning_ds=threshold_tuning_ds,
        non_target_names=non_target_names,
    )
    return QuantComparison(float_eval=f_eval, quant_eval=q_eval)


def comparison_table(cmp: QuantComparison) -> "pd.DataFrame":
    """Side-by-side per-class metrics with float→INT8 deltas."""
    import pandas as pd

    rows = []
    for name in cmp.float_eval.label_names:
        f = cmp.float_eval.per_class[name]
        q = cmp.quant_eval.per_class[name]
        rows.append(
            {
                "class": name,
                "support": f.support,
                "f1_float": round(f.f1, 4),
                "f1_int8": round(q.f1, 4),
                "df1": round(q.f1 - f.f1, 4),
                "f2_float": round(f.f2, 4),
                "f2_int8": round(q.f2, 4),
                "df2": round(q.f2 - f.f2, 4),
                "prec_float": round(f.precision, 4),
                "prec_int8": round(q.precision, 4),
                "rec_float": round(f.recall, 4),
                "rec_int8": round(q.recall, 4),
            }
        )
    rows.append(
        {
            "class": "MACRO",
            "support": sum(m.support for m in cmp.float_eval.per_class.values()),
            "f1_float": round(cmp.float_eval.macro_f1, 4),
            "f1_int8": round(cmp.quant_eval.macro_f1, 4),
            "df1": round(cmp.quant_eval.macro_f1 - cmp.float_eval.macro_f1, 4),
            "f2_float": round(cmp.float_eval.macro_f2, 4),
            "f2_int8": round(cmp.quant_eval.macro_f2, 4),
            "df2": round(cmp.quant_eval.macro_f2 - cmp.float_eval.macro_f2, 4),
            "prec_float": round(cmp.float_eval.macro_precision, 4),
            "prec_int8": round(cmp.quant_eval.macro_precision, 4),
            "rec_float": round(cmp.float_eval.macro_recall, 4),
            "rec_int8": round(cmp.quant_eval.macro_recall, 4),
        }
    )
    return pd.DataFrame(rows)


def plot_quantization_drop(
    cmp: QuantComparison,
    metric: Literal["f1", "f2", "precision", "recall"] = "f1",
) -> "matplotlib.figure.Figure":
    """Per-class side-by-side bars (float vs INT8) for `metric`.

    Reproduces the float→quantized comparison figure style used in
    arXiv:2407.21453 — same axes, drop annotated above each pair.
    """
    import matplotlib.pyplot as plt

    names = cmp.float_eval.label_names
    f_vals = np.array(
        [getattr(cmp.float_eval.per_class[n], metric) for n in names]
    )
    q_vals = np.array(
        [getattr(cmp.quant_eval.per_class[n], metric) for n in names]
    )

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names) + 2), 4.5))
    ax.bar(x - w / 2, f_vals, w, label="float", color="#3b6fa3")
    ax.bar(x + w / 2, q_vals, w, label="INT8", color="#d97a3f")
    for xi, fv, qv in zip(x, f_vals, q_vals):
        ax.text(
            xi, max(fv, qv) + 0.01, f"{qv - fv:+.3f}",
            ha="center", va="bottom", fontsize=8, color="#444",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, min(1.05, max(f_vals.max(), q_vals.max()) + 0.12))
    ax.set_ylabel(metric)
    ax.set_title(f"Float vs INT8 quantized — {metric} per class")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def print_comparison(cmp: QuantComparison) -> None:
    f, q = cmp.float_eval, cmp.quant_eval
    print(f"{'metric':<20s} {'float':>10s} {'INT8':>10s} {'delta':>10s}")
    print("-" * 52)

    def row(label: str, fv: float | None, qv: float | None) -> None:
        if fv is None or qv is None:
            return
        print(f"{label:<20s} {fv:>10.4f} {qv:>10.4f} {qv - fv:>+10.4f}")

    row("top-1 accuracy", f.top1_accuracy, q.top1_accuracy)
    row("subset accuracy", f.subset_accuracy, q.subset_accuracy)
    row("macro precision", f.macro_precision, q.macro_precision)
    row("macro recall", f.macro_recall, q.macro_recall)
    row("macro F1", f.macro_f1, q.macro_f1)
    row("macro F2", f.macro_f2, q.macro_f2)
    row("macro AUC", f.macro_auc, q.macro_auc)
    row("avg inference ms", f.avg_inference_ms, q.avg_inference_ms)
    if q.tflite_stats is not None:
        print("-" * 52)
        ts = q.tflite_stats
        print(f"{'flash (weights) KB':<20s} {'':>10s} {ts.model_size_kb:>10.1f}")
        if ts.arena_size_kb is not None:
            print(f"{'arena KB':<20s} {'':>10s} {ts.arena_size_kb:>10.1f}")
        print(f"{'MFLOPs (est.)':<20s} {'':>10s} {ts.flops_mflops:>10.3f}")
