from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from datetime import datetime, timezone

from building.training import MacroBlock

if TYPE_CHECKING:
    from building.models.bake import TFLiteStats
    from building.models.model_eval import EvalResult, QuantComparison

SWEEP_THRESHOLDS = np.linspace(0.0, 1.0, 101)
RECORD_SCHEMA_VERSION = 2


class PerClassBlock(BaseModel):
    support: int
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    auc: float | None = None

    @classmethod
    def from_metric(cls, m: Any) -> "PerClassBlock":
        return cls(
            support=int(m.support),
            threshold=float(m.threshold),
            precision=float(m.precision),
            recall=float(m.recall),
            f1=float(m.f1),
            f2=float(m.f2),
            auc=None if m.auc is None else float(m.auc),
        )


class BackendMetrics(BaseModel):
    backend: Literal["keras", "tflite"]
    threshold_mode: str
    loss: float | None = None
    top1_accuracy: float
    subset_accuracy: float
    avg_inference_ms: float
    macro_all: MacroBlock
    macro_targets: MacroBlock | None = None
    non_target_names: list[str] = Field(default_factory=list)
    per_class: dict[str, PerClassBlock]

    @classmethod
    def from_eval(
        cls, e: "EvalResult", *, loss: float | None = None
    ) -> "BackendMetrics":
        macro_targets: MacroBlock | None = None
        if e.non_target_names:
            macro_targets = MacroBlock(
                precision=float(e.macro_precision_targets or 0.0),
                recall=float(e.macro_recall_targets or 0.0),
                f1=float(e.macro_f1_targets or 0.0),
                f2=float(e.macro_f2_targets or 0.0),
                auc=e.macro_auc_targets,
            )
        return cls(
            backend=e.backend,
            threshold_mode=e.threshold_mode,
            loss=loss,
            top1_accuracy=float(e.top1_accuracy),
            subset_accuracy=float(e.subset_accuracy),
            avg_inference_ms=float(e.avg_inference_ms),
            macro_all=MacroBlock(
                precision=float(e.macro_precision),
                recall=float(e.macro_recall),
                f1=float(e.macro_f1),
                f2=float(e.macro_f2),
                auc=e.macro_auc,
            ),
            macro_targets=macro_targets,
            non_target_names=list(e.non_target_names),
            per_class={n: PerClassBlock.from_metric(m) for n, m in e.per_class.items()},
        )


class MacroDelta(BaseModel):
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    f2: float | None = None
    auc: float | None = None


class PerClassDelta(BaseModel):
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    f2: float | None = None
    auc: float | None = None


class DeltaBlock(BaseModel):
    top1_accuracy: float | None = None
    subset_accuracy: float | None = None
    loss: float | None = None
    macro_all: MacroDelta
    macro_targets: MacroDelta | None = None
    per_class: dict[str, PerClassDelta]

    @classmethod
    def from_backends(
        cls, f: BackendMetrics, q: BackendMetrics
    ) -> "DeltaBlock":
        def d(a: float | None, b: float | None) -> float | None:
            return None if (a is None or b is None) else float(b) - float(a)

        macro_targets: MacroDelta | None = None
        if f.macro_targets is not None and q.macro_targets is not None:
            macro_targets = MacroDelta(
                **{
                    k: d(getattr(f.macro_targets, k), getattr(q.macro_targets, k))
                    for k in ("precision", "recall", "f1", "f2", "auc")
                }
            )
        per_class = {
            name: PerClassDelta(
                **{
                    k: d(getattr(f.per_class[name], k), getattr(q.per_class[name], k))
                    for k in ("precision", "recall", "f1", "f2", "auc")
                }
            )
            for name in f.per_class
        }
        return cls(
            top1_accuracy=d(f.top1_accuracy, q.top1_accuracy),
            subset_accuracy=d(f.subset_accuracy, q.subset_accuracy),
            loss=d(f.loss, q.loss),
            macro_all=MacroDelta(
                **{
                    k: d(getattr(f.macro_all, k), getattr(q.macro_all, k))
                    for k in ("precision", "recall", "f1", "f2", "auc")
                }
            ),
            macro_targets=macro_targets,
            per_class=per_class,
        )


class TFLiteStatsBlock(BaseModel):
    path: str
    model_size_kb: float
    arena_size_kb: float | None = None
    flops_mflops: float
    input_dtype: str
    output_dtype: str
    input_shape: list[int]

    @classmethod
    def from_stats(cls, stats: "TFLiteStats | None") -> "TFLiteStatsBlock | None":
        if stats is None:
            return None
        return cls(
            path=str(stats.path),
            model_size_kb=float(stats.model_size_kb),
            arena_size_kb=(
                None if stats.arena_size_kb is None else float(stats.arena_size_kb)
            ),
            flops_mflops=float(stats.flops_mflops),
            input_dtype=str(stats.input_dtype),
            output_dtype=str(stats.output_dtype),
            input_shape=[int(s) for s in stats.input_shape],
        )


class ThresholdCurve(BaseModel):
    threshold: list[float]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    f2: list[float]
    accuracy: list[float]


class CurvesBlock(BaseModel):
    thresholds: list[float]
    per_class: dict[str, ThresholdCurve]

    @classmethod
    def from_eval(cls, e: "EvalResult", *, target_only: bool) -> "CurvesBlock":
        from building.models.model_eval import threshold_curve

        excluded = set(e.non_target_names) if target_only else set()
        per_class: dict[str, ThresholdCurve] = {}
        for c, name in enumerate(e.label_names):
            if name in excluded:
                continue
            yt = e.y_true[:, c]
            if (yt >= 0.5).sum() == 0:
                continue
            curve = threshold_curve(yt, e.y_score[:, c], SWEEP_THRESHOLDS)
            per_class[name] = ThresholdCurve(
                threshold=curve["threshold"].tolist(),
                precision=curve["precision"].tolist(),
                recall=curve["recall"].tolist(),
                f1=curve["f1"].tolist(),
                f2=curve["f2"].tolist(),
                accuracy=curve["accuracy"].tolist(),
            )
        return cls(
            thresholds=SWEEP_THRESHOLDS.tolist(),
            per_class=per_class,
        )


class ROCCurve(BaseModel):
    fpr: list[float]
    tpr: list[float]
    auc: float


def _roc_block(e: "EvalResult", *, target_only: bool) -> dict[str, ROCCurve]:
    from building.models.model_eval import roc_arrays

    excluded = set(e.non_target_names) if target_only else set()
    out: dict[str, ROCCurve] = {}
    for c, name in enumerate(e.label_names):
        if name in excluded:
            continue
        r = roc_arrays(e.y_true[:, c], e.y_score[:, c])
        if r["auc"] is None:
            continue
        out[name] = ROCCurve(
            fpr=[float(v) for v in r["fpr"]],
            tpr=[float(v) for v in r["tpr"]],
            auc=float(r["auc"]),
        )
    return out


def _prediction_rates(e: "EvalResult") -> list[list[float]]:
    # Diagonal = recall; off-diagonal cells can sum past 1 (predictions are
    # independent multi-label sigmoids).
    n = len(e.label_names)
    y_true_bin = (e.y_true >= 0.5).astype(np.int32)
    thresholds = np.array(
        [e.per_class[name].threshold for name in e.label_names], dtype=float
    )
    y_pred_bin = (e.y_score >= thresholds[None, :]).astype(np.int32)
    rates = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mask = y_true_bin[:, i] == 1
        if mask.any():
            rates[i] = y_pred_bin[mask].mean(axis=0)
    return rates.tolist()


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[2] = 2
    class_names: list[str]
    non_target_names: list[str]
    float_: BackendMetrics = Field(alias="float")
    quantized: BackendMetrics
    deltas: DeltaBlock
    tflite_stats: TFLiteStatsBlock | None = None
    curves_quantized: CurvesBlock
    roc_quantized: dict[str, ROCCurve] = Field(default_factory=dict)
    prediction_rates_quantized: list[list[float]] | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Transient, never serialized. Set by from_comparison so save() can flush
    # the raw arrays without the caller having to thread them through again.
    _cmp: "QuantComparison | None" = PrivateAttr(default=None)
    _extra_arrays: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_comparison(
        cls,
        cmp: "QuantComparison",
        *,
        non_target_names: Iterable[str],
        tflite_stats: "TFLiteStats | None" = None,
        losses: tuple[float | None, float | None] = (None, None),
        extra_arrays: dict[str, np.ndarray] | None = None,
        **meta: Any,
    ) -> "RunRecord":
        nt = list(non_target_names)
        float_loss, quant_loss = losses
        float_block = BackendMetrics.from_eval(cmp.float_eval, loss=float_loss)
        quant_block = BackendMetrics.from_eval(cmp.quant_eval, loss=quant_loss)
        payload: dict[str, Any] = {
            "class_names": list(cmp.float_eval.label_names),
            "non_target_names": nt,
            "float": float_block,
            "quantized": quant_block,
            "deltas": DeltaBlock.from_backends(float_block, quant_block),
            "tflite_stats": TFLiteStatsBlock.from_stats(tflite_stats),
            "curves_quantized": CurvesBlock.from_eval(cmp.quant_eval, target_only=True),
            "roc_quantized": _roc_block(cmp.quant_eval, target_only=True),
            "prediction_rates_quantized": _prediction_rates(cmp.quant_eval),
        }
        payload.update(meta)
        rec = cls.model_validate(payload)
        rec._cmp = cmp
        if extra_arrays:
            rec._extra_arrays = dict(extra_arrays)
        return rec

    def save(
        self,
        results_file: Path | str,
        *,
        with_arrays: bool = True,
        npz_file: Path | str | None = None,
    ) -> Path:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(self.model_dump_json(indent=2, by_alias=True))
        if with_arrays:
            if self._cmp is None:
                raise ValueError(
                    "Cannot write arrays, record was not built from a QuantComparison."
                )
            npz_path = (
                Path(npz_file)
                if npz_file is not None
                else results_path.with_suffix(".npz")
            )
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = build_arrays_npz(
                float_eval=self._cmp.float_eval, quant_eval=self._cmp.quant_eval
            )
            arrays.update(self._extra_arrays)
            np.savez_compressed(npz_path, **arrays)
        return results_path

    def save_jsonl(
        self,
        results_file: Path | str,
        *,
        npz_dir: Path | str | None = None,
        npz_key: str | None = None,
    ) -> Path:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(self.model_dump_json(by_alias=True) + "\n")
        if self._cmp is not None and npz_dir is not None and npz_key is not None:
            d = Path(npz_dir)
            d.mkdir(parents=True, exist_ok=True)
            arrays = build_arrays_npz(
                float_eval=self._cmp.float_eval, quant_eval=self._cmp.quant_eval
            )
            arrays.update(self._extra_arrays)
            np.savez_compressed(d / f"{npz_key}.npz", **arrays)
        return results_path


def build_arrays_npz(
    *,
    float_eval: "EvalResult",
    quant_eval: "EvalResult",
    target_only: bool = False,
) -> dict[str, np.ndarray]:
    from building.models.model_eval import roc_arrays, threshold_curve

    out: dict[str, np.ndarray] = {
        "label_names": np.array(float_eval.label_names),
        "non_target_names": np.array(list(float_eval.non_target_names)),
        "thresholds": SWEEP_THRESHOLDS.copy(),
        "y_true": float_eval.y_true.astype(np.float32),
        "y_score_float": float_eval.y_score.astype(np.float32),
        "y_score_quant": quant_eval.y_score.astype(np.float32),
    }
    excluded = set(float_eval.non_target_names) if target_only else set()
    for c, name in enumerate(quant_eval.label_names):
        if name in excluded:
            continue
        yt = quant_eval.y_true[:, c]
        ysq = quant_eval.y_score[:, c]
        ysf = float_eval.y_score[:, c]
        if (yt >= 0.5).sum() == 0:
            continue
        r = roc_arrays(yt, ysq)
        if r["auc"] is not None:
            out[f"roc_fpr__{name}"] = r["fpr"].astype(np.float32)
            out[f"roc_tpr__{name}"] = r["tpr"].astype(np.float32)
            out[f"roc_thresholds__{name}"] = r["thresholds"].astype(np.float32)
        for label, ys in (("quant", ysq), ("float", ysf)):
            curve = threshold_curve(yt, ys, SWEEP_THRESHOLDS)
            for metric, arr in curve.items():
                if metric == "threshold":
                    continue
                out[f"sweep_{label}_{metric}__{name}"] = arr.astype(np.float32)
    return out




class LoadedRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record: RunRecord
    float_eval: Any  # building.models.model_eval.EvalResult
    quant_eval: Any
    cmp: Any         # building.models.model_eval.QuantComparison
    label_names: list[str]
    non_target_names: list[str]
    npz: Any         # numpy.lib.npyio.NpzFile (lazy-loaded view of the sidecar)

    @property
    def meta(self) -> dict[str, Any]:
        return self.record.model_dump(mode="json", by_alias=True)


def _eval_from_block(
    *,
    backend: Literal["keras", "tflite"],
    block: BackendMetrics,
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    non_target_names: list[str],
    tflite_stats: "TFLiteStats | None",
) -> "EvalResult":
    from building.models.model_eval import eval_result_from_predictions

    thresholds = np.array(
        [
            float(block.per_class[n].threshold) if n in block.per_class else 0.5
            for n in label_names
        ],
        dtype=float,
    )
    return eval_result_from_predictions(
        backend=backend,
        label_names=label_names,
        y_true=y_true,
        y_score=y_score,
        threshold=thresholds,
        threshold_mode=block.threshold_mode,
        non_target_names=non_target_names,
        avg_inference_ms=float(block.avg_inference_ms),
        tflite_stats=tflite_stats,
    )


def load_run(
    json_path: Path | str,
    *,
    npz_path: Path | str | None = None,
) -> LoadedRun:
    from building.models.bake import TFLiteStats
    from building.models.model_eval import QuantComparison

    jp = Path(json_path)
    record = RunRecord.model_validate_json(jp.read_text())
    if record.schema_version != RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"{jp} schema_version={record.schema_version!r} (expected {RECORD_SCHEMA_VERSION})."
        )

    np_path = Path(npz_path) if npz_path is not None else jp.with_suffix(".npz")
    if not np_path.exists():
        raise FileNotFoundError(f"NPZ sidecar not found: {np_path}")
    npz = np.load(np_path, allow_pickle=False)

    label_names = list(record.class_names)
    non_target_names = list(record.non_target_names)

    tflite_stats: TFLiteStats | None = None
    if record.tflite_stats is not None:
        tflite_stats = TFLiteStats.model_validate(
            record.tflite_stats.model_dump(mode="python")
        )

    float_eval = _eval_from_block(
        backend="keras",
        block=record.float_,
        y_true=npz["y_true"],
        y_score=npz["y_score_float"],
        label_names=label_names,
        non_target_names=non_target_names,
        tflite_stats=None,
    )
    quant_eval = _eval_from_block(
        backend="tflite",
        block=record.quantized,
        y_true=npz["y_true"],
        y_score=npz["y_score_quant"],
        label_names=label_names,
        non_target_names=non_target_names,
        tflite_stats=tflite_stats,
    )
    return LoadedRun(
        record=record,
        float_eval=float_eval,
        quant_eval=quant_eval,
        cmp=QuantComparison(float_eval=float_eval, quant_eval=quant_eval),
        label_names=label_names,
        non_target_names=non_target_names,
        npz=npz,
    )
