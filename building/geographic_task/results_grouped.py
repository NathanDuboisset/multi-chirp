"""Load per-run task results into a tidy DataFrame for cross-run comparison.

Handles three on-disk schemas:

* **v2 standard** — written by the post-PTQ ``train.ipynb``. Top-level
  ``float``, ``quantized``, ``deltas`` blocks; per-class metrics live inside
  each backend's ``per_class``. Canonical "macro" columns surface the
  quantized **target-only** macro (``quantized.macro_targets``); the float
  variant lands in ``float_macro_*``.
* **v2 cascade** — same shape as v2 standard but with ``is_cascade=True``
  and additional ``stage1_*`` / ``stage2_*`` keys in the run-meta.
* **legacy** — pre-PTQ schemas: top-level ``macro`` + ``per_class`` (standard
  train.ipynb) or ``cascade.macro`` + ``cascade.per_class`` (legacy cascading).

The output frame has one row per JSON file with a normalised column set so
all variants compare side by side. Macros are over target species only when
the new schema is detected; legacy rows keep their original "all-classes"
macros so old comparisons stay reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd


_MACRO_FIELDS = ("precision", "recall", "f1", "f2", "auc")
_PER_CLASS_METRICS = ("precision", "recall", "f1", "f2", "auc")


def _parse_v2(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """Parser for the post-PTQ schema (results_io.build_run_record output)."""
    quant = data["quantized"]
    flt = data["float"]
    deltas = data.get("deltas") or {}
    class_names: list[str] = data.get("class_names") or list(quant["per_class"].keys())
    non_target_names = set(data.get("non_target_names") or [])
    is_cascade = bool(data.get("is_cascade") or "stage1_model" in data)

    qmt = quant.get("macro_targets") or {}
    fmt = flt.get("macro_targets") or {}
    qma = quant.get("macro_all") or {}
    fma = flt.get("macro_all") or {}

    row: dict[str, Any] = {
        "file": path.name,
        "schema": "v2",
        "collection": data.get("collection"),
        "variant": "cascading" if is_cascade else "standard",
        "model": (
            f"{data.get('stage1_model')}+{data.get('stage2_model')}"
            if is_cascade
            else data.get("build_model", "?")
        ),
        "subset_accuracy": quant.get("subset_accuracy"),
        "top1_accuracy": quant.get("top1_accuracy"),
        "test_loss": quant.get("loss"),
        "test_loss_float": flt.get("loss"),
        "delta_loss": deltas.get("loss"),
        "epochs_trained": data.get("epochs_trained")
        if isinstance(data.get("epochs_trained"), int)
        else (data.get("epochs_trained") or {}).get("stage2"),
        "timestamp": data.get("timestamp"),
        # Macro columns surface the quantized target-only macro by default —
        # that's the deployment number, computed over target species only.
        **{f"macro_{k}": qmt.get(k) for k in _MACRO_FIELDS},
        **{f"float_macro_{k}": fmt.get(k) for k in _MACRO_FIELDS},
        # Macro-all (includes non_target) — useful for sanity checking.
        **{f"macro_all_{k}": qma.get(k) for k in _MACRO_FIELDS},
        **{f"float_macro_all_{k}": fma.get(k) for k in _MACRO_FIELDS},
    }

    # TFLite footprint (None for cascade — stats live per-stage in extras).
    stats = data.get("tflite_stats") or {}
    row["flash_kb"] = stats.get("model_size_kb")
    row["arena_kb"] = stats.get("arena_size_kb")
    row["mflops"] = stats.get("flops_mflops")

    if is_cascade:
        row["stage1_model"] = data.get("stage1_model")
        row["stage2_model"] = data.get("stage2_model")
        row["fraction_routed_to_stage2"] = data.get("fraction_routed_to_stage2")
    else:
        row["stage1_model"] = None
        row["stage2_model"] = None
        row["fraction_routed_to_stage2"] = None

    # Per-species columns: precision/recall/f1/f2/auc (quantized) +
    # df1/df2 deltas. Skip the non_target class.
    pc_q = quant.get("per_class") or {}
    pc_d = (deltas or {}).get("per_class") or {}
    for name in class_names:
        if name in non_target_names:
            continue
        q = pc_q.get(name) or {}
        d = pc_d.get(name) or {}
        for metric in _PER_CLASS_METRICS:
            row[f"{metric}__{name}"] = q.get(metric)
        row[f"df1__{name}"] = d.get("f1")
        row[f"df2__{name}"] = d.get("f2")
        row[f"support__{name}"] = q.get("support")

    return row


def _parse_legacy_standard(data: dict[str, Any], path: Path) -> dict[str, Any]:
    macro = data.get("macro") or {}
    pc = data.get("per_class") or {}
    row: dict[str, Any] = {
        "file": path.name,
        "schema": "legacy",
        "collection": data.get("collection"),
        "variant": "standard",
        "model": data.get("build_model", "?"),
        "subset_accuracy": data.get("subset_accuracy"),
        "top1_accuracy": None,
        "test_loss": data.get("test_loss"),
        "test_loss_float": data.get("test_loss"),
        "delta_loss": None,
        "epochs_trained": data.get("epochs_trained")
        if isinstance(data.get("epochs_trained"), int)
        else None,
        "timestamp": data.get("timestamp"),
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1"),
        "macro_f2": macro.get("f2"),
        "macro_auc": macro.get("auc_ovr"),
        "stage1_model": None,
        "stage2_model": None,
        "fraction_routed_to_stage2": None,
    }
    for name, m in pc.items():
        for metric in ("precision", "recall", "f1", "f2"):
            row[f"{metric}__{name}"] = m.get(metric)
        row[f"auc__{name}"] = m.get("auc_ovr")
        row[f"support__{name}"] = m.get("support")
    return row


def _parse_legacy_cascade(data: dict[str, Any], path: Path) -> dict[str, Any]:
    block = data["cascade"]
    macro = block.get("macro") or {}
    pc = block.get("per_class") or {}
    row: dict[str, Any] = {
        "file": path.name,
        "schema": "legacy",
        "collection": data.get("collection"),
        "variant": "cascading",
        "model": f"{data.get('stage1_model', '?')}+{data.get('stage2_model', '?')}",
        "subset_accuracy": block.get("subset_accuracy"),
        "top1_accuracy": None,
        "test_loss": None,
        "test_loss_float": None,
        "delta_loss": None,
        "epochs_trained": (data.get("epochs_trained") or {}).get("stage2"),
        "timestamp": data.get("timestamp"),
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1"),
        "macro_f2": macro.get("f2"),
        "macro_auc": macro.get("auc_ovr"),
        "stage1_model": data.get("stage1_model"),
        "stage2_model": data.get("stage2_model"),
        "fraction_routed_to_stage2": block.get("fraction_routed_to_stage2"),
    }
    for name, m in pc.items():
        for metric in ("precision", "recall", "f1", "f2"):
            row[f"{metric}__{name}"] = m.get(metric)
        row[f"auc__{name}"] = m.get("auc_ovr")
        row[f"support__{name}"] = m.get("support")
    return row


def _parse_record(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") == 2 or (
        "float" in data and "quantized" in data
    ):
        return _parse_v2(data, path)
    if "cascade" in data:
        return _parse_legacy_cascade(data, path)
    return _parse_legacy_standard(data, path)


def gather_results(results_dir: Path) -> pd.DataFrame:
    """Load every ``*.json`` file under ``results_dir`` into one DataFrame.

    Rows are sorted by ``macro_f1`` descending so the best models surface at
    the top. Missing files / unreadable JSON are skipped silently.
    """
    rows: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        r = _parse_record(p)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False, na_position="last").reset_index(drop=True)
    return df


def per_class_metric_table(
    df: pd.DataFrame,
    metric: Literal["precision", "recall", "f1", "f2", "auc"] = "f1",
) -> pd.DataFrame:
    """Wide-format per-class table for `metric`.

    Rows = (variant, model), columns = class names. Quantized values when
    the source row is v2; float values for legacy rows.
    """
    prefix = f"{metric}__"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    out = df.set_index(["variant", "model"])[cols].copy()
    out.columns = [c.removeprefix(prefix) for c in out.columns]
    return out


def per_class_delta_table(
    df: pd.DataFrame, metric: Literal["f1", "f2"] = "f2"
) -> pd.DataFrame:
    """Quantized − float per-class delta table for v2 rows."""
    prefix = f"d{metric}__"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    out = df.set_index(["variant", "model"])[cols].copy()
    out.columns = [c.removeprefix(prefix) for c in out.columns]
    return out


# Back-compat alias for code still calling the old name.
per_class_f1_table = per_class_metric_table
