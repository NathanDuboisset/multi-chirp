"""Load per-run task results into a tidy DataFrame for cross-run comparison.

Handles both schemas produced by the task notebooks:

* **standard** — written by ``train.ipynb``. Top-level keys include
  ``build_model``, ``macro``, ``per_class``, ``subset_accuracy``.
* **cascading** — written by ``cascading_train.ipynb``. Has a ``cascade``
  sub-dict mirroring the standard schema, plus ``stage1_model``,
  ``stage2_model``, ``fraction_routed_to_stage2``.

The output frame has one row per JSON file with a normalised column set so
both variants can be compared side-by-side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_MACRO_FIELDS = ("precision", "recall", "f1", "f2", "auc_ovr")


def _parse_record(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if "cascade" in data:
        variant = "cascading"
        block = data["cascade"]
        model = f"{data.get('stage1_model', '?')}+{data.get('stage2_model', '?')}"
        extras = {
            "stage1_model": data.get("stage1_model"),
            "stage2_model": data.get("stage2_model"),
            "fraction_routed_to_stage2": block.get("fraction_routed_to_stage2"),
        }
    else:
        variant = "standard"
        block = data
        model = data.get("build_model", "?")
        extras = {
            "stage1_model": None,
            "stage2_model": None,
            "fraction_routed_to_stage2": None,
        }

    macro = block.get("macro", {})
    row: dict[str, Any] = {
        "file": path.name,
        "collection": data.get("collection"),
        "variant": variant,
        "model": model,
        "subset_accuracy": block.get("subset_accuracy"),
        "test_loss": data.get("test_loss") if variant == "standard" else None,
        "epochs_trained": (
            data.get("epochs_trained")
            if isinstance(data.get("epochs_trained"), int)
            else (data.get("epochs_trained") or {}).get("stage2")
            if variant == "cascading"
            else None
        ),
        "timestamp": data.get("timestamp"),
        **{f"macro_{k}": macro.get(k) for k in _MACRO_FIELDS},
        **extras,
    }

    for cls, m in (block.get("per_class") or {}).items():
        row[f"f1__{cls}"] = m.get("f1")

    return row


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


def per_class_f1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-format per-class F1 table: rows = (variant, model), cols = class names."""
    cols = [c for c in df.columns if c.startswith("f1__")]
    if not cols:
        return pd.DataFrame()
    out = df.set_index(["variant", "model"])[cols].copy()
    out.columns = [c.removeprefix("f1__") for c in out.columns]
    return out
