"""Bake a trained Keras model into an INT8-quantized TFLite flatbuffer.

The flow mirrors tiny-chirp-microflow/building_tensorflow: a Keras model plus a
representative dataset are converted to a fully-INT8 .tflite with int8 I/O. A
post-conversion analyzer pulls out the on-flash weight bytes, RAM arena size
and a 4-D-tensor MFLOPs estimate so the same numbers can be reported next to
the float accuracy.

The representative_dataset is shape-agnostic — the existing
`building.utils.export.build_representative_batches` hard-codes a 1-D waveform
reshape, so mel inputs cannot use it. `bake_model` wraps the conversion with
the right shape handling and writes to `models/<collection>/<stem>.tflite` by
default.
"""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras

from building.utils import MODELS_DIR


def _iter_features(dataset: tf.data.Dataset) -> Iterable[np.ndarray]:
    """Yield numpy feature samples (no batch dim) from a (features, label[, sw]) dataset."""
    for elt in dataset.unbatch():
        x = elt[0] if isinstance(elt, tuple) else elt
        yield x.numpy().astype(np.float32)


def build_representative_batches(
    dataset: tf.data.Dataset,
    take: int = 100,
) -> List[np.ndarray]:
    """Collect `take` samples and re-add the batch axis.

    Works for any input rank (time-domain `(T, 1)`, mel `(F, M, 1)`, etc.).
    """
    batches: List[np.ndarray] = []
    for i, x in enumerate(_iter_features(dataset)):
        if i >= take:
            break
        batches.append(x[None, ...])
    return batches


def representative_dataset_from_batches(
    batches: Iterable[np.ndarray],
) -> Callable[[], Iterable[List[np.ndarray]]]:
    def gen():
        for sample in batches:
            yield [sample]

    return gen


@dataclass
class TFLiteStats:
    path: Path
    model_size_kb: float
    arena_size_kb: float | None
    flops_mflops: float
    input_dtype: str
    output_dtype: str
    input_shape: tuple[int, ...]


_DTYPE_BYTES = {"INT8": 1, "UINT8": 1, "INT16": 2, "INT32": 4, "FLOAT32": 4}


def analyze_tflite(tflite_path: Path) -> TFLiteStats:
    """Run `tf.lite.experimental.Analyzer` and extract size / arena / flops.

    `model_size_kb` is the "Total data buffer size" line (weights + biases,
    excluding flatbuffer overhead — closer to flash usage than file size).
    `arena_size_kb` sums activation tensors marked with a dynamic batch dim;
    if no such tensors exist, it stays None. `flops_mflops` is a coarse
    proxy: sum of products of 4-D tensor shapes / 1e6 (matches the tinychirp
    notebook number used for cross-model comparisons).
    """
    tflite_path = Path(tflite_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_path))
    text = buf.getvalue()

    m = re.search(r"Total data buffer size:\s*(\d+)\s*bytes", text)
    model_size_kb = (
        int(m.group(1)) / 1024.0 if m else tflite_path.stat().st_size / 1024.0
    )

    activation_total = 0
    for line in text.splitlines():
        sig = re.search(r"shape_signature:\[([-\d,\s]+)\],\s*type:(\w+)", line)
        if sig:
            dims = [max(1, int(d)) for d in sig.group(1).split(",")]
            activation_total += int(np.prod(dims)) * _DTYPE_BYTES.get(sig.group(2), 1)
    arena_size_kb = activation_total / 1024.0 if activation_total > 0 else None

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    flops_mflops = sum(
        float(np.prod(t["shape"])) / 1e6
        for t in interp.get_tensor_details()
        if len(t["shape"]) == 4
    )
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return TFLiteStats(
        path=tflite_path,
        model_size_kb=model_size_kb,
        arena_size_kb=arena_size_kb,
        flops_mflops=flops_mflops,
        input_dtype=str(np.dtype(inp["dtype"]).name),
        output_dtype=str(np.dtype(out["dtype"]).name),
        input_shape=tuple(int(s) for s in inp["shape"]),
    )


def bake_model(
    model: "keras.Model",
    rep_dataset: tf.data.Dataset,
    out_path: Path | str,
    *,
    n_representative: int = 100,
    verbose: bool = True,
) -> TFLiteStats:
    """Quantize `model` to INT8 .tflite using `rep_dataset` for calibration.

    `rep_dataset` should be a *non-shuffled* slice of the training split (val
    works too) yielding `(features, label[, sw])` with features already in the
    model's input space (time waveform or log-mel). The first
    `n_representative` samples are used as the calibration set.

    The TFLite I/O dtypes are int8 — downstream evaluation has to quantize
    inputs and dequantize outputs (see `model_eval.evaluate`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_batches = build_representative_batches(rep_dataset, take=n_representative)
    if not rep_batches:
        raise ValueError(
            "Representative dataset is empty — cannot calibrate INT8 quantization."
        )

    sample_shape = rep_batches[0].shape
    expected = tuple(model.input_shape)
    if expected[1:] != sample_shape[1:]:
        raise ValueError(
            f"Representative sample shape {sample_shape} does not match model "
            f"input shape {expected}. Pass the dataset used to train this model."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_signature = [
            tf.TensorSpec(shape=[1] + list(expected[1:]), dtype=tf.float32)
        ]
        model.export(tmp_dir, input_signature=input_signature)

        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_from_batches(
            rep_batches
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        out_path.write_bytes(converter.convert())

    stats = analyze_tflite(out_path)
    if verbose:
        print(f"Baked INT8 TFLite -> {out_path}")
        print(f"  flash (weights)  : {stats.model_size_kb:>8.1f} KB")
        if stats.arena_size_kb is not None:
            print(f"  arena (activ.)   : {stats.arena_size_kb:>8.1f} KB")
        print(f"  est. MFLOPs      : {stats.flops_mflops:>8.3f}")
        print(f"  input shape/dtype: {stats.input_shape} / {stats.input_dtype}")
    return stats


def default_tflite_path(model_stem: str, collection: str | None = None) -> Path:
    """Return `models/<collection>/<stem>.tflite` (or `models/<stem>.tflite`).

    Matches the layout used by `geographic_task/train.ipynb` for `.keras`
    checkpoints, so the baked file sits next to its float counterpart.
    """
    base = MODELS_DIR if collection is None else MODELS_DIR / collection
    return base / f"{model_stem}.tflite"
