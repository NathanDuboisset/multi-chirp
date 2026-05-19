"""INT8 TFLite export helpers for Keras / SavedModel inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras


def build_representative_batches(
    dataset: tf.data.Dataset,
    target_len: int,
    take: int = 100,
) -> List[np.ndarray]:
    batches: List[np.ndarray] = []
    for x_batch, _ in dataset.unbatch().take(take):
        sample = x_batch.numpy().astype(np.float32)
        sample = np.reshape(sample, (1, target_len, 1))
        batches.append(sample)
    return batches


def representative_dataset_from_batches(
    batches: Iterable[np.ndarray],
) -> Callable[[], Iterable[List[np.ndarray]]]:
    def gen():
        for sample in batches:
            yield [sample]

    return gen


def export_int8_tflite_from_saved_model(
    saved_model_dir: str,
    out_tflite: Path,
    rep_batches: Iterable[np.ndarray],
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_from_batches(rep_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    out_tflite.write_bytes(converter.convert())


def export_keras_model_to_int8_tflite(
    model: keras.Model,
    rep_batches: Iterable[np.ndarray],
    out_tflite: Path,
    tmp_dir: str = "temp_saved_model",
) -> None:
    model.export(tmp_dir)
    export_int8_tflite_from_saved_model(tmp_dir, out_tflite, rep_batches)
