"""Shared model constants + compile helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import keras
else:
    import tensorflow as tf

    keras = tf.keras

from keras import Model

from building.utils import (
    NUM_MEL_BINS_MEL as NUM_MEL_BINS,
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TARGET_FRAMES_MEL as TARGET_FRAMES,
)

MEL_INPUT_SHAPE = (TARGET_FRAMES, NUM_MEL_BINS, 1)


def compile_model(model: Model) -> Model:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
