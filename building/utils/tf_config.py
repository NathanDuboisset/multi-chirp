"""TensorFlow runtime config: seeds + GPU memory growth."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf

from .constants import SEED


def set_global_seed(seed: int = SEED) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_tf_runtime() -> None:
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
