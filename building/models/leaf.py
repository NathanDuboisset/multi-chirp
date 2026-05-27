"""LEAF: Gabor filterbank + Gaussian pooling + PCEN on raw waveform.

PTQ recipe
----------
Full-integer PTQ collapses this model regardless of `denylisted_ops`,
`int16_activations`, or representative-set diversity (F2 -0.68, AUC -0.31
on the 4-class geographic task). The blame falls on per-tensor activation
scales fundamentally failing to capture per-channel distributions:

  * PCEN learns per-channel `alpha`, `delta`, `root` so the output has wildly
    different ranges per channel.
  * The standalone BatchNormalization right after PCEN cannot fold into a
    preceding op (PCEN is not a Conv), so its per-channel mul/add survive
    as standalone INT8 ops with rmse/scale ≈ 6-8.
  * Even with float fallback for POW/DIV, the downstream global average
    pool + dense layers see catastrophic precision loss (rmse/scale > 40).

Use `bake_model(..., weight_only=True)` — dynamic-range PTQ keeps INT8
weights but leaves activations in float32. Flash 15.7 KB (vs 13.2 KB for
full INT8); inference recovers float accuracy (F2 = 0.944 vs 0.956 float).
"""

from __future__ import annotations

import math

import tensorflow as tf
from keras import Model, layers
from keras.saving import register_keras_serializable

from building.models._common import TARGET_AUDIO_LEN, compile_model

NUM_FILTERS = 32
KERNEL_SIZE = 64
GABOR_STRIDE = 16
POOL_STRIDE = 4  # GABOR_STRIDE * POOL_STRIDE = 64, preserves the original frame rate.
PCEN_SMOOTH_SIZE = 15
EPS = 1e-3


@register_keras_serializable(package="leaf")
class GaborConv1D(layers.Layer):
    def __init__(self, num_filters, kernel_size, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.center_freqs = self.add_weight(
            shape=(1, 1, num_filters), initializer="random_uniform"
        )
        self.bandwidths = self.add_weight(shape=(1, 1, num_filters), initializer="ones")

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
        }

    def get_filters(self):
        limit = (self.kernel_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.kernel_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        env = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        cos_mod = tf.cos(2.0 * math.pi * self.center_freqs * t)
        sin_mod = tf.sin(2.0 * math.pi * self.center_freqs * t)
        return tf.concat([env * cos_mod, env * sin_mod], axis=-1)

    def call(self, inputs):
        conv = tf.nn.conv1d(inputs, self.get_filters(), stride=self.stride, padding="SAME")
        real, imag = tf.split(conv, 2, axis=-1)
        # Magnitude (not energy) keeps dynamic range INT8-friendly.
        return tf.sqrt(tf.square(real) + tf.square(imag) + EPS)


@register_keras_serializable(package="leaf")
class GaussianPool1D(layers.Layer):
    def __init__(self, num_filters, pool_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.stride = stride
        self.bandwidths = self.add_weight(
            shape=(1, num_filters, 1), initializer=tf.constant_initializer(0.4)
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "pool_size": self.pool_size,
            "stride": self.stride,
        }

    def get_filters(self):
        limit = (self.pool_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.pool_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        return gauss / tf.reduce_sum(gauss, axis=0, keepdims=True)

    def call(self, inputs):
        # GPU depthwise_conv2d requires equal row/col strides; emulate
        # `stride=S SAME` via explicit pad + stride=1 VALID + decimate.
        k, s = self.pool_size, self.stride
        t = tf.shape(inputs)[1]
        out_t = -(-t // s)  # ceil(t / s)
        total_pad = tf.maximum((out_t - 1) * s + k - t, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        padded = tf.pad(inputs, [[0, 0], [pad_left, pad_right], [0, 0]])
        smoothed = tf.nn.depthwise_conv2d(
            tf.expand_dims(padded, axis=1),
            tf.expand_dims(self.get_filters(), axis=0),
            strides=[1, 1, 1, 1],
            padding="VALID",
        )[:, 0, :, :]
        if s == 1:
            return smoothed
        return smoothed[:, ::s, :]


@register_keras_serializable(package="leaf")
class PCEN(layers.Layer):
    """Per-Channel Energy Normalization: y = ((x+eps)/(eps+M)^alpha + delta)^r - delta^r.

    M = depthwise Gaussian smoothing of x (parallel, INT8-friendly — no IIR scan).
    Replaces log compression; activation distribution is closer to Gaussian.
    """

    def __init__(self, num_filters, smooth_size=PCEN_SMOOTH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.smooth_size = smooth_size

        self.alpha = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(0.96),
            name="alpha",
        )
        self.delta = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(2.0),
            name="delta",
        )
        self.root = self.add_weight(
            shape=(1, 1, num_filters),
            initializer=tf.constant_initializer(0.5),
            name="root",
        )
        self.smooth_bw = self.add_weight(
            shape=(1, num_filters, 1),
            initializer=tf.constant_initializer(0.2),
            name="smooth_bw",
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_filters": self.num_filters,
            "smooth_size": self.smooth_size,
        }

    def get_smoothing_filters(self):
        limit = (self.smooth_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.smooth_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.smooth_bw))
        return gauss / (tf.reduce_sum(gauss, axis=0, keepdims=True) + 1e-12)

    def call(self, inputs):
        m = tf.nn.depthwise_conv2d(
            tf.expand_dims(inputs, axis=1),
            tf.expand_dims(self.get_smoothing_filters(), axis=0),
            strides=[1, 1, 1, 1],
            padding="SAME",
        )[:, 0, :, :]
        alpha = tf.clip_by_value(self.alpha, 0.0, 1.0)
        root = tf.clip_by_value(self.root, 1e-2, 1.0)
        delta = tf.maximum(self.delta, 0.0)
        smooth = tf.pow(EPS + m, alpha)
        return tf.pow((inputs + EPS) / smooth + delta, root) - tf.pow(delta, root)


def build(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    inputs = layers.Input(shape=(input_len, 1), name="audio")
    x = GaborConv1D(
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        stride=GABOR_STRIDE,
        name="gabor_conv",
    )(inputs)
    x = GaussianPool1D(
        num_filters=NUM_FILTERS,
        pool_size=KERNEL_SIZE,
        stride=POOL_STRIDE,
        name="gauss_pool",
    )(x)
    x = PCEN(num_filters=NUM_FILTERS, name="pcen")(x)
    # BN momentum=0.9: moving stats track val-time distribution within a few
    # epochs; default 0.99 takes ~100 and amplifies train/val mismatch.
    x = layers.BatchNormalization(momentum=0.9, name="pcen_bn")(x)

    # Post-Conv BN is required for INT8 — without it, per-tensor calibration
    # binarizes the sigmoid output.
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inputs, outputs, name="leaf"))
