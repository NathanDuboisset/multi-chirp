"""
time_models.py — CNN models operating on raw audio time series.

All models output Dense(n_classes, activation="sigmoid") for multi-label
classification, compiled with binary cross-entropy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import keras
    import tensorflow as tf
else:
    import tensorflow as tf

    keras = tf.keras

from keras import layers, Model

# Matches utils.py constants (16 kHz, 184 frames)
TARGET_AUDIO_LEN = 47_872  # (184 - 1) * 256 + 1024  — same as utils.SAMPLE_RATE


def _compile(model: Model, n_classes: int) -> Model:
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


CNN1D_CONV1_FILTERS = 4
CNN1D_CONV1_KERNEL = 3
CNN1D_POOL_SIZE = 2
CNN1D_POOL_STRIDES = 2
CNN1D_CONV2_FILTERS = 8
CNN1D_CONV2_KERNEL = 3
CNN1D_DENSE_HIDDEN = 64


def build_cnn1d(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    """Lightweight 1-D CNN on raw waveform."""
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Conv1D(CNN1D_CONV1_FILTERS, CNN1D_CONV1_KERNEL, activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=CNN1D_POOL_SIZE, strides=CNN1D_POOL_STRIDES)(x)
    x = layers.Conv1D(CNN1D_CONV2_FILTERS, CNN1D_CONV2_KERNEL, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(CNN1D_DENSE_HIDDEN, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return _compile(Model(inp, out, name="cnn1d"), n_classes)


SINCNET_NUM_FILTERS = 48
SINCNET_DENSE_HIDDEN = 64
SINCNET_KERNEL_SIZE = 32
SINCNET_STRIDE = 8


class SincLayer(layers.Layer):
    def __init__(self, num_filters: int, kernel_size: int, stride: int, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.params = self.add_weight(
            shape=(kernel_size, 1, num_filters),
            initializer="random_normal",
            trainable=True,
            name="sinc_params",
        )

    def get_filters(self) -> tf.Tensor:
        return tf.math.sin(self.params)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv1d(
            inputs, self.get_filters(), stride=self.stride, padding="VALID"
        )


def build_sincnet(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    """SincNet-inspired model: learnable bandpass filters + 1-D CNN."""
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = SincLayer(
        num_filters=SINCNET_NUM_FILTERS,
        kernel_size=SINCNET_KERNEL_SIZE,
        stride=SINCNET_STRIDE,
        name="sinc_frontend",
    )(inp)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(SINCNET_DENSE_HIDDEN, activation="relu")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return _compile(Model(inp, out, name="sincnet"), n_classes)


LEAF_NUM_FILTERS = 32
LEAF_KERNEL_SIZE = 64
LEAF_STRIDE = 64


class GaborConv1D(layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.center_freqs = self.add_weight(
            shape=(1, 1, num_filters), initializer="random_uniform"
        )
        self.bandwidths = self.add_weight(shape=(1, 1, num_filters), initializer="ones")

    def get_filters(self):
        limit = (self.kernel_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.kernel_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        env = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        cos_mod = tf.cos(2.0 * math.pi * self.center_freqs * t)
        sin_mod = tf.sin(2.0 * math.pi * self.center_freqs * t)
        return tf.concat([env * cos_mod, env * sin_mod], axis=-1)

    def call(self, inputs):
        conv = tf.nn.conv1d(inputs, self.get_filters(), stride=1, padding="SAME")
        real, imag = tf.split(conv, 2, axis=-1)
        return tf.square(real) + tf.square(imag)


class GaussianPool1D(layers.Layer):
    def __init__(self, num_filters, pool_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.stride = stride
        self.bandwidths = self.add_weight(
            shape=(1, num_filters, 1), initializer=tf.constant_initializer(0.4)
        )

    def get_filters(self):
        limit = (self.pool_size - 1) / 2.0
        t = tf.cast(tf.linspace(-limit, limit, self.pool_size), tf.float32)
        t = tf.reshape(t, [-1, 1, 1])
        gauss = tf.exp(-0.5 * tf.square(t * self.bandwidths))
        return gauss / tf.reduce_sum(gauss, axis=0, keepdims=True)

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(
            tf.expand_dims(inputs, axis=1),
            tf.expand_dims(self.get_filters(), axis=0),
            strides=[1, 1, self.stride, 1],
            padding="SAME",
        )[:, 0, :, :]


class LogCompression(layers.Layer):
    """Instantly compresses audio without heavy loops. Perfect for CPU."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6

    def call(self, inputs):
        return tf.math.log(inputs + self.epsilon)


def build_leaf(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    """LEAF model natively for 1-D waveform."""
    inputs = layers.Input(shape=(input_len, 1), name="audio")
    x = GaborConv1D(
        num_filters=LEAF_NUM_FILTERS, kernel_size=LEAF_KERNEL_SIZE, name="gabor_conv"
    )(inputs)
    x = GaussianPool1D(
        num_filters=LEAF_NUM_FILTERS,
        pool_size=LEAF_KERNEL_SIZE,
        stride=LEAF_STRIDE,
        name="gauss_pool",
    )(x)
    x = LogCompression(name="log_compress")(x)

    x = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return _compile(Model(inputs, outputs, name="leaf"), n_classes)


def get_time_model(name: str) -> Callable[[int, int], Model]:
    if name == "cnn1d":
        return build_cnn1d
    elif name == "sincnet":
        return build_sincnet
    elif name == "leaf":
        return build_leaf
    else:
        raise ValueError(f"Unknown model: {name}")
