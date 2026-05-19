"""LEAF: Gabor filterbank + Gaussian pooling + log compression on raw waveform."""

from __future__ import annotations

import math

import tensorflow as tf
from keras import Model, layers

from building.models._common import TARGET_AUDIO_LEN, compile_model

NUM_FILTERS = 32
KERNEL_SIZE = 64
STRIDE = 64


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6

    def call(self, inputs):
        return tf.math.log(inputs + self.epsilon)


def build(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    inputs = layers.Input(shape=(input_len, 1), name="audio")
    x = GaborConv1D(
        num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, name="gabor_conv"
    )(inputs)
    x = GaussianPool1D(
        num_filters=NUM_FILTERS,
        pool_size=KERNEL_SIZE,
        stride=STRIDE,
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
    return compile_model(Model(inputs, outputs, name="leaf"))
