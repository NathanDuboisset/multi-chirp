"""
time_models.py — CNN models operating on raw audio time series.

All models output Dense(n_classes, activation="sigmoid") for multi-label
classification, compiled with binary cross-entropy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import keras
    import tensorflow as tf
else:
    import tensorflow as tf

    keras = tf.keras

from keras import layers, Model

# Matches utils.py constants (16 kHz, 184 frames)
TARGET_AUDIO_LEN = 47_872  # (184 - 1) * 256 + 1024  — same as utils.SAMPLE_RATE
SAMPLE_RATE = 16_000


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


SINCNET_NUM_FILTERS = 32
SINCNET_KERNEL_SIZE = 64
SINCNET_STRIDE = 16
SINCNET_CONV_FILTERS = 16
SINCNET_CONV_FILTER_SIZE = 8
SINCNET_CONV_STRIDE = 2
SINCNET_DENSE_HIDDEN = 64


class SincnetConv(layers.Layer):
    """SincNet-style learnable bandpass filterbank on rank-4 NHWC audio.

    Filters are parameterized by low cutoff (f1) and bandwidth, initialized on
    the mel scale, then composed as the difference of two sincs and multiplied
    by a Hamming window. Time runs along the spatial height axis so the graph
    stays microflow-compatible after baking.
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        stride: int,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.stride = stride
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

    def build(self, _input_shape):
        mel_min = 0.0
        mel_max = self._hz_to_mel(self.sample_rate / 2.0)
        mel_points = np.linspace(mel_min, mel_max, self.num_filters + 1)
        hz_points = self._mel_to_hz(mel_points)
        f1_init = hz_points[:-1] / self.sample_rate
        band_init = np.diff(hz_points) / self.sample_rate

        self.f1 = self.add_weight(
            name="f1",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(f1_init),
            trainable=True,
        )
        self.band = self.add_weight(
            name="band",
            shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(band_init),
            trainable=True,
        )
        t = np.linspace(
            -(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size
        )
        self.t = tf.constant(t, dtype=tf.float32)
        window = 0.54 - 0.46 * np.cos(
            2 * math.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )
        self.window = tf.constant(window, dtype=tf.float32)

    def get_filters(self) -> tf.Tensor:
        f1_safe = tf.math.abs(self.f1)
        f2_safe = f1_safe + tf.math.abs(self.band)

        f1_mat = tf.reshape(f1_safe, (1, -1))
        f2_mat = tf.reshape(f2_safe, (1, -1))
        t_mat = tf.reshape(self.t, (-1, 1))

        pi_t = math.pi * t_mat
        denom = tf.where(t_mat == 0.0, 1.0, pi_t)
        filters = (
            tf.math.sin(2.0 * math.pi * f2_mat * t_mat)
            - tf.math.sin(2.0 * math.pi * f1_mat * t_mat)
        ) / denom

        center_values = 2.0 * (f2_mat - f1_mat)
        mask = tf.cast(t_mat == 0.0, tf.float32)
        filters = filters * (1.0 - mask) + center_values * mask

        filters = filters * tf.reshape(self.window, (-1, 1))
        return tf.reshape(filters, (self.kernel_size, 1, self.num_filters))

    def get_filters_nhwc(self) -> tf.Tensor:
        return tf.reshape(
            self.get_filters(), (self.kernel_size, 1, 1, self.num_filters)
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            inputs,
            self.get_filters_nhwc(),
            strides=[1, self.stride, 1, 1],
            padding="VALID",
            data_format="NHWC",
        )

    def export_to_conv2d(self, name: str = "baked_sinc_conv") -> layers.Conv2D:
        """Bake learned Sinc filters into a static Conv2D for TFLite / microflow."""
        baked = self.get_filters().numpy()
        w = np.reshape(baked, (self.kernel_size, 1, 1, self.num_filters))
        conv_layer = layers.Conv2D(
            filters=self.num_filters,
            kernel_size=(self.kernel_size, 1),
            strides=(self.stride, 1),
            padding="valid",
            use_bias=False,
            name=name,
        )
        conv_layer.build(input_shape=(None, None, 1, 1))
        conv_layer.set_weights([w])
        return conv_layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "sample_rate": self.sample_rate,
            }
        )
        return config

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_sincnet(n_classes: int, input_len: int = TARGET_AUDIO_LEN) -> Model:
    """Multi-layer SincNet: learnable bandpass frontend + 2 temporal Conv2D blocks.

    Input is the project-standard [batch, time, 1]; reshaped internally to
    rank-4 NHWC (time = height) so Conv2D / AveragePooling2D map to ops
    supported by microflow.
    """
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Reshape((input_len, 1, 1), name="to_nhwc")(inp)

    x = SincnetConv(
        num_filters=SINCNET_NUM_FILTERS,
        kernel_size=SINCNET_KERNEL_SIZE,
        stride=SINCNET_STRIDE,
        sample_rate=SAMPLE_RATE,
        name="sincnet_convolution",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="envelope_pool")(x)

    x = layers.Conv2D(
        filters=SINCNET_CONV_FILTERS,
        kernel_size=(SINCNET_CONV_FILTER_SIZE, 1),
        strides=(SINCNET_CONV_STRIDE, 1),
        padding="same",
        name="temporal_conv_1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="temporal_pool_1")(x)

    x = layers.Conv2D(
        filters=SINCNET_CONV_FILTERS,
        kernel_size=(SINCNET_CONV_FILTER_SIZE, 1),
        strides=(SINCNET_CONV_STRIDE, 1),
        padding="same",
        name="temporal_conv_2",
    )(x)
    x = layers.ReLU()(x)

    x = layers.AveragePooling2D(
        pool_size=(x.shape[1], 1), padding="valid", name="final_pool"
    )(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(SINCNET_DENSE_HIDDEN, activation="relu", name="dense_hidden")(x)
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
