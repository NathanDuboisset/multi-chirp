"""SincNet on raw waveform."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from keras import Model, layers

from building.models._common import SAMPLE_RATE, TARGET_AUDIO_LEN, compile_model

NUM_FILTERS = 32
KERNEL_SIZE = 64
STRIDE = 16
CONV_FILTERS = 16
CONV_FILTER_SIZE = 8
CONV_STRIDE = 2
DENSE_HIDDEN = 64
DROPOUT = 0.3


class SincnetConv(layers.Layer):
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


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    num_filters: int = NUM_FILTERS,
    kernel_size: int = KERNEL_SIZE,
    stride: int = STRIDE,
    conv_filters: int = CONV_FILTERS,
    conv_filter_size: int = CONV_FILTER_SIZE,
    conv_stride: int = CONV_STRIDE,
    dense_hidden: int = DENSE_HIDDEN,
    dropout: float = DROPOUT,
) -> Model:
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Reshape((input_len, 1, 1), name="to_nhwc")(inp)

    x = SincnetConv(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        sample_rate=SAMPLE_RATE,
        name="sincnet_convolution",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="envelope_pool")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(conv_filter_size, 1),
        strides=(conv_stride, 1),
        padding="same",
        name="temporal_conv_1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(4, 1), name="temporal_pool_1")(x)

    x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(conv_filter_size, 1),
        strides=(conv_stride, 1),
        padding="same",
        name="temporal_conv_2",
    )(x)
    x = layers.ReLU()(x)

    x = layers.AveragePooling2D(
        pool_size=(x.shape[1], 1), padding="valid", name="final_pool"
    )(x)
    x = layers.Flatten(name="flatten")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_pre_dense")(x)
    x = layers.Dense(dense_hidden, activation="relu", name="dense_hidden")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_post_dense")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="sincnet"))
