"""Single-head Transformer on raw waveform (TinyChirp paper).

Mirrors RawAudioTransformerModel from
github.com/TinyPART/TinyChirp/blob/main/tinyml_models/Transformer_Time/TransformerModel.py
with the exact hyperparameters used in their evaluate notebook:
    num_classes=2, n_embd=16, n_head=1, block_size=16, hidden_size=32, n_layers=1.

Forward pass:
    Conv1D(1->16, k=3, valid) -> ReLU -> MaxPool(2,2) -> Dropout(0.25)
    -> GlobalAvgPool (length -> 1)              # one token of dim 16
    -> [LN -> SA(no bias on q/k/v, bias on proj) + residual
        -> LN -> FFN(16->32->16) + residual]    # n_layers=1
    -> LN_f -> Dense(num_classes).

Parameter count: 2,306 (the paper's Table II claim of "1.6K parameters" is
inconsistent with the actual repo code; we match the published code instead).
"""

from __future__ import annotations

import tensorflow as tf
from keras import Model, layers

from building.models._common import TARGET_AUDIO_LEN, compile_model

CONV_FILTERS = 16
CONV_KERNEL = 3
POOL_SIZE = 2
POOL_STRIDES = 2
N_EMBD = 16
HIDDEN_SIZE = 32
N_LAYERS = 1
DROPOUT = 0.25


def _scaled_dot_product_attention(q, k, v):
    # Inputs: (B, T, D). Matches torch.nn.functional.scaled_dot_product_attention.
    scale = tf.cast(tf.shape(q)[-1], q.dtype) ** -0.5
    scores = tf.matmul(q, k, transpose_b=True) * scale
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, v)


def _one_head_attention(x, n_embd: int, name: str):
    # q/k/v have no bias in the reference PyTorch code; output proj has bias.
    q = layers.Dense(n_embd, use_bias=False, name=f"{name}_q")(x)
    k = layers.Dense(n_embd, use_bias=False, name=f"{name}_k")(x)
    v = layers.Dense(n_embd, use_bias=False, name=f"{name}_v")(x)
    attn = layers.Lambda(
        lambda qkv: _scaled_dot_product_attention(*qkv),
        name=f"{name}_sdpa",
    )([q, k, v])
    return layers.Dense(n_embd, name=f"{name}_proj")(attn)


def _transformer_block(x, n_embd: int, hidden_size: int, name: str):
    h = layers.LayerNormalization(name=f"{name}_ln1")(x)
    attn = _one_head_attention(h, n_embd, name=f"{name}_sa")
    x = layers.Add(name=f"{name}_res1")([x, attn])

    h = layers.LayerNormalization(name=f"{name}_ln2")(x)
    h = layers.Dense(hidden_size, activation="relu", name=f"{name}_ff1")(h)
    h = layers.Dense(n_embd, name=f"{name}_ff2")(h)
    return layers.Add(name=f"{name}_res2")([x, h])


def build(
    n_classes: int,
    input_len: int = TARGET_AUDIO_LEN,
    conv_filters: int = CONV_FILTERS,
    conv_kernel: int = CONV_KERNEL,
    pool_size: int = POOL_SIZE,
    pool_strides: int = POOL_STRIDES,
    n_embd: int = N_EMBD,
    hidden_size: int = HIDDEN_SIZE,
    n_layers: int = N_LAYERS,
    dropout: float = DROPOUT,
) -> Model:
    inp = layers.Input(shape=(input_len, 1), name="audio")

    x = layers.Conv1D(
        conv_filters, conv_kernel, padding="valid", activation="relu", name="conv1",
    )(inp)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=pool_strides, name="pool1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout1")(x)

    # Adaptive average pool to a single token of dimension `conv_filters`.
    x = layers.GlobalAveragePooling1D(name="adpool")(x)
    x = layers.Reshape((1, conv_filters), name="to_token")(x)

    if conv_filters != n_embd:
        x = layers.Dense(n_embd, name="proj_to_embd")(x)

    for i in range(n_layers):
        x = _transformer_block(x, n_embd, hidden_size, name=f"block{i}")

    x = layers.LayerNormalization(name="ln_f")(x)
    x = layers.Reshape((n_embd,), name="from_token")(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="predictions")(x)
    return compile_model(Model(inp, out, name="transformer"))
