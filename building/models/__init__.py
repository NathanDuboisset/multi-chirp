"""Model registry: name to (builder, input_repr)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple

if TYPE_CHECKING:
    from keras import Model

from building.models import cnn1d, leaf, mel_cnn, sincnet
from building.models._common import (
    MEL_INPUT_SHAPE,
    NUM_MEL_BINS,
    SAMPLE_RATE,
    TARGET_AUDIO_LEN,
    TARGET_FRAMES,
)

InputRepr = Literal["time", "mel"]


class _Entry(NamedTuple):
    build: Callable[..., "Model"]
    input_repr: InputRepr


REGISTRY: dict[str, _Entry] = {
    "cnn1d": _Entry(cnn1d.build, "time"),
    "sincnet": _Entry(sincnet.build, "time"),
    "leaf": _Entry(leaf.build, "time"),
    "mel_cnn": _Entry(mel_cnn.build, "mel"),
    "mel_cnn_2": _Entry(mel_cnn.build_2, "mel"),
}


def available_models() -> list[str]:
    return list(REGISTRY.keys())


def build_model(name: str, n_classes: int, **kwargs: Any) -> "Model":
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name].build(n_classes, **kwargs)


def input_repr_for(name: str) -> InputRepr:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name].input_repr


__all__ = [
    "MEL_INPUT_SHAPE",
    "NUM_MEL_BINS",
    "SAMPLE_RATE",
    "TARGET_AUDIO_LEN",
    "TARGET_FRAMES",
    "REGISTRY",
    "available_models",
    "build_model",
    "input_repr_for",
]
