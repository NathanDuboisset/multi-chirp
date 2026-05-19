"""Public surface of building.utils — re-exports from the focused submodules.

Existing call sites (e.g. `from building.utils import SAMPLE_RATE,
fix_audio_length_mel, create_log_mel_spectrogram, TARGET_FRAMES_MEL,
NUM_MEL_BINS_MEL`) continue to work because every public name in the
submodules is re-exported here.
"""

from __future__ import annotations

from .audio_io import (
    fix_audio_length_mel,
    fix_audio_length_time,
    make_audio_datasets,
    make_time_datasets,
    time_to_features,
)
from .constants import (
    CLIP_DURATION_SEC,
    FFT_LENGTH_MEL,
    FRAME_LENGTH,
    FRAME_STEP,
    LOWER_EDGE_HERTZ,
    NUM_MEL_BINS_MEL,
    SAMPLE_RATE,
    SEED,
    TARGET_AUDIO_LEN,
    TARGET_AUDIO_LEN_MEL,
    TARGET_AUDIO_LEN_TIME,
    TARGET_FRAMES_MEL,
    TARGET_FRAMES_TIME,
    UPPER_EDGE_HERTZ,
)
from .export import (
    build_representative_batches,
    export_int8_tflite_from_saved_model,
    export_keras_model_to_int8_tflite,
    representative_dataset_from_batches,
)
from .mel_spec import (
    RUST_MEL_MATRIX,
    build_rust_mel_matrix,
    create_log_mel_spectrogram,
    hz_to_mel,
    make_mel_datasets,
    mel_to_hz,
)
from .paths import DATASET_ROOT, MODELS_DIR, REPO_ROOT, SRC_DIR, ModelPaths, get_paths
from .rust_clips import TestClip, collect_test_clips_for_rs, write_audio_sample_rs
from .tf_config import configure_tf_runtime, set_global_seed

__all__ = [
    # constants
    "SEED",
    "SAMPLE_RATE",
    "CLIP_DURATION_SEC",
    "FRAME_LENGTH",
    "FRAME_STEP",
    "TARGET_AUDIO_LEN",
    "TARGET_AUDIO_LEN_TIME",
    "TARGET_AUDIO_LEN_MEL",
    "FFT_LENGTH_MEL",
    "NUM_MEL_BINS_MEL",
    "LOWER_EDGE_HERTZ",
    "UPPER_EDGE_HERTZ",
    "TARGET_FRAMES_MEL",
    "TARGET_FRAMES_TIME",
    # paths
    "REPO_ROOT",
    "MODELS_DIR",
    "SRC_DIR",
    "DATASET_ROOT",
    "ModelPaths",
    "get_paths",
    # tf_config
    "set_global_seed",
    "configure_tf_runtime",
    # audio_io
    "make_audio_datasets",
    "fix_audio_length_time",
    "fix_audio_length_mel",
    "time_to_features",
    "make_time_datasets",
    # mel_spec
    "hz_to_mel",
    "mel_to_hz",
    "build_rust_mel_matrix",
    "RUST_MEL_MATRIX",
    "create_log_mel_spectrogram",
    "make_mel_datasets",
    # export
    "build_representative_batches",
    "representative_dataset_from_batches",
    "export_int8_tflite_from_saved_model",
    "export_keras_model_to_int8_tflite",
    # rust_clips
    "TestClip",
    "collect_test_clips_for_rs",
    "write_audio_sample_rs",
]
