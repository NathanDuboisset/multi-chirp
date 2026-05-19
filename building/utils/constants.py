"""Audio geometry and training constants. No external deps beyond numbers."""

from __future__ import annotations

SEED = 3407

SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 3.0
FRAME_LENGTH = 1024
FRAME_STEP = 256

# Full clip = 3 s at 16 kHz = 48000 samples. Source of truth for every pipeline.
TARGET_AUDIO_LEN = int(SAMPLE_RATE * CLIP_DURATION_SEC)
TARGET_AUDIO_LEN_TIME = TARGET_AUDIO_LEN
TARGET_AUDIO_LEN_MEL = TARGET_AUDIO_LEN

# Mel CNN parameters.
FFT_LENGTH_MEL = FRAME_LENGTH
NUM_MEL_BINS_MEL = 80
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 8000.0
# Number of STFT frames produced on TARGET_AUDIO_LEN samples.
TARGET_FRAMES_MEL = 1 + (TARGET_AUDIO_LEN - FRAME_LENGTH) // FRAME_STEP
TARGET_FRAMES_TIME = TARGET_FRAMES_MEL
