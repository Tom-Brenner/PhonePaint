"""MAPS acoustic feature extraction (MFCC + delta + delta-delta)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import python_speech_features as psf
from scipy.io import wavfile

FRAME_INTERVAL = 0.01  # 10 ms hop, matches original MAPS frontend


def read_mono_wav(wav_path: Path | str) -> tuple[int, np.ndarray]:
    """Read WAV; return sample rate and mono samples (left channel if stereo)."""
    sr, samples = wavfile.read(wav_path)
    if samples.ndim == 2:
        samples = samples[:, 0]
    return sr, samples


def extract_features(samples: np.ndarray, sr: int) -> np.ndarray:
    """Return ``[frames, 39]`` float32 feature matrix."""
    mfcc = psf.mfcc(samples, sr, winstep=FRAME_INTERVAL)
    delta = psf.delta(mfcc, 2)
    deltadelta = psf.delta(delta, 2)
    x = np.hstack((mfcc, delta, deltadelta))
    return x.astype(np.float32, copy=False)


def extract_features_batch(samples: np.ndarray, sr: int) -> np.ndarray:
    """Return ``[1, frames, 39]`` for model input."""
    feats = extract_features(samples, sr)
    return feats[None, ...]
