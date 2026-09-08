"""In-process MAPS forced alignment (no TensorFlow, no subprocess sidecars)."""

from __future__ import annotations

import itertools
import re
import statistics
import tempfile
import warnings
from pathlib import Path

import natsort
import numpy as np
import soxr
import torch
from textgrid import Interval, IntervalTier, TextGrid

from maps_torch.alignment import force_align, load_dictionary
from maps_torch.cli_types import WordString
from maps_torch.features import extract_features_batch, read_mono_wav
from maps_torch.model import load_checkpoint
from maps_torch.textgrid_io import make_textgrid

TARGET_SR = 16_000
FRAME_INTERVAL = 0.01
EPS = 1e-8


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unknown device {name!r}; choose auto, cpu, or cuda.")


def discover_models(model_path: Path) -> list[Path]:
    if model_path.suffix == ".tf":
        raise RuntimeError(
            f"TensorFlow models ({model_path}) are not supported at runtime. "
            "Use converted .pt checkpoints under tools/maps/torch_models/."
        )
    if model_path.suffix == ".pt" and model_path.is_file():
        return [model_path]
    if model_path.is_dir():
        models = natsort.natsorted(p for p in model_path.iterdir() if p.suffix == ".pt")
        if not models:
            raise RuntimeError(f"No .pt checkpoints found in {model_path}")
        return models
    raise RuntimeError(f"Model not found: {model_path}")


def whisper_to_maps_words(text: str) -> list[str]:
    """Normalize Whisper transcripts for CMUdict lookup (uppercase, no punctuation)."""
    cleaned = re.sub(r"[^\w']+", " ", text.upper())
    return [word for word in cleaned.split() if word]


def load_16k_samples(wav_path: Path) -> tuple[np.ndarray, float]:
    """Return mono int16-like samples at 16 kHz and duration in seconds."""
    sr, samples = read_mono_wav(wav_path)
    if samples.ndim == 2:
        samples = samples[:, 0]
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    if sr != TARGET_SR:
        resampled = soxr.resample(samples.astype(np.float32), sr, TARGET_SR, quality="HQ")
        samples = np.clip(resampled, -32768, 32767).astype(np.int16)
        sr = TARGET_SR
    duration = samples.size / float(sr)
    return samples, duration


def _align_one_model(
    samples: np.ndarray,
    sr: int,
    word_labels: list[str],
    word2phone: dict,
    model: torch.nn.Module,
    device: torch.device,
    *,
    add_sil: bool,
    use_interp: bool,
    check_variants: bool,
) -> TextGrid:
    duration = samples.size / float(sr)
    x = extract_features_batch(samples, sr)
    with torch.inference_mode():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        yhat = model(tensor).cpu().numpy()

    labels = list(word_labels)
    if add_sil and duration >= 0.045:
        labels = ["sil"] + labels + ["sil"]
    elif add_sil:
        warnings.warn(
            f"Silence segments not added because duration {duration:.3f}s is too short."
        )

    word_chain = [word2phone[w] for w in labels]
    best_score = np.inf
    best_w_string: WordString | None = None
    best_seq = None
    best_m = None

    for c in itertools.product(*word_chain):
        this_word_labels = labels
        pron = list(c)
        if add_sil:
            this_word_labels = [x for cI, x in zip(c, labels) if cI]
            pron = [x for x in c if x]

        w_string = WordString(this_word_labels, pron)
        if add_sil and len(w_string.collapsed_string) > (duration - 0.015) / 0.01:
            if this_word_labels and this_word_labels[0] == "sil":
                this_word_labels = this_word_labels[1:]
                pron = pron[1:]
            if this_word_labels and this_word_labels[-1] == "sil":
                this_word_labels = this_word_labels[:-1]
                pron = pron[:-1]
            w_string = WordString(this_word_labels, pron)

        if best_w_string is None:
            best_w_string = w_string

        seq, m = force_align(w_string.collapsed_string, yhat)
        if m[-1, -1] < best_score:
            best_seq = seq
            best_m = m
            best_score = m[-1, -1]
            best_w_string = w_string

        if not check_variants:
            break

    assert best_w_string is not None and best_seq is not None and best_m is not None

    with tempfile.NamedTemporaryFile(suffix=".TextGrid", delete=False) as tmp:
        tg_path = Path(tmp.name)

    try:
        make_textgrid(
            best_seq,
            tg_path,
            duration,
            best_w_string,
            interpolate=use_interp,
            probs=best_m.T,
        )
        tg = TextGrid()
        tg.read(str(tg_path), round_digits=1000)
        return tg
    finally:
        tg_path.unlink(missing_ok=True)


def _median_ensemble(tgs: list[TextGrid]) -> TextGrid:
    n_tgs = len(tgs)
    n_intervals = len(tgs[0].tiers[1].intervals)
    intervals: list = []
    cis: list = []

    for i in range(n_intervals):
        lab = tgs[0].tiers[1].intervals[i].mark
        mintimes = [tg.tiers[1].intervals[i].minTime for tg in tgs]
        maxtimes = [tg.tiers[1].intervals[i].maxTime for tg in tgs]
        mintime = statistics.median(mintimes)
        maxtime = statistics.median(maxtimes)
        times_sorted = sorted(maxtimes)
        ci_lo = times_sorted[1]
        ci_hi = times_sorted[8]
        if ci_lo == ci_hi:
            ci_lo -= EPS
            ci_hi += EPS
        intervals.append(Interval(minTime=mintime, maxTime=maxtime, mark=lab))

    n_word_intervals = len(tgs[0].tiers[0].intervals)
    word_intervals = []
    for i in range(n_word_intervals):
        lab = tgs[0].tiers[0].intervals[i].mark
        mintimes = [tg.tiers[0].intervals[i].minTime for tg in tgs]
        maxtimes = [tg.tiers[0].intervals[i].maxTime for tg in tgs]
        word_intervals.append(
            Interval(
                minTime=statistics.median(mintimes),
                maxTime=statistics.median(maxtimes),
                mark=lab,
            )
        )

    ens = TextGrid(maxTime=tgs[0].maxTime)
    word_tier = IntervalTier(name="words")
    word_tier.intervals = word_intervals
    ens.tiers.append(word_tier)
    phone_tier = IntervalTier(name="phones")
    phone_tier.intervals = intervals
    ens.tiers.append(phone_tier)
    return ens


def align_wav(
    wav_path: Path,
    text: str,
    *,
    models: list[torch.nn.Module],
    word2phone: dict,
    device: torch.device,
    add_sil: bool = True,
    use_interp: bool = True,
    check_variants: bool = False,
) -> TextGrid:
    words = whisper_to_maps_words(text)
    if not words:
        raise ValueError(f"Empty transcript for {wav_path.name}")
    missing = [w for w in words if w not in word2phone]
    if missing:
        raise RuntimeError(
            f"Dictionary missing word(s) for {wav_path.name}: {', '.join(sorted(missing))}"
        )

    samples, _duration = load_16k_samples(wav_path)
    per_model = [
        _align_one_model(
            samples,
            TARGET_SR,
            words,
            word2phone,
            model,
            device,
            add_sil=add_sil,
            use_interp=use_interp,
            check_variants=check_variants,
        )
        for model in models
    ]
    if len(per_model) == 1:
        return per_model[0]
    return _median_ensemble(per_model)


def load_models(model_path: Path, device: torch.device) -> list[torch.nn.Module]:
    return [load_checkpoint(path, device=device) for path in discover_models(model_path)]
