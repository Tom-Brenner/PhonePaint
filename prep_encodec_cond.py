#!/usr/bin/env python3
"""
Pre-compute VoiceCraft 16 kHz EnCodec latents and per-phone frame masks for the
phone-conditioned training pipeline (train_phone_ec_cond.py).

Data sources
------------
* LibriSpeech audio:  /media/tom/SATAM/King/LibriSpeech/{subset}/...
* LibriSpeech phones: /media/tom/SATAM/librispeech-phone-alignments/{subset}/...
  (per-utterance Praat TextGrids with an ARPAbet ``phones`` tier)
* VCTK (downsampled to 16 kHz on load from the existing 24 kHz ``*_safe`` wavs):
  audio  /media/tom/SATAM/lip/vctk_audio/{spk}_safe/*.wav
  phones /media/tom/SATAM/lip/vctk_audio/aligned_json/{spk}_safe_aligned.json

Each NPZ stores continuous encoder latents ``z_e`` [T, D] float16 plus one
bool[T] mask per requested phone (``{phone}_mask``).  Frame rate is 50 fps
(16 kHz audio, VoiceCraft EnCodec hop 320).

Examples
--------
    python prep_encodec_cond.py --phone_masks s r
    python prep_encodec_cond.py --phone_masks s r \\
        --librispeech_subsets train-clean-100 train-clean-360
    python prep_encodec_cond.py --phone_masks s r --skip_vctk
    python prep_encodec_cond.py --phone_masks s r --vctk_only
    python prep_encodec_cond.py --phone_masks s r --refresh_utterances
    # Add new phones later without re-encoding existing NPZs:
    python prep_encodec_cond.py --phone_masks z k --add_phones
    # Reuse z_e / copy NPZs from a previous prep (any source):
    python prep_encodec_cond.py --phone_masks s r --copy_npz_from /path/to/old_prep
    python prep_encodec_cond.py --phone_masks s r --skip_vctk \\
        --copy_npz_from /path/to/old_prep --output_dir /path/to/new_prep

Utterance lists are cached in ``<output_dir>/utterances_cache.json`` so reruns skip
the slow alignment scan.  ``manifest.jsonl`` is rewritten at the end of each run.

With ``--add_phones``, pass only the new ``--phone_masks``; utterances that already
have ``z_e`` get the new mask arrays merged in and other masks are left untouched.
Without ``--add_phones``, each NPZ is re-encoded and stores only the requested masks.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

from phone_labels import (
    DEFAULT_TARGET_PHONES,
    align_label,
    build_alignment_char_vocab,
    build_label_to_phone,
    encode_label_to_ids,
    mask_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
LIBRISPEECH_AUDIO_ROOT = Path("/media/tom/SATAM/King/LibriSpeech")
LIBRISPEECH_ALIGN_ROOT = Path("/media/tom/SATAM/librispeech-phone-alignments")
VCTK_AUDIO_BASE = Path("/media/tom/SATAM/lip/vctk_audio")
VCTK_ALIGNED_DIR = VCTK_AUDIO_BASE / "aligned_json"
DEFAULT_OUTPUT_DIR = Path("/media/tom/SATAM/lip/encodec_npzs_cond")

DEFAULT_LIBRISPEECH_SUBSETS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

UTTERANCES_CACHE_VERSION = 1
DEFAULT_GPU_CLEAR_EVERY = 50

# ─────────────────────────────────────────────────────────────────────────────
# Audio / codec config (VoiceCraft 16 kHz EnCodec)
# ─────────────────────────────────────────────────────────────────────────────
FS = 16_000
ENCODEC_FPS = 50.0
ENCODEC_DIM = 128
MAX_UTT_SEC = 2000.0
MIN_CONTEXT_SEC = 0.15
MIN_PHONE_DUR_SEC = 0.03

SKIP_PHONE_LABELS = frozenset({"", "sil", "spn", "<unk>"})

Span = Tuple[float, float]
PhoneSpans = Dict[str, List[Span]]

Utterance = namedtuple(
    "Utterance",
    [
        "wav_path", "phone_spans", "pad_pre_sec", "pad_post_sec",
        "speaker", "source", "phone_intervals",
    ],
    defaults=((),),
)

_PHONE_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*\n"
    r"\s*xmin = ([\d.]+)\s*\n"
    r"\s*xmax = ([\d.]+)\s*\n"
    r'\s*text = "([^"]*)"',
    re.MULTILINE,
)


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# TextGrid parsing (phones tier, ARPAbet)
# ─────────────────────────────────────────────────────────────────────────────

def parse_textgrid_phone_intervals(tg_path: Path) -> List[Tuple[float, float, str]]:
    """Return (xmin, xmax, label) intervals from the ``phones`` tier."""
    try:
        content = tg_path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot read TextGrid %s: %s", tg_path, exc)
        return []

    marker = 'name = "phones"'
    start = content.find(marker)
    if start < 0:
        return []

    chunk = content[start:]
    next_item = re.search(r"\nitem \[\d+\]:\s*\n\s*class", chunk[len(marker):])
    if next_item is not None:
        chunk = chunk[: len(marker) + next_item.start()]

    return [
        (float(m.group(1)), float(m.group(2)), m.group(3).strip())
        for m in _PHONE_INTERVAL_RE.finditer(chunk)
    ]


def label_to_user_phone(label: str, label_to_phone: Dict[str, str]) -> Optional[str]:
    if label in SKIP_PHONE_LABELS:
        return None
    if label in label_to_phone:
        return label_to_phone[label]
    # ARPAbet vowel stress digits, e.g. S0 -> S (already in map) or AH0 -> AH
    base = re.sub(r"\d+$", "", label)
    return label_to_phone.get(base)


def spans_from_phone_intervals(
    intervals: Sequence[Tuple[float, float, str]],
    phones: Sequence[str],
    label_to_phone: Dict[str, str],
    min_phone_dur: float,
) -> PhoneSpans:
    phone_spans: PhoneSpans = {ph: [] for ph in phones}
    for xmin, xmax, label in intervals:
        user_phone = label_to_user_phone(label, label_to_phone)
        if user_phone is None:
            continue
        if xmax - xmin < min_phone_dur:
            continue
        phone_spans[user_phone].append((xmin, xmax))
    return phone_spans


def phone_intervals_from_mfa(
    phone_dict: dict,
    min_phone_dur: float = MIN_PHONE_DUR_SEC,
) -> List[Tuple[float, float, str]]:
    """All non-skip MFA phone intervals in time order."""
    items: List[Tuple[float, float, str]] = []
    for ph_entry in phone_dict.values():
        label = str(ph_entry["text"]).strip()
        if label in SKIP_PHONE_LABELS:
            continue
        xmin, xmax = float(ph_entry["xmin"]), float(ph_entry["xmax"])
        if xmax - xmin < min_phone_dur:
            continue
        items.append((xmin, xmax, label))
    return sorted(items, key=lambda x: x[0])


def spans_from_mfa_json_phones(
    phone_dict: dict,
    phones: Sequence[str],
    label_to_phone: Dict[str, str],
    min_phone_dur: float,
) -> PhoneSpans:
    phone_spans: PhoneSpans = {ph: [] for ph in phones}
    for ph_entry in phone_dict.values():
        label = ph_entry["text"]
        user_phone = label_to_user_phone(label, label_to_phone)
        if user_phone is None:
            continue
        xmin, xmax = float(ph_entry["xmin"]), float(ph_entry["xmax"])
        if xmax - xmin < min_phone_dur:
            continue
        phone_spans[user_phone].append((xmin, xmax))
    return phone_spans


# ─────────────────────────────────────────────────────────────────────────────
# Utterance collection
# ─────────────────────────────────────────────────────────────────────────────

def _finalize_utterance(
    wav_path: Path,
    phone_spans_raw: PhoneSpans,
    phones: Sequence[str],
    speaker: str,
    source: str,
    min_context: float = MIN_CONTEXT_SEC,
    max_utt_dur: float = MAX_UTT_SEC,
    phone_intervals_raw: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> Optional[Utterance]:
    if not any(phone_spans_raw[ph] for ph in phones):
        return None

    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        log.warning("sf.info failed for %s: %s", wav_path, exc)
        return None

    utt_dur = info.frames / info.samplerate
    utt_end = min(utt_dur, max_utt_dur)

    clipped: PhoneSpans = {}
    for ph in phones:
        spans = [(s, e) for s, e in phone_spans_raw[ph] if s < utt_end]
        if spans:
            clipped[ph] = spans
    if not clipped:
        return None

    all_spans = [sp for spans in clipped.values() for sp in spans]
    first_start = min(s for s, _ in all_spans)
    last_end = max(e for _, e in all_spans)
    pad_pre = max(0.0, min_context - first_start)
    pad_post = max(0.0, min_context - (utt_end - last_end))

    phone_spans_shifted = {
        ph: [(s + pad_pre, e + pad_pre) for s, e in spans]
        for ph, spans in clipped.items()
    }

    shifted_intervals: List[Tuple[float, float, str]] = []
    if phone_intervals_raw:
        for xmin, xmax, label in phone_intervals_raw:
            if label in SKIP_PHONE_LABELS:
                continue
            if xmin >= utt_end:
                continue
            xmax = min(xmax, utt_end)
            if xmax - xmin < MIN_PHONE_DUR_SEC:
                continue
            shifted_intervals.append((xmin + pad_pre, xmax + pad_pre, label))

    return Utterance(
        wav_path=wav_path,
        phone_spans=phone_spans_shifted,
        pad_pre_sec=pad_pre,
        pad_post_sec=pad_post,
        speaker=speaker,
        source=source,
        phone_intervals=tuple(shifted_intervals),
    )


def collect_utterances_librispeech(
    audio_root: Path,
    align_root: Path,
    subsets: Sequence[str],
    phones: Sequence[str],
    min_phone_dur: float = MIN_PHONE_DUR_SEC,
    min_context: float = MIN_CONTEXT_SEC,
    max_utt_dur: float = MAX_UTT_SEC,
) -> List[Utterance]:
    utterances: List[Utterance] = []
    label_to_phone = build_label_to_phone(phones)

    for subset in subsets:
        align_subset = align_root / subset
        audio_subset = audio_root / subset
        if not align_subset.is_dir():
            log.warning("Alignment subset missing: %s", align_subset)
            continue
        if not audio_subset.is_dir():
            log.warning("Audio subset missing: %s", audio_subset)
            continue

        n_subset = 0
        for tg_path in sorted(align_subset.rglob("*.TextGrid")):
            rel = tg_path.relative_to(align_subset)
            wav_path = audio_subset / rel.with_suffix(".flac")
            if not wav_path.is_file():
                continue

            parts = rel.parts
            speaker = (
                f"librispeech/{subset}/{'/'.join(parts[:-1])}"
                if len(parts) >= 2
                else f"librispeech/{subset}"
            )

            intervals = parse_textgrid_phone_intervals(tg_path)
            phone_spans_raw = spans_from_phone_intervals(
                intervals, phones, label_to_phone, min_phone_dur,
            )
            utt = _finalize_utterance(
                wav_path,
                phone_spans_raw,
                phones,
                speaker=speaker,
                source=f"librispeech/{subset}",
                min_context=min_context,
                max_utt_dur=max_utt_dur,
                phone_intervals_raw=intervals,
            )
            if utt is not None:
                utterances.append(utt)
                n_subset += 1

        log.info(
            "LibriSpeech %s: %d utterances with any of %s",
            subset, n_subset, list(phones),
        )

    log.info("Collected %d LibriSpeech utterances total", len(utterances))
    return utterances


def _subset_tag(subset: str) -> str:
    """
    Map LibriSpeech subset names to the numeric tag used in aligned-* dirs.
    Examples:
      train-clean-100 -> 100
      train-clean-360 -> 360
      train-other-500 -> 500
    """
    m = re.search(r"(\d+)$", subset)
    return m.group(1) if m else subset


def _iter_librispeech_mfa_jsons(align_root: Path, subset: str) -> Iterable[Path]:
    """
    Yield LibriSpeech MFA-style alignment JSONs.

    Expected layout (per user):
      /media/tom/SATAM/librispeech-phone-alignments/aligned-100/...
      /media/tom/SATAM/librispeech-phone-alignments/aligned-360/...
      /media/tom/SATAM/librispeech-phone-alignments/aligned-500/...
    where JSON structure is similar to VCTK aligned_json (top-level dict of utterances,
    each with a "phones" dict of {id: {xmin,xmax,text}} entries).
    """
    tag = _subset_tag(subset)

    # Primary: aligned-<tag> directory (your actual layout).
    direct_dash = align_root / f"aligned-{tag}"
    if direct_dash.is_dir():
        yield from sorted(direct_dash.rglob("*.json"))
        return

    # Backward-compatible fallback: aligned_<subset> (old guess).
    direct_us = align_root / f"aligned_{subset}"
    if direct_us.is_dir():
        yield from sorted(direct_us.rglob("*.json"))
        return

    # Last resort: scan any aligned-* / aligned_* dirs that mention the tag/subset.
    for d in sorted(p for p in align_root.glob("aligned-*") if p.is_dir()):
        if tag not in d.name and subset not in d.name:
            continue
        yield from sorted(d.rglob("*.json"))


def _resolve_librispeech_mfa_audio_path(audio_subset: Path, wav_key: str) -> Optional[Path]:
    """
    Resolve a LibriSpeech MFA JSON utterance key to an audio file.

    JSON keys look like '16-122828-0057.flac' (flat filename, .flac suffix).
    Audio is stored as:
        {audio_subset}/{speaker}/{chapter}/{speaker}-{chapter}-{utterance}.wav
    e.g. wav-360/16/122828/16-122828-0057.wav

    We parse the stem on the first two '-' delimiters to recover speaker and chapter,
    then try both .wav and .flac in the nested directory.  If parsing fails we fall back
    to trying the key flat (both suffixes) directly under audio_subset.
    """
    stem = Path(wav_key).stem  # e.g. '16-122828-0057'
    parts = stem.split("-", 2)  # ['16', '122828', '0057']
    if len(parts) == 3:
        speaker, chapter = parts[0], parts[1]
        nested = audio_subset / speaker / chapter / stem
        for suffix in (".wav", ".flac"):
            p = nested.with_suffix(suffix)
            if p.is_file():
                return p

    # Fallback: flat key under audio_subset (both suffixes).
    flat = audio_subset / stem
    for suffix in (".wav", ".flac"):
        p = flat.with_suffix(suffix)
        if p.is_file():
            return p

    return None


def collect_utterances_librispeech_mfa(
    audio_root: Path,
    align_root: Path,
    subsets: Sequence[str],
    phones: Sequence[str],
    min_phone_dur: float = MIN_PHONE_DUR_SEC,
    min_context: float = MIN_CONTEXT_SEC,
    max_utt_dur: float = MAX_UTT_SEC,
    num_workers: int = 1,
) -> List[Utterance]:
    """
    LibriSpeech utterances using MFA JSON alignments (instead of ARPAbet TextGrids).
    """
    utterances: List[Utterance] = []
    label_to_phone = build_label_to_phone(phones)

    for subset in subsets:
        tag = _subset_tag(subset)
        # Audio lives under wav-<tag>, not the Libri subset name.
        audio_subset = audio_root / f"wav-{tag}"
        if not audio_subset.is_dir():
            log.warning(
                "Audio subset dir missing for %s: tried %s", subset, audio_subset
            )
            continue

        jsons = list(_iter_librispeech_mfa_jsons(align_root, subset))
        if not jsons:
            log.warning("No Libri MFA JSONs found for %s under %s", subset, align_root)
            continue

        n_subset = 0
        n_missing_audio = 0

        def _scan_one_json(jf: Path) -> tuple[int, int, List[Utterance]]:
            """Return (n_ok, n_missing_audio, utterances_from_file)."""
            try:
                with open(jf, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Unreadable Libri MFA JSON %s: %s", jf, exc)
                return 0, 0, []

            if not isinstance(data, dict):
                return 0, 0, []

            out: List[Utterance] = []
            ok = missing = 0
            for wav_key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                wav_path = _resolve_librispeech_mfa_audio_path(audio_subset, str(wav_key))
                if wav_path is None or not wav_path.is_file():
                    missing += 1
                    continue

                # Your JSON keys end with .flac, so speaker dirs like aligned-360/16/*.json
                # naturally group by the parent directory of the key (if any). If the key is
                # just "16-....flac", we group under the JSON's parent folder name.
                rel = Path(str(wav_key))
                if len(rel.parts) >= 2:
                    speaker_bucket = rel.parent.as_posix()
                else:
                    speaker_bucket = jf.parent.name
                speaker = f"librispeech/{subset}/{speaker_bucket}"

                phone_spans_raw = spans_from_mfa_json_phones(
                    entry.get("phones", {}), phones, label_to_phone, min_phone_dur,
                )
                intervals = phone_intervals_from_mfa(entry.get("phones", {}), min_phone_dur)
                utt = _finalize_utterance(
                    wav_path,
                    phone_spans_raw,
                    phones,
                    speaker=speaker,
                    source=f"librispeech/{subset}",
                    min_context=min_context,
                    max_utt_dur=max_utt_dur,
                    phone_intervals_raw=intervals,
                )
                if utt is not None:
                    out.append(utt)
                    ok += 1
            return ok, missing, out

        workers = max(1, int(num_workers or 1))
        if workers == 1:
            for jf in jsons:
                ok, miss, out = _scan_one_json(jf)
                n_subset += ok
                n_missing_audio += miss
                utterances.extend(out)
        else:
            # Threaded JSON parsing + metadata reading can speed up on multicore.
            # This does not parallelize EnCodec encoding (still done later, sequentially).
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_scan_one_json, jf) for jf in jsons]
                for fut in as_completed(futs):
                    ok, miss, out = fut.result()
                    n_subset += ok
                    n_missing_audio += miss
                    utterances.extend(out)

        log.info(
            "LibriSpeech %s (MFA JSON): %d utterances with any of %s (missing audio=%d)",
            subset, n_subset, list(phones), n_missing_audio,
        )

    log.info("Collected %d LibriSpeech utterances total (MFA JSON)", len(utterances))
    return utterances


def collect_utterances_vctk(
    audio_base: Path,
    aligned_dir: Path,
    phones: Sequence[str],
    min_phone_dur: float = MIN_PHONE_DUR_SEC,
    min_context: float = MIN_CONTEXT_SEC,
    max_utt_dur: float = MAX_UTT_SEC,
) -> List[Utterance]:
    """VCTK utterances; audio is resampled 24 kHz -> 16 kHz during encoding prep."""
    utterances: List[Utterance] = []
    label_to_phone = build_label_to_phone(phones)

    for jf in sorted(aligned_dir.glob("*_aligned.json")):
        spk_dir = audio_base / jf.stem.replace("_aligned", "")
        if not spk_dir.is_dir():
            continue
        speaker = spk_dir.name
        with open(jf, encoding="utf-8") as fh:
            data = json.load(fh)

        for wav_name, entry in data.items():
            wav_path = spk_dir / wav_name
            if not wav_path.is_file():
                continue
            phone_spans_raw = spans_from_mfa_json_phones(
                entry.get("phones", {}), phones, label_to_phone, min_phone_dur,
            )
            intervals = phone_intervals_from_mfa(entry.get("phones", {}), min_phone_dur)
            utt = _finalize_utterance(
                wav_path,
                phone_spans_raw,
                phones,
                speaker=speaker,
                source="vctk",
                min_context=min_context,
                max_utt_dur=max_utt_dur,
                phone_intervals_raw=intervals,
            )
            if utt is not None:
                utterances.append(utt)

    log.info(
        "Collected %d VCTK utterances containing any of %s (from %s)",
        len(utterances), list(phones), aligned_dir,
    )
    return utterances


def collect_utterances(
    phones: Sequence[str],
    librispeech_subsets: Sequence[str],
    include_librispeech: bool,
    include_vctk: bool,
    audio_root: Path = LIBRISPEECH_AUDIO_ROOT,
    align_root: Path = LIBRISPEECH_ALIGN_ROOT,
    vctk_audio: Path = VCTK_AUDIO_BASE,
    vctk_aligned: Path = VCTK_ALIGNED_DIR,
    libri_mfa: bool = False,
    num_workers: int = 1,
    **kwargs,
) -> List[Utterance]:
    utterances: List[Utterance] = []
    if include_librispeech:
        if libri_mfa:
            utterances.extend(
                collect_utterances_librispeech_mfa(
                    audio_root,
                    align_root,
                    librispeech_subsets,
                    phones,
                    num_workers=num_workers,
                    **kwargs,
                )
            )
        else:
            utterances.extend(
                collect_utterances_librispeech(
                    audio_root, align_root, librispeech_subsets, phones, **kwargs,
                )
            )
    if include_vctk:
        utterances.extend(
            collect_utterances_vctk(vctk_audio, vctk_aligned, phones, **kwargs)
        )
    return utterances


# ─────────────────────────────────────────────────────────────────────────────
# Utterance cache (skip slow TextGrid / JSON scan on reruns)
# ─────────────────────────────────────────────────────────────────────────────

def utterances_cache_path(out_dir: Path) -> Path:
    return out_dir / "utterances_cache.json"


def utterance_collection_config(
    phones: Sequence[str],
    librispeech_subsets: Sequence[str],
    include_librispeech: bool,
    include_vctk: bool,
    audio_root: Path,
    align_root: Path,
    vctk_audio: Path,
    vctk_aligned: Path,
    libri_mfa: bool,
) -> dict:
    return {
        "version": UTTERANCES_CACHE_VERSION,
        "phones": list(phones),
        "librispeech_subsets": list(librispeech_subsets),
        "include_librispeech": include_librispeech,
        "include_vctk": include_vctk,
        "librispeech_audio": str(audio_root),
        "librispeech_align": str(align_root),
        "librispeech_align_format": "mfa_json" if libri_mfa else "textgrid_arpabet",
        "vctk_audio": str(vctk_audio),
        "vctk_aligned": str(vctk_aligned),
        "min_phone_dur_sec": MIN_PHONE_DUR_SEC,
        "min_context_sec": MIN_CONTEXT_SEC,
        "max_utt_sec": MAX_UTT_SEC,
    }


def utterance_to_dict(utt: Utterance) -> dict:
    return {
        "wav_path": str(utt.wav_path),
        "phone_spans": utt.phone_spans,
        "pad_pre_sec": utt.pad_pre_sec,
        "pad_post_sec": utt.pad_post_sec,
        "speaker": utt.speaker,
        "source": utt.source,
        "phone_intervals": [list(item) for item in utt.phone_intervals],
    }


def utterance_from_dict(data: dict) -> Utterance:
    intervals = data.get("phone_intervals") or ()
    return Utterance(
        wav_path=Path(data["wav_path"]),
        phone_spans={ph: [tuple(sp) for sp in spans] for ph, spans in data["phone_spans"].items()},
        pad_pre_sec=float(data["pad_pre_sec"]),
        pad_post_sec=float(data["pad_post_sec"]),
        speaker=data["speaker"],
        source=data["source"],
        phone_intervals=tuple(tuple(x) for x in intervals),
    )


def save_utterances_cache(
    cache_path: Path,
    config: dict,
    utterances: Sequence[Utterance],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **config,
        "n_utterances": len(utterances),
        "utterances": [utterance_to_dict(u) for u in utterances],
    }
    tmp = cache_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(cache_path)
    log.info("Saved utterance cache (%d records) -> %s", len(utterances), cache_path)


def load_utterances_cache(cache_path: Path, expected: dict) -> Optional[List[Utterance]]:
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Unreadable utterance cache %s: %s", cache_path, exc)
        return None

    for key, value in expected.items():
        if payload.get(key) != value:
            log.info(
                "Utterance cache stale (%s: %r != %r) — recollecting",
                key, payload.get(key), value,
            )
            return None

    utterances = [utterance_from_dict(item) for item in payload.get("utterances", [])]
    log.info("Loaded %d utterances from cache %s", len(utterances), cache_path)
    return utterances


def load_or_collect_utterances(
    phones: Sequence[str],
    librispeech_subsets: Sequence[str],
    include_librispeech: bool,
    include_vctk: bool,
    audio_root: Path,
    align_root: Path,
    vctk_audio: Path,
    vctk_aligned: Path,
    libri_mfa: bool,
    num_workers: int,
    cache_path: Path,
    refresh: bool,
    **kwargs,
) -> List[Utterance]:
    config = utterance_collection_config(
        phones, librispeech_subsets, include_librispeech, include_vctk,
        audio_root, align_root, vctk_audio, vctk_aligned, libri_mfa,
    )
    if not refresh:
        cached = load_utterances_cache(cache_path, config)
        if cached is not None:
            return cached

    utterances = collect_utterances(
        phones=phones,
        librispeech_subsets=librispeech_subsets,
        include_librispeech=include_librispeech,
        include_vctk=include_vctk,
        audio_root=audio_root,
        align_root=align_root,
        vctk_audio=vctk_audio,
        vctk_aligned=vctk_aligned,
        libri_mfa=libri_mfa,
        num_workers=num_workers,
        **kwargs,
    )
    if utterances:
        save_utterances_cache(cache_path, config, utterances)
    return utterances


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear_gpu_cache(device: str, *, sync: bool = True) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    gc.collect()
    if sync:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
def load_encodec(device: str, checkpoint: Path):
    from voicecraft_encodec import load_voicecraft_encodec as _load_voicecraft_encodec
    return _load_voicecraft_encodec(device, checkpoint, load_decoder=False)


@torch.no_grad()
def extract_ze(model, wav_np: np.ndarray, device: str) -> np.ndarray:
    """float32 mono @ FS Hz -> z_e [T, D] float16 (pre-quantization encoder output)."""
    pcm = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(device)
    z_e = model.encoder(pcm)
    out = z_e.squeeze(0).t().cpu().half().numpy()
    del pcm, z_e
    return out


def build_phone_mask(spans: List[Span], T: int, fps: float = ENCODEC_FPS) -> np.ndarray:
    mask = np.zeros(T, dtype=bool)
    for t_start, t_end in spans:
        f_start = max(0, int(t_start * fps))
        f_end = min(T, int(np.ceil(t_end * fps)))
        if f_end > f_start:
            mask[f_start:f_end] = True
    return mask


def build_phone_seq_arrays(
    phone_intervals: Sequence[Tuple[float, float, str]],
    T: int,
    char2id: Dict[str, int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Char-tokenize the utterance phone string with per-phone frame alignment."""
    phone_ids: List[int] = []
    phone_starts: List[int] = []
    phone_ends: List[int] = []
    phone_nchars: List[int] = []

    for start, end, label in phone_intervals:
        char_ids = encode_label_to_ids(label, char2id)
        if not char_ids:
            continue
        f0 = max(0, int(start * ENCODEC_FPS))
        f1 = min(T, int(np.ceil(end * ENCODEC_FPS)))
        if f1 <= f0:
            continue
        phone_starts.append(f0)
        phone_ends.append(f1)
        phone_nchars.append(len(char_ids))
        phone_ids.extend(char_ids)

    if not phone_ids:
        return None

    return (
        np.asarray(phone_ids, dtype=np.int16),
        np.asarray(phone_starts, dtype=np.int32),
        np.asarray(phone_ends, dtype=np.int32),
        np.asarray(phone_nchars, dtype=np.int16),
    )


def npz_full_phone_ok(data) -> bool:
    if not {"phone_ids", "phone_starts", "phone_ends", "phone_nchars"} <= set(data.files):
        return False
    n_phones = int(data["phone_starts"].shape[0])
    return (
        n_phones > 0
        and int(data["phone_ends"].shape[0]) == n_phones
        and int(data["phone_nchars"].shape[0]) == n_phones
        and int(data["phone_nchars"].sum()) == int(data["phone_ids"].shape[0])
    )


def attach_phone_seq_arrays(
    arrays: Dict[str, np.ndarray],
    utt: Utterance,
    T: int,
    char2id: Dict[str, int],
) -> bool:
    if not utt.phone_intervals:
        return False
    seq = build_phone_seq_arrays(utt.phone_intervals, T, char2id)
    if seq is None:
        return False
    phone_ids, phone_starts, phone_ends, phone_nchars = seq
    arrays["phone_ids"] = phone_ids
    arrays["phone_starts"] = phone_starts
    arrays["phone_ends"] = phone_ends
    arrays["phone_nchars"] = phone_nchars
    return True


def resample_wav(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        import librosa
        return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        import torchaudio
        t = torch.from_numpy(wav).float().unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_sr, target_sr)
        return t.squeeze(0).numpy()


def load_wav_padded(utt: Utterance, target_sr: int = FS) -> Optional[np.ndarray]:
    try:
        info = sf.info(str(utt.wav_path))
        read_frames = min(info.frames, int(MAX_UTT_SEC * info.samplerate))
        wav, sr = sf.read(
            str(utt.wav_path),
            stop=read_frames,
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        log.warning("Read failed for %s: %s", utt.wav_path, exc)
        return None

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = resample_wav(wav, sr, target_sr)

    if utt.pad_pre_sec > 0:
        n = int(round(utt.pad_pre_sec * target_sr))
        wav = np.concatenate([np.zeros(n, np.float32), wav])
    if utt.pad_post_sec > 0:
        n = int(round(utt.pad_post_sec * target_sr))
        wav = np.concatenate([wav, np.zeros(n, np.float32)])

    return wav.astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# NPZ I/O
# ─────────────────────────────────────────────────────────────────────────────

def phones_needing_mask_update(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
) -> Tuple[bool, List[str]]:
    """Return (need_encode, phones whose masks are missing or stale)."""
    try:
        with np.load(npz_path) as d:
            if "z_e" not in d:
                return True, list(phones)
            stale: List[str] = []
            for ph in phones:
                key = mask_key(ph)
                if key not in d:
                    stale.append(ph)
                    continue
                should_have = bool(utt.phone_spans.get(ph))
                has_frames = bool(d[key].any())
                if should_have != has_frames:
                    stale.append(ph)
            return False, stale
    except Exception:
        return True, list(phones)


def npz_masks_up_to_date(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
    *,
    add_phones: bool,
    full_phone: bool = False,
) -> bool:
    if add_phones:
        need_encode, stale = phones_needing_mask_update(utt, npz_path, phones)
        if need_encode or stale:
            return False
    else:
        try:
            with np.load(npz_path) as d:
                if "z_e" not in d:
                    return False
                for ph in phones:
                    key = mask_key(ph)
                    if key not in d:
                        return False
                    should_have = bool(utt.phone_spans.get(ph))
                    has_frames = bool(d[key].any())
                    if should_have != has_frames:
                        return False
        except Exception:
            return False

    if full_phone:
        try:
            with np.load(npz_path) as d:
                return npz_full_phone_ok(d)
        except Exception:
            return False
    return True


def phones_from_npz(d) -> List[str]:
    suffix = "_mask"
    return sorted(
        key[: -len(suffix)]
        for key in d.files
        if key.endswith(suffix)
    )


def manifest_record_from_npz(
    npz_path: Path,
    wav_path: Path,
    speaker: str,
    source: str,
    *,
    full_phone: bool = False,
) -> dict:
    with np.load(npz_path) as d:
        T = int(d["z_e"].shape[0])
        phones = phones_from_npz(d)
        n_mask_frames = {
            ph: int(d[mask_key(ph)].sum()) for ph in phones
        }
        has_full_phone = npz_full_phone_ok(d) if full_phone else False
    rec = {
        "npz": str(npz_path),
        "wav": str(wav_path),
        "speaker": speaker,
        "source": source,
        "n_frames": T,
        "phone_masks": phones,
        "n_mask_frames": n_mask_frames,
        "sample_rate": FS,
        "encodec_fps": ENCODEC_FPS,
    }
    if full_phone:
        rec["full_phone"] = has_full_phone
    return rec


def load_manifest_records(manifest_path: Path) -> List[dict]:
    if not manifest_path.is_file():
        return []
    records: List[dict] = []
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read manifest %s: %s", manifest_path, exc)
    return records


def merge_manifest_records(
    previous: Sequence[dict],
    updated: Sequence[dict],
) -> List[dict]:
    by_npz = {rec["npz"]: rec for rec in previous}
    for rec in updated:
        by_npz[rec["npz"]] = rec
    return [by_npz[k] for k in sorted(by_npz)]


def utterance_npz_path(utt: Utterance, out_dir: Path) -> Path:
    return out_dir / utt.speaker / (Path(utt.wav_path).stem + ".npz")


def plan_add_phones_manifest(
    previous: Sequence[dict],
    utterances: Sequence[Utterance],
    out_dir: Path,
) -> dict:
    previous_npz = {rec["npz"] for rec in previous}
    run_npz = {str(utterance_npz_path(utt, out_dir)) for utt in utterances}
    return {
        "previous": len(previous_npz),
        "to_process": len(run_npz),
        "already_in_manifest": len(run_npz & previous_npz),
        "new_not_in_manifest": len(run_npz - previous_npz),
        "kept_without_new_phones": len(previous_npz - run_npz),
        "expected_final": len(previous_npz | run_npz),
    }


def write_manifest(manifest_path: Path, records: Sequence[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    tmp.replace(manifest_path)


def utterance_npz_rel(utt: Utterance) -> str:
    """Speaker-relative NPZ path used under output_dir."""
    return str(Path(utt.speaker) / f"{Path(utt.wav_path).stem}.npz")


def resolve_copy_npz_from(args) -> Optional[Path]:
    """Unified previous-prep directory (--copy_npz_from or legacy aliases)."""
    if args.copy_npz_from:
        return Path(args.copy_npz_from)
    legacy_vctk = getattr(args, "copy_vctk_from", None)
    legacy_libri = getattr(args, "copy_libri_npz_from", None)
    if legacy_vctk or legacy_libri:
        if legacy_vctk and legacy_libri and Path(legacy_vctk) != Path(legacy_libri):
            raise SystemExit(
                "--copy_vctk_from and --copy_libri_npz_from point to different dirs; "
                "use --copy_npz_from DIR instead."
            )
        log.warning(
            "Deprecated: --copy_vctk_from / --copy_libri_npz_from — use --copy_npz_from"
        )
        return Path(legacy_vctk or legacy_libri)
    return None


def _npz_rel_in_out_dir(npz_path: Path, out_dir: Path) -> Optional[str]:
    try:
        return str(npz_path.relative_to(out_dir))
    except ValueError:
        return None


def copy_manifest_records_except(
    src_out_dir: Path,
    dst_out_dir: Path,
    *,
    skip_relative_npz: set[str],
) -> List[dict]:
    """
    Copy NPZs from a previous prep for records that are *not* being reprocessed.

    Used to carry forward e.g. VCTK when this run only rescans LibriSpeech, or any
    other source absent from the current utterance list.
    """
    src_manifest = src_out_dir / "manifest.jsonl"
    if not src_manifest.is_file():
        raise FileNotFoundError(f"Previous manifest not found: {src_manifest}")

    src_records = load_manifest_records(src_manifest)
    kept: List[dict] = []
    n_missing = 0
    n_skipped = 0
    for rec in src_records:
        src_npz = Path(str(rec.get("npz", "")))
        if not src_npz.is_file():
            n_missing += 1
            continue
        rel = _npz_rel_in_out_dir(src_npz, src_out_dir)
        if rel is None:
            rel = _npz_rel_in_out_dir(src_npz, dst_out_dir)
        if rel is None:
            n_missing += 1
            continue
        if rel in skip_relative_npz:
            n_skipped += 1
            continue

        dst_npz = dst_out_dir / rel
        if dst_out_dir.resolve() != src_out_dir.resolve() or not dst_npz.is_file():
            dst_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_npz, dst_npz)

        new = dict(rec)
        new["npz"] = str(dst_npz)
        kept.append(new)

    if n_missing:
        log.warning(
            "While copying from %s: %d record(s) referenced missing/unmappable NPZs",
            src_out_dir, n_missing,
        )
    if n_skipped:
        log.info(
            "Skipped copying %d record(s) already scheduled for reprocessing",
            n_skipped,
        )
    kept.sort(key=lambda r: r["npz"])
    log.info(
        "Copied %d NPZ(s) from %s -> %s (not in current utterance scan)",
        len(kept), src_out_dir, dst_out_dir,
    )
    return kept


def _build_npz_reuse_map(
    manifest_records: Sequence[dict],
    src_out_dir: Path,
) -> Dict[str, Path]:
    """
    Build lookup keys -> NPZ path for reusing z_e without re-encoding.

    Keys: wav stem, and speaker-relative path under src_out_dir.
    """
    out: Dict[str, Path] = {}
    for rec in manifest_records:
        npz = Path(str(rec.get("npz", "")))
        if not npz.is_file():
            continue
        rel = _npz_rel_in_out_dir(npz, src_out_dir)
        if rel is not None:
            out[rel] = npz
        stem = Path(str(rec.get("wav", ""))).stem or npz.stem
        out.setdefault(stem, npz)
    return out


def lookup_reuse_npz(
    utt: Utterance,
    reuse_map: Dict[str, Path],
) -> Optional[Path]:
    rel = utterance_npz_rel(utt)
    if rel in reuse_map:
        return reuse_map[rel]
    stem = Path(utt.wav_path).stem
    return reuse_map.get(stem)


def _process_one_reuse_ze(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
    old_npz: Path,
    *,
    full_phone: bool,
    char2id: Dict[str, int],
) -> Optional[dict]:
    """
    Reuse z_e from an existing NPZ and write a new NPZ with masks recomputed
    from utt.phone_spans (the new MFA alignment).  No GPU / EnCodec needed.
    """
    try:
        with np.load(old_npz) as d:
            z_e = d["z_e"]  # float16 [T, D]
    except Exception as exc:
        log.warning("Cannot load z_e from %s: %s", old_npz, exc)
        return None

    T = z_e.shape[0]
    if T == 0:
        return None

    arrays: Dict[str, np.ndarray] = {"z_e": z_e}
    for ph in phones:
        spans = utt.phone_spans.get(ph, [])
        arrays[mask_key(ph)] = build_phone_mask(spans, T)

    if not any(arrays[mask_key(ph)].any() for ph in phones if mask_key(ph) in arrays):
        log.warning(
            "No masked frames for %s after reusing z_e (T=%d) — skipped",
            Path(utt.wav_path).name, T,
        )
        return None

    if full_phone and not attach_phone_seq_arrays(arrays, utt, T, char2id):
        log.warning(
            "No full-phone sequence for %s after reusing z_e (T=%d) — skipped",
            Path(utt.wav_path).name, T,
        )
        return None

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return manifest_record_from_npz(
        npz_path, Path(utt.wav_path), utt.speaker, source=utt.source,
        full_phone=full_phone,
    )


def process_one(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
    model,
    device: str,
    *,
    add_phones: bool,
    full_phone: bool,
    char2id: Dict[str, int],
) -> Optional[dict]:
    if add_phones:
        return _process_one_add_phones(
            utt, npz_path, phones, model, device,
            full_phone=full_phone, char2id=char2id,
        )
    return _process_one_replace(
        utt, npz_path, phones, model, device,
        full_phone=full_phone, char2id=char2id,
    )


def _process_one_replace(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
    model,
    device: str,
    *,
    full_phone: bool,
    char2id: Dict[str, int],
) -> Optional[dict]:
    wav = load_wav_padded(utt)
    if wav is None or len(wav) == 0:
        return None

    try:
        z_e = extract_ze(model, wav, device)
    except Exception as exc:
        log.warning("EnCodec failed for %s: %s", Path(utt.wav_path).name, exc)
        return None

    T = z_e.shape[0]
    if T == 0:
        return None

    arrays: Dict[str, np.ndarray] = {"z_e": z_e}
    n_mask_frames: Dict[str, int] = {}
    for ph in phones:
        spans = utt.phone_spans.get(ph, [])
        mask = build_phone_mask(spans, T)
        arrays[mask_key(ph)] = mask
        n_mask_frames[ph] = int(mask.sum())

    if not any(n_mask_frames.values()):
        log.warning(
            "No masked frames for %s after encoding (T=%d) — skipped",
            Path(utt.wav_path).name, T,
        )
        return None

    if full_phone and not attach_phone_seq_arrays(arrays, utt, T, char2id):
        log.warning(
            "No full-phone sequence for %s after encoding (T=%d) — skipped",
            Path(utt.wav_path).name, T,
        )
        return None

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return manifest_record_from_npz(
        npz_path, Path(utt.wav_path), utt.speaker, source=utt.source,
        full_phone=full_phone,
    )


def _process_one_add_phones(
    utt: Utterance,
    npz_path: Path,
    phones: Sequence[str],
    model,
    device: str,
    *,
    full_phone: bool,
    char2id: Dict[str, int],
) -> Optional[dict]:
    existing: Dict[str, np.ndarray] = {}
    need_encode, phones_to_write = phones_needing_mask_update(utt, npz_path, phones)

    if not need_encode:
        try:
            with np.load(npz_path) as d:
                existing = {k: d[k] for k in d.files}
        except Exception:
            need_encode = True
            phones_to_write = list(phones)
            existing = {}

    if need_encode:
        wav = load_wav_padded(utt)
        if wav is None or len(wav) == 0:
            return None
        try:
            z_e = extract_ze(model, wav, device)
        except Exception as exc:
            log.warning("EnCodec failed for %s: %s", Path(utt.wav_path).name, exc)
            return None
        phones_to_write = list(phones)
    else:
        z_e = existing["z_e"]

    T = z_e.shape[0]
    if T == 0:
        return None

    arrays = dict(existing)
    arrays["z_e"] = z_e
    for ph in phones_to_write:
        spans = utt.phone_spans.get(ph, [])
        arrays[mask_key(ph)] = build_phone_mask(spans, T)

    if full_phone and (need_encode or not npz_full_phone_ok(existing)):
        if not attach_phone_seq_arrays(arrays, utt, T, char2id):
            log.warning(
                "No full-phone sequence for %s (T=%d, phones=%s) — skipped",
                Path(utt.wav_path).name, T, list(phones),
            )
            return None

    if not any(arrays[mask_key(ph)].any() for ph in phones if mask_key(ph) in arrays):
        log.warning(
            "No masked frames for %s (T=%d, phones=%s) — skipped",
            Path(utt.wav_path).name, T, list(phones),
        )
        return None

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return manifest_record_from_npz(
        npz_path, Path(utt.wav_path), utt.speaker, source=utt.source,
        full_phone=full_phone,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    phones = list(dict.fromkeys(args.phone_masks))
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    cache_path = utterances_cache_path(out_dir)

    include_librispeech = not args.vctk_only
    include_vctk = not args.skip_vctk
    if not include_librispeech and not include_vctk:
        raise SystemExit("Nothing to do: both LibriSpeech and VCTK are disabled.")

    # Enforce the 100/360/500 subsets (user request). If the user passed others,
    # we silently drop them to avoid mixing inventories/config.
    subsets = [s for s in args.librispeech_subsets if s in DEFAULT_LIBRISPEECH_SUBSETS]
    dropped = [s for s in args.librispeech_subsets if s not in DEFAULT_LIBRISPEECH_SUBSETS]
    if dropped:
        log.warning("Ignoring unsupported LibriSpeech subsets: %s (keeping only %s)", dropped, subsets)

    libri_mfa = bool(getattr(args, "libri_mfa", False))
    copy_npz_from = resolve_copy_npz_from(args)

    full_phone = bool(args.full_phone_conditioning)
    _, char2id = build_alignment_char_vocab()

    log.info(
        "Phones: %s  add_phones=%s  full_phone=%s  FS=%d  fps=%.0f  output=%s  "
        "libri=%s  vctk=%s (libri_mfa=%s)  copy_npz_from=%s",
        phones, args.add_phones, full_phone, FS, ENCODEC_FPS, out_dir,
        include_librispeech, include_vctk, libri_mfa,
        copy_npz_from or "(none)",
    )
    for ph in phones:
        al = align_label(ph)
        if al != ph:
            log.info("  alignment alias: %r -> %r", ph, al)

    copied_records: List[dict] = []
    reuse_map: Dict[str, Path] = {}

    utterances = load_or_collect_utterances(
        phones=phones,
        librispeech_subsets=subsets,
        include_librispeech=include_librispeech,
        include_vctk=include_vctk,
        audio_root=Path(args.librispeech_audio),
        align_root=Path(args.librispeech_align),
        vctk_audio=Path(args.vctk_audio),
        vctk_aligned=Path(args.vctk_aligned),
        libri_mfa=libri_mfa,
        num_workers=args.num_workers,
        cache_path=cache_path,
        refresh=args.refresh_utterances,
    )

    processing_rel_npz = {utterance_npz_rel(utt) for utt in utterances}

    if copy_npz_from is not None:
        old_records = load_manifest_records(copy_npz_from / "manifest.jsonl")
        reuse_map = _build_npz_reuse_map(old_records, copy_npz_from)
        log.info(
            "Reuse-z_e map: %d NPZ(s) available from %s",
            len({id(p) for p in reuse_map.values()}), copy_npz_from,
        )
        copied_records = copy_manifest_records_except(
            copy_npz_from, out_dir, skip_relative_npz=processing_rel_npz,
        )

    if not utterances and not copied_records:
        raise SystemExit("No utterances collected — check paths, subsets, and phone_masks.")

    previous_manifest: List[dict] = []
    add_phones_plan: Optional[dict] = None
    if args.add_phones:
        previous_manifest = load_manifest_records(manifest_path)
        add_phones_plan = plan_add_phones_manifest(previous_manifest, utterances, out_dir)
        log.info(
            "Add-phones plan: existing manifest=%d; utterances with %s=%d "
            "(%d already listed, %d new); %d existing records lack %s and stay unchanged; "
            "expected final manifest=%d",
            add_phones_plan["previous"],
            phones,
            add_phones_plan["to_process"],
            add_phones_plan["already_in_manifest"],
            add_phones_plan["new_not_in_manifest"],
            add_phones_plan["kept_without_new_phones"],
            phones,
            add_phones_plan["expected_final"],
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.encodec is None:
        from voicecraft_encodec import DEFAULT_CHECKPOINT
        encodec_ckpt = DEFAULT_CHECKPOINT
    else:
        encodec_ckpt = Path(args.encodec)

    # EnCodec model is loaded lazily — only when the first utterance that actually
    # needs fresh encoding is encountered.  If every utterance has a reusable z_e
    # (via --copy_npz_from), the GPU is never touched.
    model = None

    def _ensure_model():
        nonlocal model
        if model is not None:
            return model
        log.info("Loading VoiceCraft EnCodec from %s on %s", encodec_ckpt, device)
        model = load_encodec(device, encodec_ckpt)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, FS, device=device)
            actual_dim = int(model.encoder(dummy).shape[1])
            del dummy
        clear_gpu_cache(device)
        if actual_dim != ENCODEC_DIM:
            log.warning("EnCodec latent dim=%d (expected %d)", actual_dim, ENCODEC_DIM)
        return model

    if args.add_phones and add_phones_plan is not None:
        log.info(
            "Processing %d utterances for new phone masks -> %s  (gpu_clear_every=%d; "
            "final manifest ~%d after merge)",
            len(utterances), out_dir, args.gpu_clear_every,
            add_phones_plan["expected_final"],
        )
    else:
        log.info(
            "Processing %d utterances -> %s  (gpu_clear_every=%d)",
            len(utterances), out_dir, args.gpu_clear_every,
        )
    if reuse_map:
        log.info(
            "  z_e reuse enabled: will skip EnCodec for utterances found in copy map "
            "(%d unique NPZ(s)); model loaded lazily only if needed.",
            len({id(p) for p in reuse_map.values()}),
        )

    records: List[dict] = list(copied_records)
    n_ok = n_skip = n_fail = n_reprocess = 0
    n_reuse_ze = 0
    gpu_clear_every = max(1, args.gpu_clear_every)

    for i, utt in enumerate(utterances):
        wav_path = Path(utt.wav_path)
        npz_path = utterance_npz_path(utt, out_dir)

        if npz_path.exists() and npz_masks_up_to_date(
            utt, npz_path, phones, add_phones=args.add_phones, full_phone=full_phone,
        ):
            records.append(manifest_record_from_npz(
                npz_path, wav_path, utt.speaker, source=utt.source,
                full_phone=full_phone,
            ))
            n_skip += 1
            if (i + 1) % gpu_clear_every == 0 and model is not None:
                clear_gpu_cache(device)
            continue

        if npz_path.exists():
            n_reprocess += 1

        # Reuse z_e from a previous prep if available (avoids re-encoding).
        old_npz = lookup_reuse_npz(utt, reuse_map) if reuse_map else None
        if old_npz is not None:
            rec = _process_one_reuse_ze(
                utt, npz_path, phones, old_npz,
                full_phone=full_phone, char2id=char2id,
            )
            if rec is not None:
                records.append(rec)
                n_ok += 1
                n_reuse_ze += 1
            else:
                # Reuse failed (z_e unreadable); fall through to fresh encoding.
                log.warning("z_e reuse failed for %s; falling back to EnCodec", wav_path.name)
                rec = process_one(
                    utt, npz_path, phones, _ensure_model(), device,
                    add_phones=args.add_phones,
                    full_phone=full_phone, char2id=char2id,
                )
                if rec is None:
                    n_fail += 1
                else:
                    records.append(rec)
                    n_ok += 1
        else:
            rec = process_one(
                utt, npz_path, phones, _ensure_model(), device,
                add_phones=args.add_phones,
                full_phone=full_phone, char2id=char2id,
            )
            if rec is None:
                n_fail += 1
            else:
                records.append(rec)
                n_ok += 1

        if (i + 1) % gpu_clear_every == 0 and model is not None:
            clear_gpu_cache(device)

        if (i + 1) % 250 == 0:
            if args.add_phones and add_phones_plan is not None:
                log.info(
                    "  %d/%d utterances for %s masks  (new=%d  reuse_ze=%d  skipped=%d  "
                    "reprocess=%d  failed=%d; manifest -> ~%d)",
                    i + 1, len(utterances), phones, n_ok, n_reuse_ze, n_skip,
                    n_reprocess, n_fail, add_phones_plan["expected_final"],
                )
            else:
                log.info(
                    "  %d/%d  (new=%d  reuse_ze=%d  skipped=%d  reprocess=%d  failed=%d)",
                    i + 1, len(utterances), n_ok, n_reuse_ze, n_skip, n_reprocess, n_fail,
                )

    if model is not None:
        clear_gpu_cache(device)

    if args.add_phones:
        records = merge_manifest_records(previous_manifest, records)

    write_manifest(manifest_path, records)

    total_f = sum(r["n_frames"] for r in records)
    manifest_phones = sorted({ph for r in records for ph in r.get("phone_masks", [])})
    log.info("Manifest: %d records -> %s", len(records), manifest_path)
    if args.add_phones and add_phones_plan is not None:
        log.info(
            "Manifest merge: was %d; now %d (+%d new npz, %d unchanged)",
            add_phones_plan["previous"],
            len(records),
            len(records) - add_phones_plan["previous"],
            add_phones_plan["kept_without_new_phones"],
        )
    log.info(
        "New=%d (of which reused_ze=%d)  Skipped=%d  Reprocessed=%d  Failed=%d",
        n_ok, n_reuse_ze, n_skip, n_reprocess, n_fail,
    )
    log.info("Total frames: %d", total_f)
    for ph in manifest_phones:
        total_ph = sum(r["n_mask_frames"].get(ph, 0) for r in records)
        log.info(
            "  %s_mask: %d frames (%.2f%% of total)",
            ph, total_ph, 100.0 * total_ph / max(total_f, 1),
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--phone_masks", nargs="+", metavar="PHONE",
        default=list(DEFAULT_TARGET_PHONES),
        help="Phones to store as separate bool masks "
             f"(default: all {len(DEFAULT_TARGET_PHONES)} phones in DEFAULT_TARGET_PHONES)",
    )
    p.add_argument(
        "--add_phones", action="store_true",
        help="Merge new masks into existing NPZs without re-encoding; pass only "
             "the phones to add. With --full_phone_conditioning, also writes "
             "phone_ids/align arrays when missing (reuses existing z_e).",
    )
    p.add_argument(
        "--full_phone_conditioning", action="store_true",
        help="Store char-tokenized full utterance phone strings in each NPZ "
             "(phone_ids, phone_starts, phone_ends, phone_nchars) for A3T-style "
             "training with train_phone_ec_cond.py --full_phone_conditioning.",
    )
    p.add_argument(
        "--librispeech_subsets", nargs="+",
        default=list(DEFAULT_LIBRISPEECH_SUBSETS),
        help="LibriSpeech subsets to include (default: train-clean-100/360 + train-other-500)",
    )
    p.add_argument(
        "--Libri_MFA",
        dest="libri_mfa",
        action="store_true",
        help="Use LibriSpeech MFA JSON alignments under --librispeech_align/aligned_* "
             "(instead of per-utterance ARPAbet TextGrids).",
    )
    p.add_argument(
        "--librispeech_audio", default=str(LIBRISPEECH_AUDIO_ROOT),
        help=f"LibriSpeech audio root (default: {LIBRISPEECH_AUDIO_ROOT})",
    )
    p.add_argument(
        "--librispeech_align", default=str(LIBRISPEECH_ALIGN_ROOT),
        help=f"LibriSpeech phone TextGrid root (default: {LIBRISPEECH_ALIGN_ROOT})",
    )
    p.add_argument(
        "--copy_npz_from",
        default=None,
        help="Previous prep output_dir (must contain manifest.jsonl).  Utterances in "
             "the current scan reuse z_e from matching NPZs (masks / phone strings "
             "recomputed; no re-encoding).  Records from sources *not* in the current "
             "scan are copied verbatim into --output_dir (e.g. keep VCTK when "
             "rescans Libri only with --skip_vctk).  Matches by speaker-relative NPZ "
             "path or wav stem.",
    )
    p.add_argument(
        "--copy_vctk_from",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--copy_libri_npz_from",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Worker threads for Libri MFA JSON scanning (default: 1). "
             "This speeds up utterance collection; EnCodec encoding remains sequential.",
    )
    p.add_argument(
        "--vctk_audio", default=str(VCTK_AUDIO_BASE),
        help=f"VCTK wav root (24 kHz; downsampled to {FS} Hz on load)",
    )
    p.add_argument(
        "--vctk_aligned", default=str(VCTK_ALIGNED_DIR),
        help=f"VCTK MFA JSON alignments (default: {VCTK_ALIGNED_DIR})",
    )
    p.add_argument("--skip_vctk", action="store_true", help="LibriSpeech only")
    p.add_argument("--vctk_only", action="store_true", help="VCTK only")
    p.add_argument(
        "--output_dir", default=None,
        help=f"NPZ + manifest root (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--encodec", default=None,
        help="VoiceCraft encodec_4cb2048_giga.th checkpoint path "
             "(default: $VOICECRAFT_ROOT/pretrained_models/encodec_4cb2048_giga.th)",
    )
    p.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--refresh_utterances", action="store_true",
        help="Ignore cached utterance list and rescan alignments",
    )
    p.add_argument(
        "--gpu_clear_every", type=int, default=DEFAULT_GPU_CLEAR_EVERY,
        help=f"Call torch.cuda.empty_cache every N utterances (default: {DEFAULT_GPU_CLEAR_EVERY})",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
