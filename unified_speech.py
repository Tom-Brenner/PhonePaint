#!/usr/bin/env python3
"""Unified segmentation, Torch/Whisper transcription, and BFA phone alignment."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import soxr

from phone_labels import canonical_phone_label

log = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
# Kept for CLI flag compatibility; Applio segmentation ignores these.
DEFAULT_MAX_SEGMENT_SEC = 28.0
DEFAULT_TARGET_SEGMENT_SEC = 18.0
# Drop/merge Applio tail stubs below this length (Whisper hallucinates; MFA fails).
MIN_SEGMENT_SEC = 1.0
_APPLIO_SEGMENT = Path(__file__).resolve().parent / "tools" / "segment" / "segment_wav_directory.py"


def _load_applio_segment():
    """Load Applio-identical segment_file from tools/segment/."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("applio_segment_wav_directory", _APPLIO_SEGMENT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Applio segmenter: {_APPLIO_SEGMENT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mono_float32(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1, dtype=np.float32)
    if sample_rate is not None and sr != sample_rate:
        mono = soxr.resample(mono, sr, sample_rate, quality="HQ").astype(np.float32)
        sr = sample_rate
    return mono, sr


def _segment_index(name: str) -> int:
    # democracy2_segment_4_segment_1.wav → 1
    stem = Path(name).stem
    marker = "_segment_"
    pos = stem.rfind(marker)
    if pos < 0:
        raise ValueError(f"Not a segment filename: {name}")
    return int(stem[pos + len(marker):])


def _collapse_short_segments(
    output_dir: Path,
    names: Sequence[str],
    *,
    min_segment_sec: float = MIN_SEGMENT_SEC,
) -> list[str]:
    """Merge chunks shorter than ``min_segment_sec`` into the previous (or next) neighbor.

    Applio's tail cut often leaves a sub-second stub on mid-length files; Whisper
    then hallucinates and MFA/BFA fail. Audio coverage is preserved.
    """
    if len(names) <= 1:
        return list(names)

    paths = [output_dir / n for n in names]
    chunks: list[tuple[np.ndarray, int]] = []
    for path in paths:
        audio, sr = _mono_float32(path)
        chunks.append((audio, sr))

    # Merge left-to-right: absorb short chunks into the previous kept chunk.
    merged: list[tuple[np.ndarray, int]] = []
    for audio, sr in chunks:
        dur = len(audio) / float(sr) if sr else 0.0
        if merged and dur < min_segment_sec:
            prev_audio, prev_sr = merged[-1]
            if prev_sr != sr:
                audio = soxr.resample(audio, sr, prev_sr, quality="HQ").astype(np.float32)
                sr = prev_sr
            merged[-1] = (np.concatenate([prev_audio, audio]), sr)
            log.info(
                "Merged %.3fs stub into previous segment (min=%.2fs)",
                dur, min_segment_sec,
            )
            continue
        merged.append((audio, sr))

    # If the first chunk was short and nothing preceded it, fold into the next.
    if len(merged) >= 2:
        first_audio, first_sr = merged[0]
        first_dur = len(first_audio) / float(first_sr) if first_sr else 0.0
        if first_dur < min_segment_sec:
            second_audio, second_sr = merged[1]
            if first_sr != second_sr:
                first_audio = soxr.resample(
                    first_audio, first_sr, second_sr, quality="HQ",
                ).astype(np.float32)
                first_sr = second_sr
            merged[1] = (np.concatenate([first_audio, second_audio]), first_sr)
            del merged[0]
            log.info(
                "Merged leading %.3fs stub into following segment (min=%.2fs)",
                first_dur, min_segment_sec,
            )

    if len(merged) == len(names):
        return list(names)

    # Derive stem prefix from the first name: {stem}_segment_{i}.wav
    first = Path(names[0]).stem
    marker = "_segment_"
    pos = first.rfind(marker)
    stem_prefix = first[:pos] if pos >= 0 else first

    for old in paths:
        old.unlink(missing_ok=True)

    out_names: list[str] = []
    for i, (audio, sr) in enumerate(merged):
        name = f"{stem_prefix}_segment_{i}.wav"
        sf.write(str(output_dir / name), audio, sr, subtype="PCM_16")
        out_names.append(name)
        log.info(
            "Rewrote %s (%.3f s @ %d Hz) after short-segment collapse",
            name, len(audio) / float(sr), sr,
        )
    return out_names


def segment_wav_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    silence_level: float = -40.0,
    min_split_sec: float = 6.0,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
    target_segment_sec: float = DEFAULT_TARGET_SEGMENT_SEC,
    min_segment_sec: float = MIN_SEGMENT_SEC,
) -> list[str]:
    """Split WAVs with Applio-identical cuts (same as ~/a3t and tools/segment/).

    Files shorter than ``min_split_sec`` stay one chunk at the source rate
    (matches ``run_inpaint_pipeline`` / a3t). Longer files use pydub silence
    detection and ffmpeg 32 kHz mono output, matching Applio41. After Applio
    cuts, chunks shorter than ``min_segment_sec`` are merged into a neighbor so
    ASR/align never see sub-second stubs.
    """
    del max_segment_sec, target_segment_sec
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.wav"):
        old.unlink()

    applio = _load_applio_segment()
    names: list[str] = []
    for path in sorted(input_dir.glob("*.wav")):
        duration = float(sf.info(str(path)).duration)
        if duration < min_split_sec:
            name = f"{path.stem}_segment_0.wav"
            shutil.copy2(path, output_dir / name)
            names.append(name)
            log.info(
                "Skipped split for %.2fs input → %s",
                duration,
                name,
            )
            continue
        before = {p.name for p in output_dir.glob("*.wav")}
        applio.segment_file(path, output_dir, silence_level)
        written = sorted(
            (p.name for p in output_dir.glob("*.wav") if p.name not in before),
            key=_segment_index,
        )
        if not written:
            raise RuntimeError(f"Applio segmenter wrote no chunks for {path}")
        for name in written:
            info = sf.info(str(output_dir / name))
            log.info(
                "Segmented %s -> %s (%.3f s @ %d Hz)",
                path.name, name, info.duration, info.samplerate,
            )
        written = _collapse_short_segments(
            output_dir, written, min_segment_sec=min_segment_sec,
        )
        names.extend(written)
    return names


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def configure_espeak() -> None:
    """Point Phonemizer at the environment-local espeak-ng distribution."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    set_data_path = getattr(EspeakWrapper, "set_data_path", None)
    if set_data_path is not None:
        set_data_path(espeakng_loader.get_data_path())


def transcribe_directory(
    input_dir: Path,
    output_json: Path,
    *,
    model_name: str = DEFAULT_ASR_MODEL,
    language: str | None = "en",
    device: str = "cuda",
    device_idx: int = 0,
    batch_size: int = 4,
) -> dict[str, dict]:
    """Transcribe segment WAVs with a Hugging Face Whisper model."""
    import torch
    from transformers import pipeline

    wav_paths = sorted(p for p in input_dir.glob("*.wav") if p.is_file())
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    pipeline_device = device_idx if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=pipeline_device,
        dtype=dtype,
        model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True},
    )

    audio_inputs: list[dict] = []
    for path in wav_paths:
        wav, _ = _mono_float32(path, sample_rate=16_000)
        audio_inputs.append({"array": wav, "sampling_rate": 16_000})

    is_multilingual = bool(
        getattr(asr.model.generation_config, "is_multilingual", True)
    )
    generate_kwargs: dict[str, str] = {}
    if is_multilingual:
        generate_kwargs["task"] = "transcribe"
        if language and language.lower() not in {"auto", "none"}:
            generate_kwargs["language"] = language
    raw_results = asr(
        audio_inputs,
        batch_size=max(1, batch_size),
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    if isinstance(raw_results, dict):
        raw_results = [raw_results]

    output: dict[str, dict] = {}
    for path, result in zip(wav_paths, raw_results):
        text = " ".join(str(result.get("text", "")).strip().split())
        if not text:
            raise RuntimeError(f"ASR returned an empty transcript for {path.name}")
        output[path.name] = {"text": text, "model": model_name}
        log.info("Transcribed %s -> %r", path.name, text)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    del asr
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return output


def _transcript_texts(path: Path, segment_names: Sequence[str]) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected transcript object in {path}")
    texts: list[str] = []
    for name in segment_names:
        entry = data.get(name)
        text = entry.get("text") if isinstance(entry, dict) else entry
        text = str(text or "").strip()
        if not text:
            raise ValueError(f"Missing transcript text for {name} in {path}")
        texts.append(text)
    return texts


def _seconds_interval(item: dict, *, duration: float) -> tuple[float, float] | None:
    start = max(0.0, float(item["start_ms"]) / 1000.0)
    end = min(duration, float(item["end_ms"]) / 1000.0)
    if end <= start:
        return None
    return round(start, 6), round(end, 6)


def _bfa_result_to_alignment(result: dict, *, duration: float) -> dict:
    segments = result.get("segments") or []
    if not segments:
        raise RuntimeError("BFA returned no segments")
    segment = segments[0]
    coverage = segment.get("coverage_analysis") or {}

    native_phones = segment.get("phoneme_ts") or []
    word_numbers = segment.get("word_num") or []
    phone_records: list[dict] = []
    for native_index, native in enumerate(native_phones):
        interval = _seconds_interval(native, duration=duration)
        if interval is None:
            continue
        ipa = str(native.get("ipa_label") or native.get("phoneme_label") or "").strip()
        if not ipa:
            continue
        start, end = interval
        target_index = int(native.get("target_seq_idx", native_index))
        word_number = (
            word_numbers[target_index]
            if 0 <= target_index < len(word_numbers)
            else None
        )
        phone_records.append({
            "xmin": start,
            "xmax": end,
            "core_xmin": start,
            "core_xmax": end,
            "text": ipa,
            "canonical": canonical_phone_label(ipa),
            "confidence": float(native.get("confidence", 0.0)),
            "is_estimated": bool(native.get("is_estimated", False)),
            "inventory": "bfa-espeak-ipa",
            "_word_number": word_number,
        })

    # BFA timestamps are high-confidence acoustic cores, so they commonly leave
    # gaps that would make a PhonePaint mask much shorter than the spoken phone.
    # Within a word, assign short gaps to the midpoint of neighboring cores.
    for previous, current in zip(phone_records, phone_records[1:]):
        gap = float(current["xmin"]) - float(previous["xmax"])
        same_word = (
            previous["_word_number"] is not None
            and previous["_word_number"] == current["_word_number"]
        )
        if same_word and 0.0 < gap <= 0.25:
            boundary = round(
                (float(previous["xmax"]) + float(current["xmin"])) / 2.0,
                6,
            )
            previous["xmax"] = boundary
            current["xmin"] = boundary

    phones: dict[str, dict] = {}
    for record in phone_records:
        record.pop("_word_number", None)
        phones[str(len(phones))] = record

    words: dict[str, dict] = {}
    for native in segment.get("words_ts") or []:
        interval = _seconds_interval(native, duration=duration)
        if interval is None:
            continue
        text = str(native.get("word", "")).strip()
        if not text:
            continue
        start, end = interval
        words[str(len(words))] = {
            "xmin": start,
            "xmax": end,
            "text": text,
            "confidence": float(native.get("confidence", 0.0)),
        }

    if not phones:
        raise RuntimeError("BFA returned no valid phone intervals")
    return {
        "words": words,
        "phones": phones,
        "alignment": {
            "backend": "bfa",
            "inventory": "bfa-espeak-ipa",
            "boundary_policy": "within-word-core-midpoint",
            "coverage_ratio": float(coverage.get("coverage_ratio", 1.0)),
        },
    }


def align_directory_bfa(
    audio_dir: Path,
    transcripts_json: Path,
    output_json: Path,
    *,
    segment_names: Sequence[str] | None = None,
    language: str = "en-us",
    device: str = "cuda",
    batch_size: int = 8,
    duration_max: float = DEFAULT_MAX_SEGMENT_SEC,
    minimum_coverage: float = 0.95,
) -> dict[str, dict]:
    """Align transcripts to phones with one cached BFA model."""
    import torch
    configure_espeak()
    from bournemouth_aligner import PhonemeTimestampAligner

    if segment_names is None:
        segment_names = [p.name for p in sorted(audio_dir.glob("*.wav"))]
    segment_names = list(segment_names)
    texts = _transcript_texts(transcripts_json, segment_names)
    preset = "en-us" if language.lower() in {"en", "english", "en_us"} else language

    aligner = PhonemeTimestampAligner(
        preset=preset,
        duration_max=duration_max,
        device=device,
        silence_anchors=0,
        boost_targets=True,
        enforce_minimum=True,
        enforce_all_targets=True,
        ensure_completeness=False,
        ignore_noise=True,
        extend_soft_boundaries=True,
        boundary_softness=3,
    )

    output: dict[str, dict] = {}
    indexed = list(zip(segment_names, texts))
    for chunk in _batched(indexed, max(1, batch_size)):
        names = [name for name, _ in chunk]
        chunk_texts = [text for _, text in chunk]
        waves = [aligner.load_audio(str(audio_dir / name)) for name in names]
        try:
            results = aligner.process_sentences_batch(
                chunk_texts, waves, extract_embeddings=False, do_groups=False,
            )
        except Exception:
            log.exception("BFA batch failed; retrying %d item(s) individually", len(chunk))
            results = [
                aligner.process_sentence(text, wav, extract_embeddings=False, do_groups=False)
                for text, wav in zip(chunk_texts, waves)
            ]
        for name, result in zip(names, results):
            duration = sf.info(str(audio_dir / name)).duration
            converted = _bfa_result_to_alignment(result, duration=duration)
            coverage = converted["alignment"]["coverage_ratio"]
            if coverage < minimum_coverage:
                raise RuntimeError(
                    f"BFA coverage {coverage:.1%} for {name} is below "
                    f"{minimum_coverage:.1%}"
                )
            output[name] = converted
            log.info(
                "Aligned %s: %d phones, %.1f%% coverage",
                name, len(converted["phones"]), coverage * 100.0,
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    del aligner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    segment = sub.add_parser("segment")
    segment.add_argument("--input-dir", type=Path, required=True)
    segment.add_argument("--output-dir", type=Path, required=True)
    segment.add_argument("--silence-level", type=float, default=-40.0)
    segment.add_argument("--max-segment-sec", type=float, default=DEFAULT_MAX_SEGMENT_SEC)

    transcribe = sub.add_parser("transcribe")
    transcribe.add_argument("--input-dir", type=Path, required=True)
    transcribe.add_argument("--output-json", type=Path, required=True)
    transcribe.add_argument("--model", default=DEFAULT_ASR_MODEL)
    transcribe.add_argument("--language", default="en")
    transcribe.add_argument("--device", default="cuda")
    transcribe.add_argument("--device-idx", type=int, default=0)
    transcribe.add_argument("--batch-size", type=int, default=4)

    align = sub.add_parser("align")
    align.add_argument("--audio-dir", type=Path, required=True)
    align.add_argument("--transcripts-json", type=Path, required=True)
    align.add_argument("--output-json", type=Path, required=True)
    align.add_argument("--language", default="en-us")
    align.add_argument("--device", default="cuda")
    align.add_argument("--batch-size", type=int, default=8)
    align.add_argument("--minimum-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.command == "segment":
        segment_wav_directory(
            args.input_dir,
            args.output_dir,
            silence_level=args.silence_level,
            max_segment_sec=args.max_segment_sec,
        )
    elif args.command == "transcribe":
        transcribe_directory(
            args.input_dir,
            args.output_json,
            model_name=args.model,
            language=args.language,
            device=args.device,
            device_idx=args.device_idx,
            batch_size=args.batch_size,
        )
    else:
        align_directory_bfa(
            args.audio_dir,
            args.transcripts_json,
            args.output_json,
            language=args.language,
            device=args.device,
            batch_size=args.batch_size,
            minimum_coverage=args.minimum_coverage,
        )


if __name__ == "__main__":
    main()
