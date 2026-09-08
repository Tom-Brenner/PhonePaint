"""Shared helpers for run_*_inpaint_pipeline.py drivers."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import soundfile as sf

log = logging.getLogger(__name__)

SEGMENT_RE = re.compile(r"_segment_(\d+)\.wav$", re.IGNORECASE)
MIN_SPLIT_SEC = 6.0


def _query_working_gpus() -> list[tuple[int, int]]:
    """Return (nvidia-smi index, free MiB) for GPUs that respond to nvidia-smi."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus: list[tuple[int, int]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpus.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return sorted(gpus, key=lambda pair: pair[0])


def _smi_to_torch_idx(smi_idx: int, working: list[tuple[int, int]]) -> int:
    """Map an nvidia-smi GPU index to the ordinal PyTorch/CUDA exposes."""
    smi_indices = [idx for idx, _ in working]
    if smi_idx not in smi_indices:
        raise ValueError(
            f"nvidia-smi GPU {smi_idx} is unavailable; working GPUs: {smi_indices or 'none'}"
        )
    return smi_indices.index(smi_idx)


def pick_asr_device_idx(explicit: Optional[int] = None) -> int:
    """Return PyTorch CUDA index for CrisperWhisper (auto-picks most free VRAM)."""
    try:
        working = _query_working_gpus()
        if not working:
            raise RuntimeError("no GPUs responded to nvidia-smi")

        if explicit is not None:
            torch_idx = _smi_to_torch_idx(explicit, working)
            free_mib = next(free for idx, free in working if idx == explicit)
            log.info(
                "ASR using CUDA device %d (nvidia-smi GPU %d, %d MiB free)",
                torch_idx,
                explicit,
                free_mib,
            )
            return torch_idx

        smi_idx, free_mib = max(working, key=lambda pair: pair[1])
        torch_idx = _smi_to_torch_idx(smi_idx, working)
        if torch_idx != smi_idx:
            log.info(
                "ASR using CUDA device %d (nvidia-smi GPU %d, %d MiB free)",
                torch_idx,
                smi_idx,
                free_mib,
            )
        else:
            log.info("ASR using CUDA device %d (%d MiB free)", torch_idx, free_mib)
        return torch_idx
    except Exception as exc:
        if explicit is not None:
            raise
        log.warning("nvidia-smi unavailable (%s); ASR defaulting to CUDA device 0", exc)
        return 0


def input_wavs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.wav") if p.is_file())


def all_inputs_shorter_than(input_dir: Path, max_sec: float = MIN_SPLIT_SEC) -> bool:
    """True when every ``*.wav`` under *input_dir* is shorter than *max_sec*."""
    wavs = input_wavs(input_dir)
    if not wavs:
        return False
    return all(sf.info(str(w)).duration < max_sec for w in wavs)


def copy_as_single_segments(input_dir: Path, segments_dir: Path) -> list[str]:
    """
    Treat each input wav as one chunk: ``{stem}_segment_0.wav``.

    Used when audio is too short to benefit from silence-based splitting.
    """
    if segments_dir.is_dir():
        for old in segments_dir.glob("*.wav"):
            old.unlink()
    else:
        segments_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for wav in input_wavs(input_dir):
        dst_name = f"{wav.stem}_segment_0.wav"
        dst = segments_dir / dst_name
        shutil.copy2(wav, dst)
        names.append(dst_name)
        log.info(
            "Skipped split for %.2fs input → %s",
            sf.info(str(wav)).duration,
            dst_name,
        )
    return names


def source_wav_name(segment_name: str) -> str:
    """``foo_segment_3.wav`` → ``foo.wav``."""
    m = SEGMENT_RE.search(segment_name)
    if m:
        return f"{segment_name[:m.start()]}.wav"
    return segment_name


def segment_index(segment_name: str) -> int:
    m = SEGMENT_RE.search(segment_name)
    return int(m.group(1)) if m else 10**9


def group_segments_by_source(segment_names: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in segment_names:
        groups.setdefault(source_wav_name(name), []).append(name)
    for names in groups.values():
        names.sort(key=segment_index)
    return groups


def per_segment_alignments(aligned: dict) -> dict:
    """Keep only per-chunk alignment entries (exclude merged full-file keys)."""
    return {
        k: v for k, v in aligned.items()
        if SEGMENT_RE.search(k) and isinstance(v, dict)
    }


def _shift_tier(tier: dict, offset: float) -> dict:
    ordered = sorted(
        tier.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
    )
    out: dict[str, dict] = {}
    for _, item in ordered:
        if not isinstance(item, dict) or "text" not in item:
            continue
        shifted = {
            "xmin": round(float(item["xmin"]) + offset, 3),
            "xmax": round(float(item["xmax"]) + offset, 3),
            "text": item["text"],
        }
        for key in ("core_xmin", "core_xmax"):
            if key in item:
                shifted[key] = round(float(item[key]) + offset, 3)
        for key in ("canonical", "confidence", "is_estimated", "inventory"):
            if key in item:
                shifted[key] = item[key]
        out[str(len(out))] = shifted
    return out


def merge_segment_alignments(
    per_segment: dict,
    segment_names: Sequence[str],
    segments_dir: Path,
) -> dict[str, dict]:
    """
    For each source input wav, merge per-chunk alignments into one entry whose
    word/phone timestamps are relative to the full original recording.
    """
    merged: dict[str, dict] = {}
    for source_name, names in group_segments_by_source(segment_names).items():
        words: dict[str, dict] = {}
        phones: dict[str, dict] = {}
        offset = 0.0
        for seg_name in names:
            entry = per_segment.get(seg_name)
            seg_path = segments_dir / seg_name
            if entry is None or not seg_path.is_file():
                log.warning("Skipping merge for missing segment %s", seg_name)
                if seg_path.is_file():
                    offset += sf.info(str(seg_path)).duration
                continue
            for item in _shift_tier(entry.get("words", {}), offset).values():
                words[str(len(words))] = item
            for item in _shift_tier(entry.get("phones", {}), offset).values():
                phones[str(len(phones))] = item
            offset += sf.info(str(seg_path)).duration
        merged[source_name] = {"words": words, "phones": phones}
    return merged


def write_aligned_json(
    per_segment: dict,
    segment_names: Sequence[str],
    segments_dir: Path,
    output_path: Path,
) -> None:
    """Write merged + per-segment alignments without dropping any prior keys.

    Always retains every ``segment_names`` entry and any existing keys already
    present in ``output_path`` (so a later writer cannot erase segment keys).
    """
    merged = merge_segment_alignments(per_segment, segment_names, segments_dir)
    combined: dict = {}
    if output_path.is_file():
        try:
            prev = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = None
        if isinstance(prev, dict):
            combined.update(prev)
    # Refresh segment keys from this align pass, then overlay source merges.
    combined.update(per_segment)
    combined.update(merged)
    missing = [name for name in segment_names if name not in combined]
    if missing:
        raise RuntimeError(
            f"write_aligned_json refused to drop segment key(s) {missing}; "
            f"present keys={sorted(combined)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def deploy_stitched_output(stitched: Path, output: Path) -> None:
    """Copy the workspace stitched wav to the user-facing --output path."""
    if not stitched.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stitched, output)
    log.info("Copied final output to %s", output)


def load_rewrite_texts(
    args_texts: Optional[Sequence[str]],
) -> Optional[list[str] | dict[str, str]]:
    """
    Parse ``--rewrite-transcription`` values.

    Accepts either:
      - N quoted strings (one per segment wav, in segment order) → ``list[str]``, or
      - a single path to a JSON file containing:
          * a flat list of strings → ``list[str]`` (matched by segment order), or
          * ``{segment_name: "text"}`` / ``{segment_name: {"text": "..."}}``
            → ``dict[str, str]`` (matched by filename).

    Nested array-of-arrays CLI forms are rejected.
    """
    if not args_texts:
        return None
    if len(args_texts) == 1:
        candidate = Path(args_texts[0])
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                texts: list[str] = []
                for i, item in enumerate(raw):
                    if isinstance(item, list):
                        raise SystemExit(
                            "--rewrite-transcription JSON must be a flat list of strings "
                            f"(or a name→text object); got a nested list at index {i}"
                        )
                    if not isinstance(item, str):
                        raise SystemExit(
                            f"--rewrite-transcription JSON list item {i} must be a string, "
                            f"got {type(item).__name__}"
                        )
                    texts.append(item)
                return texts
            if isinstance(raw, dict):
                mapping: dict[str, str] = {}
                for key, val in raw.items():
                    if isinstance(val, dict) and "text" in val:
                        mapping[str(key)] = str(val["text"])
                    elif isinstance(val, str):
                        mapping[str(key)] = val
                    else:
                        raise SystemExit(
                            f"--rewrite-transcription JSON entry {key!r} must be a string "
                            f'or {{"text": ...}}, got {type(val).__name__}'
                        )
                return mapping
            raise SystemExit(
                "--rewrite-transcription JSON must be a list of strings or a "
                "segment-name → text object"
            )
    for i, t in enumerate(args_texts):
        if t.startswith("[") and t.endswith("]"):
            raise SystemExit(
                "--rewrite-transcription expects one quoted string per segment "
                f"(or a .json file), not nested arrays; got {t!r} at position {i}"
            )
    return list(args_texts)


def rewrite_transcriptions(
    transcripts_path: Path,
    segment_names: Sequence[str],
    texts: Sequence[str] | dict[str, str],
) -> None:
    """
    Overwrite ``02_transcripts.json`` texts for *segment_names*.

    *texts* is either a list (positional, must match segment count) or a dict
    keyed by segment filename (must cover every segment).
    """
    if not transcripts_path.is_file():
        raise SystemExit(
            f"Cannot rewrite transcriptions: {transcripts_path} does not exist. "
            "Run the transcribe stage first, or omit --resume so transcripts are regenerated."
        )

    if isinstance(texts, dict):
        missing = [n for n in segment_names if n not in texts]
        extra = [k for k in texts if k not in set(segment_names)]
        if missing or extra or len(texts) != len(segment_names):
            raise SystemExit(
                f"--rewrite-transcription dict has {len(texts)} entries but there are "
                f"{len(segment_names)} segment file(s); keys must match segment names.\n"
                f"  segments: {list(segment_names)}\n"
                f"  missing:  {missing}\n"
                f"  extra:    {extra}"
            )
        ordered = [texts[n] for n in segment_names]
    else:
        if len(texts) != len(segment_names):
            raise SystemExit(
                f"--rewrite-transcription has {len(texts)} text(s) but there are "
                f"{len(segment_names)} segment file(s). Counts must match.\n"
                f"  segments: {list(segment_names)}\n"
                f"  texts:    {list(texts)}"
            )
        ordered = list(texts)

    data = json.loads(transcripts_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected transcripts format in {transcripts_path}: expected object")
    for name, text in zip(segment_names, ordered):
        entry = data.get(name)
        if isinstance(entry, dict):
            entry["text"] = text
        else:
            data[name] = {"text": text}
        log.info("Rewrote transcript for %s → %r", name, text)
    transcripts_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_stage_start(
    ws: dict[str, Path],
    manifest: dict,
    *,
    stages: Sequence[str],
    from_stage: Optional[str],
    resume: bool,
) -> str:
    """
    Resolve the effective start stage and drop artifacts that must be regenerated.

    - Explicit ``--from-stage X``: always invalidate X and later (even with
      ``--resume``), so those stages re-run and ``06_stitched.wav`` is rebuilt.
    - ``--resume`` alone: keep all artifacts; ``stage_should_run`` skips done stages.
    - Neither: invalidate from the first stage (full rebuild).
    """
    if from_stage is not None:
        invalidate_from_stage(ws, from_stage, manifest, stages=stages)
        return from_stage
    start = stages[0]
    if not resume:
        invalidate_from_stage(ws, start, manifest, stages=stages)
    return start


def invalidate_from_stage(
    ws: dict[str, Path],
    stage: str,
    manifest: dict,
    *,
    stages: Sequence[str],
) -> None:
    """Drop manifest markers and artifacts for *stage* and everything after it."""
    idx = stages.index(stage)
    for s in stages[idx:]:
        manifest.get("stages", {}).pop(s, None)
    if idx <= stages.index("segment"):
        shutil.rmtree(ws["segments"], ignore_errors=True)
        shutil.rmtree(ws["input"], ignore_errors=True)
    if idx <= stages.index("transcribe"):
        ws["transcripts"].unlink(missing_ok=True)
    if idx <= stages.index("align"):
        ws["aligned"].unlink(missing_ok=True)
        ws["aligned"].with_name("03_aligned_pre_g2p.json").unlink(missing_ok=True)
        ws["aligned"].with_name("03_aligned_per_segment.json").unlink(missing_ok=True)
        ws["aligned"].with_name("03_aligned_per_segment_pre_g2p.json").unlink(missing_ok=True)
    if idx <= stages.index("inpaint"):
        shutil.rmtree(ws["spans"], ignore_errors=True)
        shutil.rmtree(ws["inpainted"], ignore_errors=True)
    if idx <= stages.index("stitch"):
        ws["stitched"].unlink(missing_ok=True)
