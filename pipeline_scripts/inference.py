"""Source-level span collection, inference invocation, and stitching."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import soxr

from pipeline_utils import per_segment_alignments

from .constants import INFER_SCRIPT, OUTPUT_FS, REPO_ROOT, _SEGMENT_RE
from .runtime import run_python
from .spans import build_remapped_phone_intervals, build_spans_for_phone_map, collect_labeled_spans
from .timing import _fmt_elapsed

log = logging.getLogger(__name__)

def _cmd_flag(name: str, value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [name] if value else []
    return [name, str(value)]


def build_infer_command(
    args: argparse.Namespace,
    wav_path: Path,
    spans_path: Path,
    output_path: Path,
) -> List[str]:
    cmd: List[str] = [
        "python",
        str(INFER_SCRIPT),
        "--wav",
        str(wav_path),
        "--spans",
        str(spans_path),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(output_path),
    ]
    cmd.extend(_cmd_flag("--save-resampled", args.save_resampled))
    cmd.extend(_cmd_flag("--device", args.infer_device))
    if args.max_utt_sec != float("inf"):
        cmd.extend(["--max_utt_sec", str(args.max_utt_sec)])
    if args.ola:
        cmd.extend([
            "--ola",
            "--ola_chunk_frames", str(args.ola_chunk_frames),
            "--ola_hop_factor", str(args.ola_hop_factor),
            "--ola_ctx_frames", str(args.ola_ctx_frames),
        ])
    if args.windowed_attention:
        cmd.append("--windowed_attention")
        cmd.extend(["--context_frames", str(args.context_frames)])
    cmd.extend(_cmd_flag("--normalize_vol", args.normalize_vol))
    cmd.extend(_cmd_flag("--encodec-ckpt", args.encodec_ckpt))
    if getattr(args, "full_phone_conditioning", False):
        cmd.append("--full_phone_conditioning")
    if getattr(args, "full_decode", False):
        cmd.append("--full-decode")
    return cmd


def build_batch_infer_command(
    args: argparse.Namespace,
    manifest_path: Path,
) -> List[str]:
    """Like build_infer_command, but for one --batch_manifest covering every
    source file at once (checkpoint/EnCodec loaded a single time)."""
    cmd: List[str] = [
        "python",
        str(INFER_SCRIPT),
        "--batch_manifest",
        str(manifest_path),
        "--checkpoint",
        str(args.checkpoint),
    ]
    cmd.extend(_cmd_flag("--device", args.infer_device))
    if args.max_utt_sec != float("inf"):
        cmd.extend(["--max_utt_sec", str(args.max_utt_sec)])
    if args.ola:
        cmd.extend([
            "--ola",
            "--ola_chunk_frames", str(args.ola_chunk_frames),
            "--ola_hop_factor", str(args.ola_hop_factor),
            "--ola_ctx_frames", str(args.ola_ctx_frames),
        ])
    if args.windowed_attention:
        cmd.append("--windowed_attention")
        cmd.extend(["--context_frames", str(args.context_frames)])
    cmd.extend(_cmd_flag("--normalize_vol", args.normalize_vol))
    cmd.extend(_cmd_flag("--encodec-ckpt", args.encodec_ckpt))
    if getattr(args, "full_phone_conditioning", False):
        cmd.append("--full_phone_conditioning")
    if getattr(args, "full_decode", False):
        cmd.append("--full-decode")
    return cmd



def segment_source_stem(name: str) -> str:
    """Map a segment filename back to its source wav stem (foo_segment_3.wav → foo)."""
    m = _SEGMENT_RE.search(name)
    if m:
        return name[:m.start()].rstrip("_")
    return Path(name).stem


def segment_index(name: str) -> int:
    m = _SEGMENT_RE.search(name)
    return int(m.group(1)) if m else 0


def segments_by_source(segment_names: Sequence[str]) -> Dict[str, List[str]]:
    """Group segment filenames by source stem, ordered by segment index."""
    grouped: Dict[str, List[tuple[int, str]]] = {}
    for name in segment_names:
        stem = segment_source_stem(name)
        grouped.setdefault(stem, []).append((segment_index(name), name))
    return {
        stem: [name for _, name in sorted(items, key=lambda x: x[0])]
        for stem, items in grouped.items()
    }


def source_order_from_segments(segment_names: Sequence[str]) -> List[str]:
    """Source stems in first-seen order (matches original file / segment order)."""
    order: List[str] = []
    seen: set[str] = set()
    for name in segment_names:
        stem = segment_source_stem(name)
        if stem not in seen:
            seen.add(stem)
            order.append(stem)
    return order


def segment_start_offsets(segment_names: Sequence[str], segments_dir: Path) -> Dict[str, float]:
    """Cumulative start time of each segment in the original source timeline."""
    offsets: Dict[str, float] = {}
    t = 0.0
    for name in segment_names:
        offsets[name] = t
        t += sf.info(str(segments_dir / name)).duration
    return offsets


def resolve_source_wav(source_stem: str, input_dir: Path, ws: dict[str, Path]) -> Path:
    for root in (input_dir, ws["input"]):
        path = root / f"{source_stem}.wav"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Source wav not found for {source_stem!r} under {input_dir} or {ws['input']}"
    )


def _shift_spans(spans: list, offset_sec: float) -> list:
    """Add offset_sec to every span start/end field (seconds)."""
    shifted: list = []
    for sp in spans:
        if isinstance(sp, dict):
            out = dict(sp)
            for key in ("start", "end", "t_start", "t_end"):
                if key in out and out[key] is not None:
                    out[key] = float(out[key]) + offset_sec
            shifted.append(out)
        elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
            shifted.append([float(sp[0]) + offset_sec, float(sp[1]) + offset_sec, *sp[2:]])
        else:
            shifted.append(sp)
    return shifted


def passthrough_wav(src: Path, dst: Path, *, out_fs: int = OUTPUT_FS) -> None:
    wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != out_fs:
        wav = soxr.resample(wav, sr, out_fs, quality="VHQ").astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav, out_fs)


def collect_inpaint_jobs(
    ws: dict[str, Path],
    *,
    infer_args: argparse.Namespace,
    segment_names: Sequence[str],
    input_dir: Path,
    out_fs: int = OUTPUT_FS,
) -> Tuple[List[str], List[dict], List[int]]:
    """Build spans JSONs and return infer-script jobs without running inference.

    Returns ``(source_order, batch_jobs, span_counts)``. Missing segment
    alignments or zero matching spans abort with ``SystemExit`` (no silent
    passthrough).
    """
    aligned = json.loads(ws["aligned"].read_text(encoding="utf-8"))
    per_segment = per_segment_alignments(aligned)
    # Prefer the dedicated per-segment file when present (survives writers that
    # only keep merged source keys in 03_aligned.json).
    per_seg_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    if per_seg_path.is_file():
        try:
            extra = json.loads(per_seg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(
                f"ERROR: could not read {per_seg_path}: {exc}"
            ) from exc
        if isinstance(extra, dict):
            for key, val in extra.items():
                if _SEGMENT_RE.search(key) and isinstance(val, dict):
                    per_segment[key] = val
    ws["spans"].mkdir(parents=True, exist_ok=True)
    ws["inpainted"].mkdir(parents=True, exist_ok=True)

    phone_map = ", ".join(
        f"{pi!r}->{po!r}" for pi, po in zip(infer_args.phones_in, infer_args.phones_out)
    )
    grouped = segments_by_source(segment_names)
    source_order = source_order_from_segments(segment_names)
    batch_jobs: List[dict] = []
    batch_spans_counts: List[int] = []

    for source_stem in source_order:
        seg_names = grouped[source_stem]
        offsets = segment_start_offsets(seg_names, ws["segments"])
        source_wav = resolve_source_wav(source_stem, input_dir, ws)
        out_path = ws["inpainted"] / f"{source_stem}.wav"
        source_dur = sf.info(str(source_wav)).duration

        all_spans: list = []
        all_phone_intervals: list = []
        for name in seg_names:
            chunk_path = ws["segments"] / name
            if not chunk_path.is_file():
                raise FileNotFoundError(f"Missing segment wav: {chunk_path}")

            entry = per_segment.get(name)
            if entry is None:
                raise SystemExit(
                    f"ERROR: No alignment for segment {name} (source {source_stem}) "
                    f"in {ws['aligned']} (and {per_seg_path.name if per_seg_path.is_file() else 'no per-segment file'}). "
                    f"Present keys: {sorted(aligned)}. Refusing silent passthrough — "
                    f"re-run align or restore work/{source_stem}/03_aligned*.json."
                )

            try:
                labeled = collect_labeled_spans(
                    entry.get("phones", {}), wanted=infer_args.phones_in
                )
                chunk_dur = sf.info(str(chunk_path)).duration
                spans = build_spans_for_phone_map(
                    labeled,
                    infer_args.phones_in,
                    infer_args.phones_out,
                    ext_sec=infer_args.ext,
                    max_end=chunk_dur,
                    force_ext=getattr(infer_args, "force_ext", True),
                )
            except ValueError:
                continue

            all_spans.extend(_shift_spans(spans, offsets[name]))
            if getattr(infer_args, "full_phone_conditioning", False):
                intervals = build_remapped_phone_intervals(
                    entry.get("phones", {}),
                    infer_args.phones_in,
                    infer_args.phones_out,
                )
                all_phone_intervals.extend(_shift_spans(intervals, offsets[name]))

        if not all_spans:
            raise SystemExit(
                f"ERROR: No spans for source {source_stem} ({phone_map}); "
                f"refusing silent passthrough. Check --phones_in against the "
                f"alignment phones in {ws['aligned']} / {per_seg_path.name}."
            )

        spans_payload: dict = {"spans": all_spans, "source": source_stem}
        if getattr(infer_args, "full_phone_conditioning", False):
            if not all_phone_intervals:
                raise SystemExit(
                    f"--full_phone_conditioning set but no phone_intervals for {source_stem}"
                )
            spans_payload["phone_intervals"] = all_phone_intervals

        spans_path = ws["spans"] / f"{source_stem}.json"
        spans_path.write_text(
            json.dumps(spans_payload, indent=2),
            encoding="utf-8",
        )

        log.info(
            "Queued %d span(s) in %s (from %d segment(s), %.2f s, %s)",
            len(all_spans), source_stem, len(seg_names), source_dur, phone_map,
        )
        job: dict = {
            "wav": str(source_wav),
            "spans": str(spans_path),
            "output": str(out_path),
        }
        if infer_args.save_resampled is not None:
            job["save_resampled"] = str(infer_args.save_resampled)
        batch_jobs.append(job)
        batch_spans_counts.append(len(all_spans))

    return source_order, batch_jobs, batch_spans_counts


def execute_infer_batch(
    *,
    infer_args: argparse.Namespace,
    batch_jobs: Sequence[dict],
    manifest_path: Path,
) -> None:
    """Write *batch_jobs* to *manifest_path* and run one infer subprocess."""
    if not batch_jobs:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": list(batch_jobs)}, indent=2),
        encoding="utf-8",
    )
    log.info(
        "Inpainting %d source file(s) in one batch — checkpoint + EnCodec loaded once",
        len(batch_jobs),
    )
    cmd = build_batch_infer_command(infer_args, manifest_path)
    t0 = time.perf_counter()
    run_python(cmd, cwd=REPO_ROOT)
    dt = time.perf_counter() - t0
    log.info(
        "[timing] inpaint total — %d source file(s), %s (avg %s/file)",
        len(batch_jobs),
        _fmt_elapsed(dt),
        _fmt_elapsed(dt / len(batch_jobs)),
    )


def run_inpaint(
    ws: dict[str, Path],
    *,
    infer_args: argparse.Namespace,
    segment_names: Sequence[str],
    input_dir: Path,
    out_fs: int = OUTPUT_FS,
) -> List[str]:
    source_order, batch_jobs, batch_spans_counts = collect_inpaint_jobs(
        ws,
        infer_args=infer_args,
        segment_names=segment_names,
        input_dir=input_dir,
        out_fs=out_fs,
    )
    if batch_jobs:
        phone_map = ", ".join(
            f"{pi!r}->{po!r}"
            for pi, po in zip(infer_args.phones_in, infer_args.phones_out)
        )
        log.info(
            "Inpainting %d source file(s) (%d span(s) total, %s)",
            len(batch_jobs), sum(batch_spans_counts), phone_map,
        )
        execute_infer_batch(
            infer_args=infer_args,
            batch_jobs=batch_jobs,
            manifest_path=ws["spans"] / "_batch_manifest.json",
        )
    return source_order


def stitch_wavs(
    paths: Sequence[Path],
    output_path: Path,
    *,
    crossfade_ms: float = 0.0,
    out_fs: int = OUTPUT_FS,
) -> None:
    if not paths:
        raise ValueError("No wav files to stitch")

    chunks: List[np.ndarray] = []
    for p in paths:
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != out_fs:
            wav = soxr.resample(wav, sr, out_fs, quality="VHQ").astype(np.float32)
        chunks.append(wav.astype(np.float32))

    if crossfade_ms <= 0 or len(chunks) == 1:
        out = np.concatenate(chunks)
    else:
        fade = max(1, int(out_fs * crossfade_ms / 1000.0))
        out = chunks[0]
        for nxt in chunks[1:]:
            if len(out) >= fade and len(nxt) >= fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                blended = out[-fade:] * (1.0 - ramp) + nxt[:fade] * ramp
                out = np.concatenate([out[:-fade], blended, nxt[fade:]])
            else:
                out = np.concatenate([out, nxt])

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1.0:
        out = out / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), out, out_fs)
    log.info("Wrote stitched output (%.2f s): %s", len(out) / out_fs, output_path)

