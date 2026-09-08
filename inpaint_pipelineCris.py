#!/usr/bin/env python3
"""
Production pipeline with CrisperWhisper sidecar ASR:
  segment → Crisper sidecar → MFA align → phone inpaint → stitch.

This entrypoint does **not** use conda hops. Point ``--crisper-cmd`` (or
``PHONEPAINT_CRISPER_CMD``) at a venv/Docker wrapper that speaks the JSON
transcript contract (``{segment.wav: {"text": "..."}}``).

Aligner is MFA only (no BFA/FALCON/MAPS). For Whisper ASR use inpaint_pipeline.py.
Legacy multi-backend CLI remains in run_inpaint_pipeline.py (untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import soxr

from phone_labels import (
    canonical_phone_label,
    canonical_user_phone,
)
from pipeline_utils import (
    deploy_stitched_output,
    invalidate_from_stage,
    load_rewrite_texts,
    all_inputs_shorter_than,
    copy_as_single_segments,
    per_segment_alignments,
    pick_asr_device_idx,
    prepare_stage_start,
    rewrite_transcriptions,
    write_aligned_json,
)
from unified_speech import segment_wav_directory
from voicecraft_encodec import DEFAULT_CHECKPOINT as ENCODEC_16K_CKPT

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
MFA_ALIGN_SCRIPT = REPO_ROOT / "tools" / "mfa" / "mfa_align_json.py"
# Reference CLI used by the legacy conda hop; production uses --crisper-cmd.
CRISPER_REF_SCRIPT = REPO_ROOT / "tools" / "crisper" / "crisper_for_mfa_w_batching.py"
INFER_SCRIPT_DEFAULT = REPO_ROOT / "infer_phone_ec.py"
INFER_SCRIPT_COND = REPO_ROOT / "infer_phone_ec_cond.py"

# Skip silence splitting below this duration (seconds).
MIN_SPLIT_SEC = 6.0
DEFAULT_CRISPER_MODEL = "nyrahealth/CrisperWhisper"

# Defaults match infer_phone_ec_cond (16k @ 50 fps) / infer_phone_ec (24k @ 75 fps):
# ±0.5 s of context on each side of the span for both rates.
CONTEXT_FRAMES_16K = 50   # ±25 frames @ 50 fps
CONTEXT_FRAMES_24K = 75   # ±37 frames @ 75 fps

_ENCODEC_FS = {"24k": 24_000, "16k": 16_000}
OUTPUT_FS = 24_000  # overridden at runtime by --encodec
_SEGMENT_RE = re.compile(r"_segment_(\d+)\.wav$", re.IGNORECASE)

STAGES = ("segment", "transcribe", "align", "inpaint", "stitch")

# Per-stage wall times (seconds) and peak VRAM for the current pipeline job.
# Written to work_dir/stage_stats.json via write_stage_stats().
_STAGE_TIMING: Dict[str, float] = {}
_VRAM_PEAK_MB: float = 0.0

# Alignment labels that count as a preceding /t/ for --ext suppression.
_T_ALIGN_LABELS = frozenset({canonical_user_phone("t")})
_SHORT_SZ_PHONES = frozenset({"S", "Z"})


def reset_stage_stats() -> None:
    global _STAGE_TIMING, _VRAM_PEAK_MB
    _STAGE_TIMING = {}
    _VRAM_PEAK_MB = 0.0


def load_stage_stats_state(timing: Dict[str, float], vram_mb: float) -> None:
    global _STAGE_TIMING, _VRAM_PEAK_MB
    _STAGE_TIMING = dict(timing)
    _VRAM_PEAK_MB = float(vram_mb)


def _cuda_peak_mb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
    except Exception:
        return None


def _note_vram_peak() -> None:
    global _VRAM_PEAK_MB
    peak = _cuda_peak_mb()
    if peak is not None:
        _VRAM_PEAK_MB = max(_VRAM_PEAK_MB, peak)


def write_stage_stats(work_dir: Path, *, stem: str) -> Path:
    """Persist stage timings / peak VRAM under ``work_dir/stage_stats.json``.

    A passthrough inpaint (no infer jobs) records ``inpaint`` as 0. Keep a
    previously measured positive inpaint wall so demo stats still have a value.
    """
    global _VRAM_PEAK_MB
    path = work_dir / "stage_stats.json"
    inpaint = float(_STAGE_TIMING.get("inpaint", 0.0))
    vram = float(_VRAM_PEAK_MB)
    if path.is_file() and (inpaint <= 0 or vram <= 0):
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = {}
        prev_stages = prev.get("stages") or {}
        prev_inpaint = float(prev_stages.get("inpaint") or 0.0)
        if inpaint <= 0 and prev_inpaint > 0:
            _STAGE_TIMING["inpaint"] = prev_inpaint
            inpaint = prev_inpaint
        prev_vram = prev.get("vram_mb_peak")
        if vram <= 0 and prev_vram is not None:
            try:
                prev_vram_f = float(prev_vram)
            except (TypeError, ValueError):
                prev_vram_f = 0.0
            if prev_vram_f > 0:
                _VRAM_PEAK_MB = prev_vram_f
                vram = prev_vram_f
    rt = (
        float(_STAGE_TIMING.get("transcribe", 0.0))
        + float(_STAGE_TIMING.get("align", 0.0))
        + inpaint
    )
    payload = {
        "stem": stem,
        "stages": {k: round(v, 3) for k, v in sorted(_STAGE_TIMING.items())},
        "rt_sec": round(rt, 3),
        "vram_mb_peak": round(vram, 1) if vram > 0 else None,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("[stats] wrote %s (rt=%.3fs excl. segment/stitch)", path, rt)
    return path


def _cond_out_label(phone_out: str, source_text: str) -> str:
    """Map user phones_out to the unified canonical inventory."""
    del source_text
    return canonical_user_phone(phone_out)


# ── Phone-map / span helpers (self-contained; no infer_phone_ec_lip) ──────────

def _iter_phones(phones) -> Iterable[dict]:
    if isinstance(phones, dict):
        return phones.values()
    if isinstance(phones, list):
        return phones
    raise TypeError(f"Expected phones dict or list, got {type(phones).__name__}")


def _extend_short_sz(start: float, end: float, phone: str) -> Tuple[float, float]:
    """Widen very short s/z spans."""
    if phone not in _SHORT_SZ_PHONES:
        return start, end
    dur = round(end - start, 3)
    if dur == 0.03:
        return start - 0.01, end + 0.01
    if dur == 0.04:
        return start - 0.01, end
    return start, end


def _prepare_span(
    start: float,
    end: float,
    phone: str,
    *,
    follows_t: bool,
    ext_sec: float,
    max_end: Optional[float],
    force_ext: bool = True,
) -> Optional[Tuple[float, float]]:
    start, end = _extend_short_sz(start, end, phone)
    use_ext = ext_sec if (force_ext or not follows_t) else 0.0
    if use_ext > 0:
        start -= use_ext
        end += use_ext
    start = max(0.0, start)
    if max_end is not None:
        end = min(max_end, end)
    if end <= start:
        return None
    return start, end


def collect_labeled_spans(
    phones, *, wanted: Sequence[str]
) -> List[Tuple[float, float, str, bool]]:
    """Return (start, end, align_text, follows_t) for wanted phones.

    ``follows_t`` is true when the immediately preceding alignment phone is a
    /t/ (after inventory normalization). Those spans get ``--ext`` forced
    to 0 so extension does not bleed into the /t/, unless ``--force_ext``.
    """
    wanted_labels = {canonical_user_phone(p) for p in wanted}
    ordered: List[Tuple[float, float, str]] = []
    for phone in _iter_phones(phones):
        if not isinstance(phone, dict):
            continue
        text = phone.get("text")
        if text is None:
            continue
        try:
            start = float(phone["xmin"])
            end = float(phone["xmax"])
        except KeyError as exc:
            raise ValueError(f"Phone entry missing xmin/xmax: {phone!r}") from exc
        if end <= start:
            raise ValueError(f"Phone span end must be > start: ({start}, {end}) for {phone!r}")
        canonical = str(phone.get("canonical") or canonical_phone_label(str(text)))
        ordered.append((start, end, canonical))
    ordered.sort(key=lambda pair: pair[0])

    spans: List[Tuple[float, float, str, bool]] = []
    for i, (start, end, text) in enumerate(ordered):
        if text not in wanted_labels:
            continue
        follows_t = i > 0 and ordered[i - 1][2] in _T_ALIGN_LABELS
        spans.append((start, end, text, follows_t))
    if not spans:
        labels = " or ".join(sorted(wanted_labels))
        raise ValueError(f"No {labels!r} phones found in alignment")
    return spans


def build_spans_for_phone_map(
    labeled: Sequence[Tuple[float, float, str, bool]],
    phones_in: Sequence[str],
    phones_out: Sequence[str],
    *,
    ext_sec: float = 0.0,
    max_end: Optional[float] = None,
    force_ext: bool = True,
) -> List[dict]:
    in_to_out = {
        canonical_user_phone(phone_in): phone_out
        for phone_in, phone_out in zip(phones_in, phones_out)
    }
    spans: List[dict] = []
    skipped = 0
    n_no_ext = 0
    for start, end, align_text, follows_t in labeled:
        phone_out = in_to_out.get(align_text)
        if phone_out is None:
            continue
        if max_end is not None and start >= max_end:
            skipped += 1
            continue
        if follows_t and ext_sec > 0 and not force_ext:
            n_no_ext += 1
        prepared = _prepare_span(
            start, end, align_text,
            follows_t=follows_t, ext_sec=ext_sec, max_end=max_end,
            force_ext=force_ext,
        )
        if prepared is None:
            skipped += 1
            continue
        s, e = prepared
        spans.append({
            "start": s,
            "end": e,
            "phone": phone_out,
            "follows_t": follows_t,
        })
    if skipped:
        log.warning(
            "Skipped %d span(s) with timestamps outside the audio range", skipped,
        )
    if n_no_ext:
        log.info(
            "--ext %gs skipped for %d span(s) immediately following /t/ "
            "(use --force_ext to override)",
            ext_sec, n_no_ext,
        )
    spans.sort(key=lambda item: item["start"])
    if not spans:
        raise ValueError("No spans found for any entry in --phones_in")
    return spans


def build_remapped_phone_intervals(
    phones,
    phones_in: Sequence[str],
    phones_out: Sequence[str],
) -> List[dict]:
    """Full-utterance phone intervals with phones_in→phones_out label remapping."""
    in_phones = {
        canonical_user_phone(phone_in): phone_out
        for phone_in, phone_out in zip(phones_in, phones_out)
    }

    intervals: List[dict] = []
    for phone in _iter_phones(phones):
        if not isinstance(phone, dict):
            continue
        text = phone.get("text")
        if text is None:
            continue
        try:
            start = float(phone["xmin"])
            end = float(phone["xmax"])
        except KeyError as exc:
            raise ValueError(f"Phone entry missing xmin/xmax: {phone!r}") from exc
        if end <= start:
            continue
        src = str(phone.get("canonical") or canonical_phone_label(str(text)))
        phone_out = in_phones.get(src)
        label = _cond_out_label(phone_out, src) if phone_out is not None else src
        intervals.append({"start": start, "end": end, "text": label})
    intervals.sort(key=lambda item: item["start"])
    return intervals


def validate_phone_map(phones_in: Sequence[str], phones_out: Sequence[str]) -> None:
    if not phones_in:
        raise SystemExit("--phones_in must not be empty")
    if len(phones_in) != len(phones_out):
        raise SystemExit(
            f"--phones_in ({len(phones_in)}) and --phones_out ({len(phones_out)}) "
            "must have equal length"
        )


def add_phone_map_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--phones_in",
        nargs="+",
        required=required,
        metavar="PHONE",
        help="Aligned phone(s) to replace (user-facing; aliases applied) "
             "(ignored / not required with --batch_manifest)",
    )
    parser.add_argument(
        "--phones_out",
        nargs="+",
        required=required,
        metavar="PHONE",
        help="Target phone for each corresponding --phones_in entry "
             "(ignored / not required with --batch_manifest)",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_elapsed(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.2f} s"


def _stage_key_from_label(label: str) -> Optional[str]:
    """Map ``stage transcribe (stem)`` / ``stage inpaint (shared batch)`` → stage name."""
    parts = label.strip().split()
    if len(parts) >= 2 and parts[0] == "stage" and parts[1] in STAGES:
        return parts[1]
    return None


@contextmanager
def wallclock(label: str):
    """Log wall-clock time for a block; always prints even on failure.

    When *label* starts with ``stage <name>``, records seconds into
    ``_STAGE_TIMING`` and updates peak CUDA memory for stats.json.
    """
    t0 = time.perf_counter()
    stage = _stage_key_from_label(label)
    try:
        import torch
        if stage and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    log.info("[timing] %s — started", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.info("[timing] %s — %s", label, _fmt_elapsed(elapsed))
        if stage is not None:
            _STAGE_TIMING[stage] = _STAGE_TIMING.get(stage, 0.0) + elapsed
            _note_vram_peak()


def run_python(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    """Run a stage with this environment's interpreter."""
    cmd = list(cmd)
    if cmd and cmd[0] in {"python", Path(sys.executable).name}:
        cmd[0] = sys.executable
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_crisper_cmd(explicit: Optional[str]) -> List[str]:
    """Resolve the CrisperWhisper sidecar argv prefix (no conda hop).

    Preference order:
      1. ``--crisper-cmd`` / *explicit*
      2. ``PHONEPAINT_CRISPER_CMD`` (shell-split)
      3. ``sys.executable CRISPER_REF_SCRIPT`` when the reference script exists
         (local venv that already has the nyrahealth transformers fork)
    """
    import shlex

    raw = (explicit or os.environ.get("PHONEPAINT_CRISPER_CMD") or "").strip()
    if raw:
        return shlex.split(raw)
    if CRISPER_REF_SCRIPT.is_file():
        return [sys.executable, str(CRISPER_REF_SCRIPT)]
    raise SystemExit(
        "CrisperWhisper sidecar not configured. Set --crisper-cmd or "
        "PHONEPAINT_CRISPER_CMD to a wrapper that accepts the same flags as "
        f"{CRISPER_REF_SCRIPT.name} (see run_inpaint_pipeline_prod.md)."
    )


def run_crisper_sidecar(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    """Invoke the Crisper ASR sidecar as a subprocess (venv/Docker/wrapper)."""
    cmd = list(cmd)
    log.info("Running Crisper sidecar: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def active_conda_env_name() -> Optional[str]:
    """Return the active conda env name, or None if not in a conda env."""
    name = (os.environ.get("CONDA_DEFAULT_ENV") or "").strip()
    if name:
        return name
    prefix = (os.environ.get("CONDA_PREFIX") or "").strip()
    if prefix:
        return Path(prefix).name
    return None


def is_phonepaint_mfa_env() -> bool:
    """True when the active interpreter is the legacy in-env MFA+MAPS stack."""
    return active_conda_env_name() == "PhonePaintMFA"


def resolve_mfa_bin(env_mfa: str) -> str:
    """Absolute path to the MFA CLI (subprocess; not imported in-process).

    Prefers MFA in the active conda env. The shipped ``PhonePaint`` env has no
    in-env MFA, so it falls back to ``env_mfa`` (default ``mfa_env``), the
    sidecar built by ``./create_mfa_envs.sh``. Legacy ``PhonePaintMFA`` keeps
    MFA in-env and never hops to the sidecar.
    """
    explicit = os.environ.get("MFA_BIN")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        raise SystemExit(f"MFA_BIN is set but not a file: {explicit}")

    candidates: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            which_path = Path(which).resolve()
            prefix_path = Path(conda_prefix).resolve()
            try:
                which_path.relative_to(prefix_path)
                candidates.append(which_path)
            except ValueError:
                pass

    allow_sidecar = not is_phonepaint_mfa_env()
    if allow_sidecar:
        if conda_prefix:
            candidates.append(Path(conda_prefix).parent / env_mfa / "bin" / "mfa")
        home = Path.home()
        for base in (home / "miniconda3", home / "anaconda3", home / "mambaforge", home / "miniforge3"):
            candidates.append(base / "envs" / env_mfa / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            candidates.append(Path(which))
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return str(cand)
    if is_phonepaint_mfa_env():
        env = active_conda_env_name() or "PhonePaint"
        raise SystemExit(
            f"{env} is active but MFA CLI was not found in this env "
            f"($CONDA_PREFIX/bin/mfa). Install MFA in-env or set MFA_BIN; "
            f"the mfa_env sidecar is not used while {env} is active."
        )
    raise SystemExit(
        f"MFA CLI not found in the active env or fallback {env_mfa!r}. "
        "Install MFA in-env, run ./create_mfa_envs.sh, or set MFA_BIN."
    )


def workspace_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "root": work_dir,
        "input": work_dir / "00_input",
        "segments": work_dir / "01_segments",
        "transcripts": work_dir / "02_transcripts.json",
        "aligned": work_dir / "03_aligned.json",
        "spans": work_dir / "04_spans",
        "inpainted": work_dir / "05_inpainted",
        "stitched": work_dir / "06_stitched.wav",
        "manifest": work_dir / "manifest.json",
    }


def load_manifest(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def mark_stage(manifest: dict, stage: str) -> None:
    manifest.setdefault("stages", {})[stage] = {"done": True, "at": _utc_now()}


def stage_done(manifest: dict, stage: str) -> bool:
    return bool(manifest.get("stages", {}).get(stage, {}).get("done"))


def sorted_segment_wavs(segments_dir: Path) -> List[Path]:
    wavs = list(segments_dir.glob("*.wav"))
    if not wavs:
        return []

    def _key(p: Path) -> Tuple[int, str]:
        m = _SEGMENT_RE.search(p.name)
        return (int(m.group(1)) if m else 10**9, p.name)

    return sorted(wavs, key=_key)


def _cmd_flag(name: str, value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [name] if value else []
    return [name, str(value)]


def _resolve_infer_script(args: argparse.Namespace) -> Path:
    if hasattr(args, "infer_script") and args.infer_script is not None:
        return args.infer_script
    return INFER_SCRIPT_COND if args.encodec == "16k" else INFER_SCRIPT_DEFAULT


def build_infer_command(
    args: argparse.Namespace,
    wav_path: Path,
    spans_path: Path,
    output_path: Path,
) -> List[str]:
    infer_script = _resolve_infer_script(args)
    is_cond = infer_script.name == INFER_SCRIPT_COND.name
    cmd: List[str] = [
        "python",
        str(infer_script),
        "--wav",
        str(wav_path),
        "--spans",
        str(spans_path),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(output_path),
    ]
    if not is_cond:
        cmd.extend(["--phone", str(args.phones_out[0])])
    cmd.extend(_cmd_flag("--save-resampled", args.save_resampled))
    cmd.extend(_cmd_flag("--device", args.infer_device))
    if not is_cond:
        cmd.extend(_cmd_flag("--attn_heads", args.attn_heads))
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
    if is_cond:
        cmd.extend(_cmd_flag("--encodec-ckpt", args.encodec_ckpt))
        if getattr(args, "full_phone_conditioning", False):
            cmd.append("--full_phone_conditioning")
        if getattr(args, "full_decode", False):
            cmd.append("--full-decode")
    return cmd


def build_batch_infer_command(
    args: argparse.Namespace,
    manifest_path: Path,
    *,
    phone: Optional[str] = None,
) -> List[str]:
    """Like build_infer_command, but for one --batch_manifest covering every
    source file at once (checkpoint/EnCodec loaded a single time)."""
    infer_script = _resolve_infer_script(args)
    is_cond = infer_script.name == INFER_SCRIPT_COND.name
    cmd: List[str] = [
        "python",
        str(infer_script),
        "--batch_manifest",
        str(manifest_path),
        "--checkpoint",
        str(args.checkpoint),
    ]
    if not is_cond:
        phone_label = phone if phone is not None else str(args.phones_out[0])
        cmd.extend(["--phone", phone_label])
    cmd.extend(_cmd_flag("--device", args.infer_device))
    if not is_cond:
        cmd.extend(_cmd_flag("--attn_heads", args.attn_heads))
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
    if is_cond:
        cmd.extend(_cmd_flag("--encodec-ckpt", args.encodec_ckpt))
        if getattr(args, "full_phone_conditioning", False):
            cmd.append("--full_phone_conditioning")
        if getattr(args, "full_decode", False):
            cmd.append("--full-decode")
    return cmd


def prepare_input(input_path: Path, ws: dict[str, Path]) -> Path:
    input_path = input_path.resolve()
    if input_path.is_dir():
        return input_path
    ws["input"].mkdir(parents=True, exist_ok=True)
    dst = ws["input"] / input_path.name
    if not dst.exists() or dst.stat().st_mtime < input_path.stat().st_mtime:
        shutil.copy2(input_path, dst)
    return ws["input"]


def run_segment(
    input_dir: Path,
    ws: dict[str, Path],
    *,
    silence_level: float,
) -> List[str]:
    """Applio-identical silence split in-process."""
    ws["segments"].mkdir(parents=True, exist_ok=True)
    if all_inputs_shorter_than(input_dir, MIN_SPLIT_SEC):
        log.info(
            "All inputs shorter than %.1fs — skipping silence-based split",
            MIN_SPLIT_SEC,
        )
        return copy_as_single_segments(input_dir, ws["segments"])
    return segment_wav_directory(
        input_dir,
        ws["segments"],
        silence_level=silence_level,
        min_split_sec=MIN_SPLIT_SEC,
    )



def run_transcribe_crisper(
    ws: dict[str, Path],
    *,
    crisper_cmd: Optional[str],
    model_name: str,
    batch_size: int,
    language: str,
    device: str,
    device_idx: Optional[int] = None,
) -> None:
    """CrisperWhisper via configured sidecar (venv/Docker wrapper; no conda hop)."""
    prefix = resolve_crisper_cmd(crisper_cmd)
    cmd = [
        *prefix,
        "--input_dir",
        str(ws["segments"]),
        "--output_json",
        str(ws["transcripts"]),
        "--model",
        model_name,
        "--language",
        language,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
    ]
    if device.startswith("cuda"):
        idx = pick_asr_device_idx(device_idx)
        cmd.extend(["--device_idx", str(idx)])
    if batch_size > 1:
        cmd.append("--use_optimized")
    run_crisper_sidecar(cmd)


def run_transcribe(ws: dict[str, Path], args: argparse.Namespace) -> None:
    run_transcribe_crisper(
        ws,
        crisper_cmd=args.crisper_cmd,
        model_name=args.asr_model,
        batch_size=args.batch_size,
        language=args.language,
        device=args.asr_device,
        device_idx=args.asr_device_idx,
    )



@dataclass
class MfaAlignJob:
    """One MFA unit of work (usually one ``work/<stem>/`` tree)."""

    stem: str
    ws: dict[str, Path]
    segment_names: List[str]


def _transcript_text(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("text") or "").strip()
    return str(entry or "").strip()


def _mfa_align_env(env_mfa: str, *, threads_per_job: int) -> dict[str, str]:
    mfa_bin = resolve_mfa_bin(env_mfa)
    env = os.environ.copy()
    env["MFA_BIN"] = mfa_bin
    env["MFA_CONDA_ENV"] = env_mfa
    if threads_per_job <= 1:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "1"
    return env


def _mfa_align_cmd(
    *,
    audio_dir: Path,
    input_json: Path,
    output_json: Path,
    threads: int,
) -> List[str]:
    return [
        sys.executable,
        str(MFA_ALIGN_SCRIPT),
        "--audio_dir",
        str(audio_dir),
        "--input_json",
        str(input_json),
        "--output_json",
        str(output_json),
        # english_mfa only; G2P runs only for files that still have fail/OOV words
        # after the primary dictionary pass (no english_us_arpa fallback pair).
        "--use_g2p_fallback",
        "--dump_pre_g2p",
        "--mfa_model_pairs",
        "english_mfa:english_mfa",
        "-j",
        str(max(1, int(threads))),
    ]


def _finalize_mfa_workspace(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    per_segment: dict,
    *,
    pre_g2p: Optional[dict] = None,
) -> None:
    """Write ``03_aligned*.json`` contracts from a per-segment MFA dict."""
    per_segment_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    per_segment_path.parent.mkdir(parents=True, exist_ok=True)
    per_segment_path.write_text(
        json.dumps(per_segment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing = [name for name in segment_names if name not in per_segment]
    if missing:
        raise RuntimeError(f"MFA alignment omitted segments: {missing}")
    write_aligned_json(per_segment, segment_names, ws["segments"], ws["aligned"])
    if pre_g2p is not None:
        pre_path = per_segment_path.with_name("03_aligned_per_segment_pre_g2p.json")
        pre_path.write_text(
            json.dumps(pre_g2p, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_aligned_json(
            pre_g2p,
            segment_names,
            ws["segments"],
            ws["aligned"].with_name("03_aligned_pre_g2p.json"),
        )


def run_align_mfa(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    *,
    env_mfa: str,
    threads: int,
) -> None:
    """MFA JSON wrapper in-process; Kaldi work via MFA CLI (same env or fallback)."""
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")
    mfa_bin = resolve_mfa_bin(env_mfa)
    per_segment_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    env = _mfa_align_env(env_mfa, threads_per_job=threads)
    cmd = _mfa_align_cmd(
        audio_dir=ws["segments"],
        input_json=ws["transcripts"],
        output_json=per_segment_path,
        threads=threads,
    )
    log.info("Running MFA CLI (%s): %s", mfa_bin, " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=str(MFA_ALIGN_SCRIPT.parent),
        check=True,
        env=env,
    )
    per_segment = json.loads(per_segment_path.read_text(encoding="utf-8"))
    pre_g2p_src = per_segment_path.with_name("03_aligned_per_segment_pre_g2p.json")
    pre = (
        json.loads(pre_g2p_src.read_text(encoding="utf-8"))
        if pre_g2p_src.is_file()
        else None
    )
    _finalize_mfa_workspace(ws, segment_names, per_segment, pre_g2p=pre)


def run_align_mfa_batch_merged(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    threads: int,
) -> None:
    """Strategy 1: one ``mfa align`` over a merged corpus, then scatter JSON.

    Amortizes MFA cold-start across many short PhonePaint workspaces. Unique
    utterance keys encode ``stem`` + segment name so results can be remapped
    back to each ``03_aligned.json`` contract.
    """
    if not jobs:
        return
    if len(jobs) == 1:
        job = jobs[0]
        run_align_mfa(
            job.ws, job.segment_names, env_mfa=env_mfa, threads=threads,
        )
        return
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")

    # key → (stem, original segment wav name)
    key_map: Dict[str, Tuple[str, str]] = {}
    merged_manifest: Dict[str, dict] = {}
    for job in jobs:
        transcripts = json.loads(job.ws["transcripts"].read_text(encoding="utf-8"))
        if not isinstance(transcripts, dict):
            raise RuntimeError(
                f"Expected dict transcripts for stem={job.stem}; got {type(transcripts)}"
            )
        for name in job.segment_names:
            audio_path = job.ws["segments"] / name
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing segment wav: {audio_path}")
            text = _transcript_text(transcripts.get(name))
            if not text:
                raise RuntimeError(
                    f"Empty transcript for stem={job.stem} segment={name}"
                )
            # Avoid path separators / collisions across stems.
            key = f"{job.stem}__{name}"
            if key in key_map:
                raise RuntimeError(f"Duplicate MFA batch key: {key}")
            key_map[key] = (job.stem, name)
            merged_manifest[key] = {"audio": str(audio_path.resolve()), "text": text}

    mfa_bin = resolve_mfa_bin(env_mfa)
    with tempfile.TemporaryDirectory(prefix="mfa_batch_merged_") as tmp:
        scratch = Path(tmp)
        audio_dir = scratch / "audio"
        audio_dir.mkdir()
        input_json = scratch / "merged_transcripts.json"
        output_json = scratch / "merged_aligned.json"
        input_json.write_text(
            json.dumps(merged_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        env = _mfa_align_env(env_mfa, threads_per_job=threads)
        cmd = _mfa_align_cmd(
            audio_dir=audio_dir,
            input_json=input_json,
            output_json=output_json,
            threads=threads,
        )
        log.info(
            "Running merged MFA for %d workspace(s) (%s): %s",
            len(jobs), mfa_bin, " ".join(cmd),
        )
        subprocess.run(
            cmd,
            cwd=str(MFA_ALIGN_SCRIPT.parent),
            check=True,
            env=env,
        )
        aligned_all = json.loads(output_json.read_text(encoding="utf-8"))
        pre_path = output_json.with_name(output_json.stem + "_pre_g2p.json")
        pre_all = (
            json.loads(pre_path.read_text(encoding="utf-8"))
            if pre_path.is_file()
            else None
        )

    by_stem: Dict[str, dict] = {job.stem: {} for job in jobs}
    pre_by_stem: Dict[str, dict] = {job.stem: {} for job in jobs}
    for key, payload in aligned_all.items():
        mapped = key_map.get(key)
        if mapped is None:
            continue
        stem, name = mapped
        by_stem[stem][name] = payload
    if pre_all is not None:
        for key, payload in pre_all.items():
            mapped = key_map.get(key)
            if mapped is None:
                continue
            stem, name = mapped
            pre_by_stem[stem][name] = payload

    for job in jobs:
        pre = pre_by_stem.get(job.stem) or None
        if pre is not None and not pre:
            pre = None
        _finalize_mfa_workspace(
            job.ws, job.segment_names, by_stem[job.stem], pre_g2p=pre,
        )


def run_align_mfa_jobs_parallel(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    max_parallel: int,
    threads_per_job: int = 1,
    poll_interval: float = 0.05,
) -> None:
    """Strategy 2: slot pool of MFA subprocesses (Applio libri launcher pattern).

    Uses ``Popen`` + poll (not ProcessPool) so this stays safe after the parent
    has already initialized CUDA (e.g. warm Whisper in ``run_demo_batched``).
    Prefer ``threads_per_job=1`` when ``max_parallel > 1``.
    """
    if not jobs:
        return
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")

    workers = max(1, min(int(max_parallel), len(jobs)))
    threads = max(1, int(threads_per_job))
    if workers > 1 and threads > 1:
        log.warning(
            "max_parallel_align_jobs=%d with align_threads=%d; "
            "forcing -j 1 per MFA process to avoid nested parallelism",
            workers, threads,
        )
        threads = 1

    if workers <= 1:
        for job in jobs:
            run_align_mfa(
                job.ws, job.segment_names, env_mfa=env_mfa, threads=threads,
            )
        return

    queue = list(jobs)
    processes: List[Tuple[subprocess.Popen, MfaAlignJob]] = []
    log.info(
        "Running parallel MFA pool: %d job(s), max_parallel=%d, -j %d",
        len(queue), workers, threads,
    )
    while queue or processes:
        while queue and len(processes) < workers:
            job = queue.pop(0)
            per_segment_path = job.ws["aligned"].with_name(
                "03_aligned_per_segment.json"
            )
            cmd = _mfa_align_cmd(
                audio_dir=job.ws["segments"],
                input_json=job.ws["transcripts"],
                output_json=per_segment_path,
                threads=threads,
            )
            env = _mfa_align_env(env_mfa, threads_per_job=threads)
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(MFA_ALIGN_SCRIPT.parent),
            )
            processes.append((proc, job))

        alive: List[Tuple[subprocess.Popen, MfaAlignJob]] = []
        for proc, job in processes:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, job))
            elif ret != 0:
                raise RuntimeError(
                    f"MFA align failed for stem={job.stem} exit={ret}"
                )
            else:
                per_segment_path = job.ws["aligned"].with_name(
                    "03_aligned_per_segment.json"
                )
                per_segment = json.loads(
                    per_segment_path.read_text(encoding="utf-8")
                )
                pre_src = per_segment_path.with_name(
                    "03_aligned_per_segment_pre_g2p.json"
                )
                pre = (
                    json.loads(pre_src.read_text(encoding="utf-8"))
                    if pre_src.is_file()
                    else None
                )
                _finalize_mfa_workspace(
                    job.ws, job.segment_names, per_segment, pre_g2p=pre,
                )
        processes = alive
        if processes or queue:
            time.sleep(poll_interval)


def run_align_mfa_many(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    threads: int,
    max_parallel_align_jobs: int = 0,
) -> None:
    """Batch MFA for multiple workspaces.

    * ``max_parallel_align_jobs <= 0`` (default): Strategy 1 — merge corpora,
      one MFA invocation with ``--num_jobs=threads`` (best cold-start amortization).
    * ``max_parallel_align_jobs >= 1``: Strategy 2 — up to N concurrent MFA
      processes with ``-j 1`` each.
    """
    if not jobs:
        return
    if len(jobs) == 1:
        run_align_mfa(
            jobs[0].ws, jobs[0].segment_names, env_mfa=env_mfa, threads=threads,
        )
        return
    if int(max_parallel_align_jobs) >= 1:
        # N=1 → warm Whisper + serial per-stem MFA (unbatched); use full
        # --align-threads. N>1 → process pool with -j 1 each.
        threads_per = 1 if int(max_parallel_align_jobs) > 1 else max(1, int(threads))
        run_align_mfa_jobs_parallel(
            jobs,
            env_mfa=env_mfa,
            max_parallel=int(max_parallel_align_jobs),
            threads_per_job=threads_per,
        )
    else:
        run_align_mfa_batch_merged(jobs, env_mfa=env_mfa, threads=threads)




def run_align(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    run_align_mfa(
        ws,
        segment_names,
        env_mfa=args.env_mfa,
        threads=args.align_threads,
    )


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
    phone: Optional[str] = None,
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
    cmd = build_batch_infer_command(infer_args, manifest_path, phone=phone)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input", type=Path, default=None,
        help="Input WAV file or directory of WAVs "
             "(ignored / not required with --batch_manifest)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final stitched output WAV (sample rate follows --encodec) "
             "(ignored / not required with --batch_manifest)",
    )
    p.add_argument("--work-dir", type=Path, default=None, help="Workspace (default: work/<input_stem>)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint (.pt) from train_phone_ec.py or train_phone_ec_cond.py",
    )
    add_phone_map_args(p, required=False)
    p.add_argument(
        "--batch_manifest", type=Path, default=None,
        help='JSON with a top-level "pipeline_jobs" list of '
             '{"input", "output", "phones_in", "phones_out"[, "ext", "force_ext", '
             '"work_dir", "resume", "from_stage", "force"]} dicts. When set, every '
             "job shares --checkpoint/--encodec/--full_phone_conditioning/"
             "--windowed_attention/etc; segment/transcribe/align run per work "
             "dir as needed, then all inpaint jobs share one infer subprocess "
             "(checkpoint + EnCodec loaded once).",
    )
    p.add_argument(
        "--ext",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Extend each inpainted span by SEC on both sides (default: 0). "
            "Spans immediately following a /t/ also get --ext by default; "
            "pass --no-force_ext to skip those."
        ),
    )
    p.add_argument(
        "--force_ext",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply --ext even to spans immediately following a /t/ (default: on).",
    )
    p.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        help=(
            "First stage to (re)run. Always regenerates this stage and everything "
            "after it (including stitch), even with --resume. Default: full run "
            "from segment, or continue after completed stages when --resume is set alone."
        ),
    )
    p.add_argument(
        "--to-stage",
        choices=STAGES,
        default=None,
        help=(
            "Last stage to run (inclusive). Use e.g. --to-stage align to stop after "
            "forced alignment without inpaint/stitch. Default: run through stitch."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep completed stages before --from-stage (or all completed stages "
            "when --from-stage is omitted). Default without --resume: regenerate "
            "from --from-stage (or segment)."
        ),
    )
    p.add_argument("--force", action="store_true", help="Re-run stages even when --resume is set")
    p.add_argument(
        "--rewrite-transcription",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Replace ASR texts before align: pass one quoted string per segment "
            "(in segment order), or a single .json path with a list of strings / "
            "name→text object. Errors if the count does not match the number of "
            "segment files. Implies re-running align even under --resume."
        ),
    )

    # MFA is the only aligner for the Crisper sidecar product.
    p.set_defaults(speech_backend="mfa")
    p.add_argument(
        "--mfa",
        action="store_const",
        const="mfa",
        dest="speech_backend",
        help="Montreal Forced Aligner (default; only aligner for this entrypoint)",
    )

    p.add_argument("--silence-level", type=float, default=-40.0, help="Segmenter silence threshold (dBFS)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Crisper ASR batch size (default: 1)",
    )
    p.add_argument(
        "--asr-model",
        default=DEFAULT_CRISPER_MODEL,
        help=f"CrisperWhisper model id (default: {DEFAULT_CRISPER_MODEL})",
    )
    p.add_argument(
        "--crisper-cmd",
        default=None,
        help="Sidecar argv prefix (shell-quoted string). Example: "
             "'/path/to/crisper-venv/bin/python /path/to/crisper_for_mfa_w_batching.py'. "
             "Overrides PHONEPAINT_CRISPER_CMD.",
    )
    p.add_argument("--language", default="en")
    p.add_argument("--asr-device", default="cuda")
    p.add_argument(
        "--asr-device-idx",
        type=int,
        default=None,
        help="CUDA device for ASR (default: GPU with most free VRAM)",
    )
    p.add_argument(
        "--env-mfa",
        default="mfa_env",
        help="Fallback conda env for MFA if not in the active env "
             "and the active env is not PhonePaintMFA (or set MFA_BIN).",
    )
    p.add_argument("--align-threads", type=int, default=4, help="MFA worker threads")
    p.add_argument(
        "--max-parallel-align-jobs",
        type=int,
        default=0,
        help="Batch MFA across workspaces: 0 = merge corpora (default); "
             "N>=1 = up to N concurrent mfa_align_json processes with -j 1 each.",
    )

    infer = p.add_argument_group("Inpainting (infer_phone_ec / infer_phone_ec_cond)")
    infer.add_argument("--save-resampled", dest="save_resampled", type=Path, default=None)
    infer.add_argument("--device", dest="infer_device", default=None)
    infer.add_argument("--attn_heads", type=int, default=2)
    infer.add_argument("--max_utt_sec", type=float, default=float("inf"))
    infer.add_argument(
        "--ola",
        action="store_true",
        help="Enable OLA synthesis (default: disabled; single full-sequence decode)",
    )
    infer.add_argument("--ola_chunk_frames", type=int, default=32)
    infer.add_argument("--ola_hop_factor", type=int, default=4, choices=[2, 4])
    infer.add_argument("--ola_ctx_frames", type=int, default=8)
    infer.add_argument("--normalize-vol", "--normalize_vol", dest="normalize_vol", type=float, default=None)
    infer.add_argument(
        "--windowed_attention", action="store_true",
        help="Encode/inpaint/(partial-)decode only a local context window around each "
             "span (passed through to infer_phone_ec*.py; default: off). "
             "Incompatible with --full-decode.",
    )
    infer.add_argument(
        "--context_frames", type=int, default=None,
        help="Context window size when --windowed_attention is set "
             f"(default: {CONTEXT_FRAMES_16K} for --encodec 16k = ±{CONTEXT_FRAMES_16K // 2} "
             f"frames ≈ ±0.50s @ 50 fps; {CONTEXT_FRAMES_24K} for 24k = "
             f"±{CONTEXT_FRAMES_24K // 2} frames ≈ ±0.50s @ 75 fps).",
    )
    infer.add_argument(
        "--full-decode", "--full_decode", dest="full_decode", action="store_true",
        help="Pass --full-decode to infer_phone_ec_cond.py (single full-sequence "
             "EnCodec decode; requires --encodec 16k). Incompatible with "
             "--windowed_attention.",
    )
    infer.add_argument(
        "--full_phone_conditioning", action="store_true",
        help="Condition on remapped full utterance phone strings "
             "(passed through to infer_phone_ec_cond.py; requires --encodec 16k "
             "and a checkpoint trained with "
             "train_phone_ec_cond.py --full_phone_conditioning).",
    )

    p.add_argument("--crossfade-ms", type=float, default=0.0, help="Optional crossfade at stitch joins")
    p.add_argument(
        "--encodec",
        default="24k",
        choices=["24k", "16k"],
        help="EnCodec variant: '24k' uses infer_phone_ec.py, '16k' uses infer_phone_ec_cond.py (default: 24k)",
    )
    p.add_argument(
        "--encodec-ckpt",
        dest="encodec_ckpt",
        type=Path,
        default=None,
        help="Path to a local EnCodec checkpoint (.th/.pt). Required when --encodec 16k.",
    )
    p.add_argument(
        "--infer-script", dest="infer_script", type=Path, default=None,
        help="Override infer script (default: infer_phone_ec_cond.py for 16k, "
             "infer_phone_ec.py for 24k)",
    )
    args = p.parse_args()
    args.speech_backend = "mfa"
    return args


def should_run(stage: str, from_stage: str) -> bool:
    order = {s: i for i, s in enumerate(STAGES)}
    return order[stage] >= order[from_stage]


def stage_should_run(stage: str, args: argparse.Namespace, manifest: dict) -> bool:
    if not should_run(stage, args.from_stage):
        return False
    to_stage = getattr(args, "to_stage", None)
    if to_stage is not None:
        order = {s: i for i, s in enumerate(STAGES)}
        if order[stage] > order[to_stage]:
            return False
    if args.resume and not args.force and stage_done(manifest, stage):
        return False
    return True


def _resolve_shared_runtime(args: argparse.Namespace) -> int:
    """Validate encodec / full_phone / context_frames; return output sample rate."""
    if args.full_decode and args.windowed_attention:
        raise SystemExit("--full-decode and --windowed_attention cannot be combined")
    if args.full_decode and args.encodec != "16k":
        raise SystemExit("--full-decode requires --encodec 16k (infer_phone_ec_cond.py)")
    if args.encodec == "16k":
        encodec_ckpt = args.encodec_ckpt or ENCODEC_16K_CKPT
        if not encodec_ckpt.is_file():
            raise SystemExit(
                f"16 kHz EnCodec checkpoint not found: {encodec_ckpt}\n"
                "Set --encodec-ckpt or VOICECRAFT_ROOT to the VoiceCraft install."
            )
        args.encodec_ckpt = encodec_ckpt
    elif args.full_phone_conditioning:
        raise SystemExit(
            "--full_phone_conditioning requires --encodec 16k "
            "(infer_phone_ec_cond.py)"
        )

    if args.context_frames is None:
        args.context_frames = (
            CONTEXT_FRAMES_16K if args.encodec == "16k" else CONTEXT_FRAMES_24K
        )
    return _ENCODEC_FS[args.encodec]


def _overlay_job_args(
    shared: argparse.Namespace,
    job: dict,
) -> argparse.Namespace:
    """Copy *shared* and overlay per-pipeline-job fields from a batch manifest entry."""
    args = argparse.Namespace(**vars(shared))
    args.input = Path(job["input"])
    args.output = Path(job["output"])
    args.phones_in = list(job["phones_in"])
    args.phones_out = list(job["phones_out"])
    args.ext = float(job.get("ext", getattr(shared, "ext", 0.0)))
    args.force_ext = bool(job.get("force_ext", getattr(shared, "force_ext", True)))
    work_dir = job.get("work_dir")
    args.work_dir = Path(work_dir) if work_dir else None
    if "resume" in job:
        args.resume = bool(job["resume"])
    if "from_stage" in job:
        args.from_stage = job["from_stage"]
    if "force" in job:
        args.force = bool(job["force"])
    if "rewrite_transcription" in job:
        args.rewrite_transcription = job["rewrite_transcription"]
    validate_phone_map(args.phones_in, args.phones_out)
    return args


def prepare_pipeline_through_align(
    args: argparse.Namespace,
    *,
    out_fs: int,
    defer_align: bool = False,
) -> Tuple[dict[str, Path], dict, List[str], Path, bool]:
    """Run segment/transcribe/align as needed.

    Returns ``(ws, manifest, segment_names, input_dir, align_pending)``.
    When ``defer_align`` is true and the align stage would run under ``--mfa``,
    skip per-job MFA and set ``align_pending`` so the caller can batch-align.
    """
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    stem = input_path.stem if input_path.is_file() else input_path.name
    work_dir = (args.work_dir or (REPO_ROOT / "work" / stem)).resolve()
    ws = workspace_paths(work_dir)
    manifest = load_manifest(ws["manifest"])
    manifest.setdefault("input", str(input_path))
    manifest.setdefault("stem", stem)

    speech_runtime = {
        "transcriber": "crisperwhisper-sidecar",
        "asr_model": args.asr_model,
        "language": args.language,
        "crisper_cmd": args.crisper_cmd or os.environ.get("PHONEPAINT_CRISPER_CMD"),
        "aligner": "mfa",
        "env_mfa": args.env_mfa,
        "align_threads": args.align_threads,
        "phone_inventory": "phonepaint-arpa-v1",
    }
    previous_runtime = manifest.get("speech_runtime")
    if previous_runtime != speech_runtime:
        if previous_runtime is None and stage_done(manifest, "align"):
            invalidate_from_stage(ws, "align", manifest, stages=STAGES)
        elif previous_runtime is not None:
            transcription_keys = (
                "transcriber", "asr_model", "language", "crisper_cmd",
            )
            transcription_changed = any(
                previous_runtime.get(key) != speech_runtime.get(key)
                for key in transcription_keys
            )
            invalidate_from_stage(
                ws,
                "transcribe" if transcription_changed else "align",
                manifest,
                stages=STAGES,
            )
        manifest["speech_runtime"] = speech_runtime

    args.from_stage = prepare_stage_start(
        ws,
        manifest,
        stages=STAGES,
        from_stage=args.from_stage,
        resume=args.resume,
    )
    # Batch/resume may request inpaint while a backend switch invalidated align.
    if args.from_stage in ("inpaint", "stitch") and not ws["aligned"].is_file():
        log.warning(
            "%s: aligned JSON missing; re-running from align (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "align"
    if args.from_stage in ("align", "inpaint", "stitch") and not ws["transcripts"].is_file():
        log.warning(
            "%s: transcripts missing; re-running from transcribe (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "transcribe"
    if args.from_stage != "segment" and not any(ws["segments"].glob("*.wav")):
        log.warning(
            "%s: segments missing; re-running from segment (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "segment"
    save_manifest(ws["manifest"], manifest)

    input_dir = prepare_input(input_path, ws)

    if stage_should_run("segment", args, manifest):
        log.info("=== Stage: segment (%s) ===", stem)
        with wallclock(f"stage segment ({stem})"):
            segment_names = run_segment(
                input_dir, ws, silence_level=args.silence_level,
            )
        manifest["segments"] = segment_names
        mark_stage(manifest, "segment")
        save_manifest(ws["manifest"], manifest)
    else:
        segment_names = manifest.get("segments") or [
            p.name for p in sorted_segment_wavs(ws["segments"])
        ]
        if not segment_names:
            log.warning(
                "No cached segments for %s; running from segment "
                "(requested --from-stage %s)",
                stem, args.from_stage,
            )
            args.from_stage = "segment"
            log.info("=== Stage: segment (%s) ===", stem)
            with wallclock(f"stage segment ({stem})"):
                segment_names = run_segment(
                    input_dir, ws, silence_level=args.silence_level,
                )
            manifest["segments"] = segment_names
            mark_stage(manifest, "segment")
            save_manifest(ws["manifest"], manifest)

    if not segment_names:
        raise SystemExit(f"No segments found for {stem}; run segment stage first.")

    rewrite_texts = load_rewrite_texts(args.rewrite_transcription)
    if rewrite_texts is not None:
        order = {s: i for i, s in enumerate(STAGES)}
        if order[args.from_stage] > order["align"]:
            raise SystemExit(
                "--rewrite-transcription requires re-running align; "
                "use --from-stage align (or earlier), typically with --resume."
            )

    if stage_should_run("transcribe", args, manifest):
        if rewrite_texts is not None:
            raise SystemExit(
                "--rewrite-transcription replaces ASR output; do not also run the "
                "transcribe stage. Use --from-stage align (typically with --resume)."
            )
        log.info("=== Stage: transcribe (%s) ===", stem)
        with wallclock(f"stage transcribe ({stem})"):
            run_transcribe(ws, args)
        mark_stage(manifest, "transcribe")
        save_manifest(ws["manifest"], manifest)

    if rewrite_texts is not None:
        log.info(
            "=== Rewrite transcriptions (%s, %d segment(s)) ===",
            stem, len(rewrite_texts),
        )
        rewrite_transcriptions(ws["transcripts"], segment_names, rewrite_texts)
        invalidate_from_stage(ws, "align", manifest, stages=STAGES)
        mark_stage(manifest, "transcribe")
        save_manifest(ws["manifest"], manifest)

    need_align = stage_should_run("align", args, manifest) or rewrite_texts is not None
    align_pending = bool(
        need_align and defer_align and args.speech_backend == "mfa"
    )
    if need_align and not align_pending:
        log.info("=== Stage: align (%s) ===", stem)
        with wallclock(f"stage align ({stem})"):
            run_align(ws, segment_names, args)
        mark_stage(manifest, "align")
        save_manifest(ws["manifest"], manifest)
    elif align_pending:
        log.info(
            "=== Stage: align (%s) deferred for MFA batch ===", stem,
        )

    return ws, manifest, segment_names, input_dir, align_pending


def finish_pipeline_after_inpaint(
    args: argparse.Namespace,
    *,
    ws: dict[str, Path],
    manifest: dict,
    source_files: Sequence[str],
    out_fs: int,
) -> None:
    """Stitch inpainted sources and deploy to --output."""
    if stage_should_run("stitch", args, manifest):
        log.info("=== Stage: stitch (%s) ===", Path(args.output).name)
        inpainted_paths = [ws["inpainted"] / f"{stem}.wav" for stem in source_files]
        missing = [str(p) for p in inpainted_paths if not p.is_file()]
        if missing:
            raise SystemExit(
                f"Missing inpainted source files: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )
        with wallclock(f"stage stitch ({Path(args.output).name})"):
            stitch_wavs(
                inpainted_paths, ws["stitched"],
                crossfade_ms=args.crossfade_ms, out_fs=out_fs,
            )
        mark_stage(manifest, "stitch")
        save_manifest(ws["manifest"], manifest)

    deploy_stitched_output(ws["stitched"], args.output)
    if ws["stitched"].is_file():
        manifest["output"] = str(args.output.resolve())
        save_manifest(ws["manifest"], manifest)


def run_pipeline_single(args: argparse.Namespace) -> None:
    """Original single --input/--output pipeline path."""
    validate_phone_map(args.phones_in, args.phones_out)
    out_fs = _resolve_shared_runtime(args)
    pipeline_t0 = time.perf_counter()
    reset_stage_stats()

    ws, manifest, segment_names, input_dir, _align_pending = prepare_pipeline_through_align(
        args, out_fs=out_fs,
    )
    stem = Path(args.input).stem

    if args.to_stage is not None and STAGES.index(args.to_stage) < STAGES.index("inpaint"):
        write_stage_stats(ws["root"], stem=stem)
        log.info(
            "[timing] pipeline total (through %s) — %s",
            args.to_stage,
            _fmt_elapsed(time.perf_counter() - pipeline_t0),
        )
        log.info("Pipeline stopped after --to-stage %s: %s", args.to_stage, ws["root"])
        return

    if stage_should_run("inpaint", args, manifest):
        log.info("=== Stage: inpaint ===")
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        with wallclock("stage inpaint"):
            source_files = run_inpaint(
                ws, infer_args=args,
                segment_names=segment_names, input_dir=input_dir, out_fs=out_fs,
            )
        manifest["source_files"] = source_files
        mark_stage(manifest, "inpaint")
        save_manifest(ws["manifest"], manifest)
    else:
        source_files = manifest.get("source_files") or source_order_from_segments(
            manifest.get("segments")
            or [p.name for p in sorted_segment_wavs(ws["segments"])]
        )

    finish_pipeline_after_inpaint(
        args, ws=ws, manifest=manifest, source_files=source_files, out_fs=out_fs,
    )
    write_stage_stats(ws["root"], stem=stem)
    log.info(
        "[timing] pipeline total — %s",
        _fmt_elapsed(time.perf_counter() - pipeline_t0),
    )
    log.info("Pipeline complete: %s", args.output)


def run_pipeline_batch(args: argparse.Namespace) -> None:
    """Multi-input batch: prepare each workspace, one shared infer, then stitch each."""
    out_fs = _resolve_shared_runtime(args)
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    raw = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    pipeline_jobs = raw["pipeline_jobs"] if isinstance(raw, dict) else raw
    if not pipeline_jobs:
        raise SystemExit(f"--batch_manifest {args.batch_manifest} has no pipeline_jobs")

    pipeline_t0 = time.perf_counter()
    prepared: List[dict] = []
    pending_mfa: List[MfaAlignJob] = []
    all_infer_jobs: List[dict] = []
    phone0_labels: set[str] = set()
    defer_mfa = args.speech_backend == "mfa"

    for i, job in enumerate(pipeline_jobs):
        job_args = _overlay_job_args(args, job)
        log.info(
            "=== Batch pipeline job %d/%d: %s -> %s ===",
            i + 1, len(pipeline_jobs), job_args.input.name, job_args.output.name,
        )
        reset_stage_stats()
        ws, manifest, segment_names, input_dir, align_pending = (
            prepare_pipeline_through_align(
                job_args, out_fs=out_fs, defer_align=defer_mfa,
            )
        )
        stem = Path(job_args.input).stem
        if align_pending:
            pending_mfa.append(
                MfaAlignJob(stem=stem, ws=ws, segment_names=list(segment_names))
            )

        # Align may still be pending; inpaint collection needs 03_aligned.json.
        # Defer collect_inpaint_jobs until after the MFA batch below.
        prepared.append(dict(
            args=job_args, ws=ws, manifest=manifest,
            segment_names=segment_names, input_dir=input_dir,
            stem=stem, align_pending=align_pending,
            pre_inpaint_timing=dict(_STAGE_TIMING),
            pre_inpaint_vram=_VRAM_PEAK_MB,
            needs_inpaint_mark=False,
            source_files=[],
        ))

    if pending_mfa:
        log.info(
            "=== Stage: align (MFA batch, %d workspace(s), max_parallel=%d) ===",
            len(pending_mfa),
            int(getattr(args, "max_parallel_align_jobs", 0) or 0),
        )
        reset_stage_stats()
        with wallclock("stage align (MFA batch)"):
            run_align_mfa_many(
                pending_mfa,
                env_mfa=args.env_mfa,
                threads=args.align_threads,
                max_parallel_align_jobs=int(
                    getattr(args, "max_parallel_align_jobs", 0) or 0
                ),
            )
        shared_align_t = float(_STAGE_TIMING.get("align", 0.0))
        shared_align_vram = float(_VRAM_PEAK_MB)
        n_pending = max(1, len(pending_mfa))
        share = shared_align_t / float(n_pending)
        for entry in prepared:
            if not entry["align_pending"]:
                continue
            mark_stage(entry["manifest"], "align")
            save_manifest(entry["ws"]["manifest"], entry["manifest"])
            # Attribute shared MFA wall evenly across deferred stems.
            merged = dict(entry["pre_inpaint_timing"])
            merged["align"] = share
            load_stage_stats_state(
                merged,
                max(float(entry["pre_inpaint_vram"]), shared_align_vram),
            )
            entry["pre_inpaint_timing"] = dict(_STAGE_TIMING)
            entry["pre_inpaint_vram"] = _VRAM_PEAK_MB

    # Collect inpaint jobs now that alignments exist.
    for entry in prepared:
        job_args = entry["args"]
        ws = entry["ws"]
        manifest = entry["manifest"]
        segment_names = entry["segment_names"]
        input_dir = entry["input_dir"]
        if stage_should_run("inpaint", job_args, manifest):
            source_files, batch_jobs, _counts = collect_inpaint_jobs(
                ws,
                infer_args=job_args,
                segment_names=segment_names,
                input_dir=input_dir,
                out_fs=out_fs,
            )
            all_infer_jobs.extend(batch_jobs)
            phone0_labels.add(str(job_args.phones_out[0]))
            manifest["source_files"] = source_files
            entry["source_files"] = source_files
            entry["needs_inpaint_mark"] = True
        else:
            source_files = list(
                manifest.get("source_files") or source_order_from_segments(
                    manifest.get("segments")
                    or [p.name for p in sorted_segment_wavs(ws["segments"])]
                )
            )
            write_stage_stats(ws["root"], stem=entry["stem"])
            entry["source_files"] = source_files
            entry["needs_inpaint_mark"] = False

    if all_infer_jobs:
        if args.encodec != "16k" and len(phone0_labels) > 1:
            raise SystemExit(
                "--batch_manifest with --encodec 24k requires every pipeline_job's "
                f"phones_out[0] to match (model metadata is shared); got "
                f"{sorted(phone0_labels)}"
            )
        phone = next(iter(phone0_labels)) if phone0_labels else None
        # Prefer a stable temp path under the first workspace's spans dir.
        first_ws = prepared[0]["ws"]
        first_ws["spans"].mkdir(parents=True, exist_ok=True)
        manifest_path = first_ws["spans"] / "_demo_pipeline_batch_manifest.json"
        log.info("=== Stage: inpaint (shared batch, %d source file(s)) ===", len(all_infer_jobs))
        reset_stage_stats()
        with wallclock("stage inpaint (shared batch)"):
            execute_infer_batch(
                infer_args=args,
                batch_jobs=all_infer_jobs,
                manifest_path=manifest_path,
                phone=phone,
            )
        shared_inpaint_t = float(_STAGE_TIMING.get("inpaint", 0.0))
        shared_vram = float(_VRAM_PEAK_MB)
    else:
        shared_inpaint_t = 0.0
        shared_vram = 0.0

    for entry in prepared:
        job_args = entry["args"]
        ws = entry["ws"]
        manifest = entry["manifest"]
        if entry["needs_inpaint_mark"]:
            mark_stage(manifest, "inpaint")
            save_manifest(ws["manifest"], manifest)
        finish_pipeline_after_inpaint(
            job_args,
            ws=ws,
            manifest=manifest,
            source_files=entry["source_files"],
            out_fs=out_fs,
        )
        # Merge pre-inpaint stage times with this job's share of batch inpaint.
        merged = dict(entry["pre_inpaint_timing"])
        if entry["needs_inpaint_mark"]:
            merged["inpaint"] = shared_inpaint_t
        load_stage_stats_state(
            merged, max(float(entry["pre_inpaint_vram"]), shared_vram),
        )
        write_stage_stats(ws["root"], stem=entry["stem"])
        log.info("Pipeline complete: %s", job_args.output)

    log.info(
        "[timing] pipeline batch total — %d job(s), %s",
        len(pipeline_jobs),
        _fmt_elapsed(time.perf_counter() - pipeline_t0),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.batch_manifest is not None:
        if args.input is not None or args.output is not None:
            raise SystemExit(
                "--batch_manifest cannot be combined with --input/--output"
            )
        if args.phones_in is not None or args.phones_out is not None:
            raise SystemExit(
                "--batch_manifest cannot be combined with --phones_in/--phones_out "
                "(supply them per pipeline_job in the manifest)"
            )
        run_pipeline_batch(args)
        return

    if args.input is None or args.output is None:
        raise SystemExit(
            "--input and --output are required unless --batch_manifest is given"
        )
    if args.phones_in is None or args.phones_out is None:
        raise SystemExit(
            "--phones_in and --phones_out are required unless --batch_manifest is given"
        )
    run_pipeline_single(args)


if __name__ == "__main__":
    main()

