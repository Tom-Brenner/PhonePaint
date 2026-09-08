"""Pipeline timing and CUDA peak-memory accounting."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from .constants import STAGES

log = logging.getLogger(__name__)

_STAGE_TIMING: Dict[str, float] = {}
_VRAM_PEAK_MB: float = 0.0

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



def stage_timing() -> Dict[str, float]:
    """Return a copy of the current job's per-stage timings."""
    return dict(_STAGE_TIMING)

def vram_peak_mb() -> float:
    """Return the current job's peak CUDA allocation in MiB."""
    return _VRAM_PEAK_MB
