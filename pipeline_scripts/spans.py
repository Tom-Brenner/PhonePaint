"""Phone-label normalization and inpainting span construction."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from phone_labels import canonical_phone_label, canonical_user_phone

from .constants import _SHORT_SZ_PHONES, _T_ALIGN_LABELS

log = logging.getLogger(__name__)

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


