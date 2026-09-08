"""Shared time-span helpers for PhonePaint inference."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def apply_ext(
    spans: Sequence[Tuple[float, float]],
    *,
    ext_sec: float,
    max_end: float,
) -> List[Tuple[float, float]]:
    """Extend each span on both sides and clip it to the source duration."""
    out: List[Tuple[float, float]] = []
    for start, end in spans:
        clipped_start = max(0.0, start - ext_sec)
        clipped_end = min(max_end, end + ext_sec)
        if clipped_end > clipped_start:
            out.append((clipped_start, clipped_end))
    return out
