"""TextGrid construction for MAPS alignments."""

from __future__ import annotations

import itertools
import math
import re
import numpy as np
from textgrid import textgrid

from maps_torch.alignment import PhoneLabel, collapse
from maps_torch.cli_types import WordString
from maps_torch.features import FRAME_INTERVAL

FRAME_LENGTH = 0.025  # 25 ms, kept for parity with original MAPS


def to_bucket_fmt(s: list[str]) -> list[tuple[str, int]]:
    s = [re.sub(r"[0-9]", "", x) for x in s]
    buckets: list[tuple[str, int]] = []
    prev = s[0]
    count = 1
    for x in s[1:]:
        if x != prev:
            buckets.append((prev, count))
            count = 0
        prev = x
        count += 1
    buckets.append((prev, count))
    return buckets


def unmerge_phones(tier: textgrid.IntervalTier, words: WordString) -> None:
    collapsed_bucket = to_bucket_fmt(words.collapsed_string)
    uncollapsed_bucket = to_bucket_fmt(words.phone_string)
    intervals = []
    for i, (c, u) in enumerate(zip(collapsed_bucket, uncollapsed_bucket)):
        dur = tier.intervals[i].maxTime - tier.intervals[i].minTime
        chunk_dur = dur / u[1]
        mint = tier.intervals[i].minTime
        for j in range(u[1]):
            low = mint + j * chunk_dur
            high = low + chunk_dur
            intervals.append(textgrid.Interval(minTime=low, maxTime=high, mark=c[0]))
    tier.intervals = intervals


def make_word_tier(segment_tier: textgrid.IntervalTier, words: WordString) -> textgrid.IntervalTier:
    words_int = textgrid.IntervalTier()
    words_int.name = "words"
    word_ends = np.cumsum([len(p) for p in words.pronunciations]) - 1
    max_time = segment_tier[word_ends[0]].maxTime
    words_int.intervals.append(textgrid.Interval(minTime=0, maxTime=max_time, mark=words.words[0]))
    for w, w_end in zip(words.words[1:], word_ends[1:]):
        min_time = words_int[-1].maxTime
        max_time = segment_tier[w_end].maxTime
        words_int.intervals.append(textgrid.Interval(minTime=min_time, maxTime=max_time, mark=w))
    return words_int


def interpolated_part(end_cur: int, phone_n: int, probs: np.ndarray) -> float:
    phone1_curr = probs[end_cur, phone_n]
    phone1_next = probs[end_cur + 1, phone_n]
    phone2_curr = probs[end_cur, phone_n + 1]
    phone2_next = probs[end_cur + 1, phone_n + 1]

    m1 = (phone1_next - phone1_curr) / FRAME_INTERVAL
    m2 = (phone2_next - phone2_curr) / FRAME_INTERVAL

    a = np.array([[-m1, 1], [-m2, 1]])
    b = [phone1_curr, phone2_curr]

    try:
        time_point, _ = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return 0.0

    if 0 <= time_point < FRAME_INTERVAL:
        return float(time_point)
    return 0.0


def make_textgrid(
    seq: list[PhoneLabel],
    tgname,
    max_time: float,
    words: WordString,
    *,
    interpolate: bool = True,
    probs: np.ndarray | None = None,
) -> None:
    if interpolate and probs is None:
        raise ValueError("Interpolation requires the alignment probability matrix via probs")

    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier()
    tier.name = "phones"

    if len(seq) == 1:
        tier.intervals.append(textgrid.Interval(0, max_time, seq[-1].phone))
        if words.did_collapse:
            unmerge_phones(tier, words)
        tg.tiers.append(make_word_tier(tier, words))
        tg.tiers.append(tier)
        tg.write(tgname)
        return

    added_bits: list[float] = []
    frame_durs = [s.duration for s in seq]
    cumu_frame_durs = [sum(frame_durs[0 : i + 1]) for i in range(len(frame_durs))]
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015

    if interpolate:
        additional = interpolated_part(seq[0].duration - 1, 0, probs)
        if curr_dur + additional < max_time:
            curr_dur += additional
        added_bits.append(additional)

    tier.intervals.append(textgrid.Interval(0, curr_dur, seq[0].phone))

    for i, s in enumerate(seq[:-1]):
        if i == 0:
            continue
        beginning = curr_dur
        dur = FRAME_INTERVAL * s.duration
        if interpolate:
            end_cur = cumu_frame_durs[i] - 1
            dur -= added_bits[-1]
            additional = interpolated_part(end_cur, i, probs)
            if beginning + dur + additional < max_time:
                dur += additional
            added_bits.append(additional)
        ending = beginning + dur
        tier.intervals.append(textgrid.Interval(beginning, ending, s.phone))
        curr_dur = ending

    tier.intervals.append(textgrid.Interval(curr_dur, max_time, seq[-1].phone))

    if words.did_collapse:
        unmerge_phones(tier, words)

    for i in range(len(tier.intervals)):
        x = words.phone_string[i]
        if x != tier.intervals[i].mark:
            tier.intervals[i].mark = x
        if i > 0:
            prev_end = tier.intervals[i - 1].maxTime
            curr_start = tier.intervals[i].minTime
            if math.isclose(prev_end, curr_start) and prev_end != curr_start:
                tier.intervals[i - 1].maxTime = curr_start

    tg.tiers.append(make_word_tier(tier, words))
    tg.tiers.append(tier)
    tg.write(tgname)
