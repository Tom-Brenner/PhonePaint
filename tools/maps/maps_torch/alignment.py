"""Forced-alignment DP and dictionary helpers (TensorFlow-free)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from maps_torch.phones import num2phn, phn2num
from utils import align, collapse, fold_phone

__all__ = [
    "align",
    "collapse",
    "fold_phone",
    "load_dictionary",
    "force_align",
    "PhoneLabel",
]


class PhoneLabel:
    def __init__(self, phone: str, duration: int):
        self.phone = phone
        self.duration = duration

    def __str__(self) -> str:
        return str([self.phone, self.duration])


def load_dictionary(dname: Path | str, cmuformat: bool = True) -> dict:
    mapping: dict = {}
    if not cmuformat:
        print("Custom dictionary formats not supported yet. Please format use CMU dict formatting and run again.")
        sys.exit(1)

    with open(dname, "r") as d:
        for line in d:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            all_items = line.split()
            if len(all_items) < 2:
                continue
            word = re.sub(r"\(\d*\)", "", all_items[0])
            # NLTK cmudict uses ``WORD <variant_index> PHONE ...``; skip the index.
            pron_start = 2 if len(all_items) >= 3 and all_items[1].isdigit() else 1
            pronunciation = [fold_phone(p) for p in all_items[pron_start:]]
            if not pronunciation:
                continue
            if word not in mapping:
                mapping[word] = [pronunciation]
            else:
                mapping[word].append(pronunciation)

    mapping["sil"] = [["H#"], []]
    return mapping


def force_align(collapsed: list[str], yhat: np.ndarray) -> tuple[list[PhoneLabel], np.ndarray]:
    yhat = np.squeeze(yhat, 0)
    # Clamp only to keep log defined; MAPS uses abs(log(p)).
    yhat = np.clip(yhat, 1e-12, 1.0)
    predictions = np.abs(np.log(yhat))
    collapsed_ids = [phn2num[x.lower()] for x in collapsed]
    a, m = align(collapsed_ids, predictions)
    a = [num2phn[p] for p in a]
    seq = [PhoneLabel(phone=a[0], duration=1)]
    for elem in a[1:]:
        if seq[-1].phone != elem:
            seq.append(PhoneLabel(phone=elem, duration=1))
        else:
            seq[-1].duration += 1
    return seq, m
