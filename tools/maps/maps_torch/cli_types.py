"""Shared CLI datatypes."""

from __future__ import annotations

import itertools
import re

from maps_torch.alignment import collapse


class WordString:
    def __init__(self, words: list[str], pronunciations: list[list[str]]):
        self.words = words
        self.pronunciations = pronunciations
        self.phone_string = list(itertools.chain(*pronunciations))
        self.collapsed_string = collapse([re.sub(r"[0-9]", "", x) for x in self.phone_string])
        self.did_collapse = len(self.phone_string) != len(self.collapsed_string)

    def __str__(self) -> str:
        return str([self.words, f"collapsed_diff={self.did_collapse}", self.pronunciations])
