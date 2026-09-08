"""Patch panphon 0.21.0 for FALCON word G2P on Python 3.10+."""

from __future__ import annotations

import importlib.util
import os


def _panphon_dir() -> str | None:
    spec = importlib.util.find_spec("panphon")
    locs = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locs:
        return None
    return list(locs)[0]


def patch_featuretable() -> bool:
    """Fix invalid type annotation inside a function call in featuretable.py."""
    root = _panphon_dir()
    if not root:
        return False
    ft = os.path.join(root, "featuretable.py")
    bad = "word_features = self.word_fts(word: str, normalize: bool=True)"
    good = "word_features = self.word_fts(word, normalize=True)"
    try:
        with open(ft, encoding="utf-8") as fh:
            src = fh.read()
        if bad not in src:
            return False
        with open(ft, "w", encoding="utf-8") as fh:
            fh.write(src.replace(bad, good))
        return True
    except OSError:
        return False


def patch_segment_len() -> bool:
    """Add __len__ required by collections.abc.Mapping on Python 3.10+."""
    root = _panphon_dir()
    if not root:
        return False
    seg = os.path.join(root, "segment.py")
    marker = "    def __iter__(self) -> Iterator[str]:"
    insert = (
        "    def __len__(self) -> int:\n"
        "        return len(self.names)\n\n"
    )
    try:
        with open(seg, encoding="utf-8") as fh:
            src = fh.read()
        if "__len__" in src:
            return False
        if marker not in src:
            return False
        with open(seg, "w", encoding="utf-8") as fh:
            fh.write(src.replace(marker, insert + marker, 1))
        return True
    except OSError:
        return False


def patch_panphon() -> bool:
    return patch_featuretable() or patch_segment_len()


if __name__ == "__main__":
    ft = patch_featuretable()
    seg = patch_segment_len()
    if ft or seg:
        print("patched:", ", ".join(x for x, ok in [("featuretable", ft), ("segment", seg)] if ok))
    else:
        print("already ok or panphon missing")
