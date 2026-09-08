"""Workspace paths, manifests, and segment file ordering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from .constants import _SEGMENT_RE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

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

