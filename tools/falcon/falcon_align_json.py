#!/usr/bin/env python3
"""JSON-manifest FALCON word-mode alignment for run_inpaint_pipeline.py --falcon.

Reads the same segment transcript JSON as MFA/BFA (``02_transcripts.json``),
runs FALCON per segment in the **active** interpreter (PhonePaintUnified; no
conda hop), and writes per-segment alignments with PhonePaint ``canonical``
phone labels (LH39 -> stressless ARPAbet).

Typical usage:

  python tools/falcon/falcon_align_json.py \\
    --audio_dir work/Nikki2/01_segments \\
    --input_json work/Nikki2/02_transcripts.json \\
    --output_json work/Nikki2/03_aligned_per_segment.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from textgrid import IntervalTier, TextGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from phone_labels import canonical_phone_label  # noqa: E402

FALCON_ROOT = Path(__file__).resolve().parent
VENDOR = FALCON_ROOT / "vendor" / "FALCON"
RUNNER = FALCON_ROOT / "run_vendor_cli.py"
CKPT_DIR = FALCON_ROOT / "pretrained_models"
TARGET_SR = 16_000

CKPT_PRESETS = {
    "timit": CKPT_DIR / "falcon_timit_english.pt",
    "buckeye": CKPT_DIR / "falcon_buckeye_english.pt",
    "multilingual": CKPT_DIR / "falcon_joint_multilingual.pt",
}


def _resolve_ckpt(name: str) -> Path:
    path = CKPT_PRESETS.get(name)
    if path is not None:
        return path
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    raise SystemExit(f"Unknown --ckpt {name!r}; expected preset or path to .pt file")


def _ensure_16k_mono(wav_path: Path) -> None:
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR, quality="HQ").astype(np.float32)
        sf.write(str(wav_path), mono, TARGET_SR, subtype="PCM_16")


def _pick_tiers(tg: TextGrid) -> tuple[IntervalTier | None, IntervalTier | None]:
    phones_tier = None
    words_tier = None
    for tier in tg.tiers:
        if not isinstance(tier, IntervalTier):
            continue
        lname = (tier.name or "").lower()
        if "phone" in lname or "segment" in lname:
            phones_tier = tier
        if "word" in lname:
            words_tier = tier
    return phones_tier, words_tier


def _tier_to_dict(tier: IntervalTier | None, *, phones: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if tier is None:
        return out
    for interval in tier:
        text = (interval.mark or "").strip()
        if not text:
            continue
        start = round(float(interval.minTime), 3)
        end = round(float(interval.maxTime), 3)
        if end <= start:
            continue
        item: dict = {"xmin": start, "xmax": end, "text": text}
        if phones:
            item["canonical"] = canonical_phone_label(text)
            item["inventory"] = "falcon-lh39"
        out[str(len(out))] = item
    return out


def _parse_textgrid(tg_path: Path) -> dict:
    tg = TextGrid()
    tg.read(str(tg_path))
    phones_tier, words_tier = _pick_tiers(tg)
    phones = _tier_to_dict(phones_tier, phones=True)
    if not phones:
        raise RuntimeError(f"No phone intervals in {tg_path}")
    return {
        "words": _tier_to_dict(words_tier, phones=False),
        "phones": phones,
        "alignment": {
            "backend": "falcon",
            "inventory": "falcon-lh39",
        },
    }


def _load_transcripts(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected transcript object in {path}")
    texts: dict[str, str] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
        else:
            text = str(entry or "").strip()
        if not text:
            raise ValueError(f"Missing transcript text for {name} in {path}")
        texts[str(name)] = text
    return texts


def _align_segment(
    wav_path: Path,
    text: str,
    *,
    ckpt: Path,
    lang: str,
    env: dict[str, str],
) -> dict:
    txt_path = wav_path.with_suffix(".txt")
    tg_path = wav_path.with_suffix(".TextGrid")
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        raise ValueError(f"Empty transcript for {wav_path.name}")
    txt_path.write_text(clean + "\n", encoding="utf-8")
    _ensure_16k_mono(wav_path)

    cmd = [
        sys.executable,
        str(RUNNER),
        "generate_textgrids.py",
        "--wav",
        str(wav_path),
        "--mode",
        "word",
        "--lang",
        lang,
        "--annotation",
        "txt",
        "--ckpt",
        str(ckpt),
    ]
    subprocess.run(cmd, cwd=str(VENDOR), env=env, check=True)
    if not tg_path.is_file():
        raise RuntimeError(f"FALCON did not write {tg_path}")
    return _parse_textgrid(tg_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio_dir", type=Path, required=True)
    ap.add_argument("--input_json", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument(
        "--ckpt",
        default="buckeye",
        help="Checkpoint preset (timit|buckeye|multilingual) or path to .pt (default: buckeye)",
    )
    ap.add_argument(
        "--lang",
        default="english",
        choices=("english", "multilingual"),
        help="FALCON language mode (default: english)",
    )
    args = ap.parse_args()

    if not VENDOR.is_dir():
        raise SystemExit(f"FALCON vendor missing: {VENDOR}\nRun ./create_falcon_env.sh first.")
    if not RUNNER.is_file():
        raise SystemExit(f"Missing vendor runner: {RUNNER}")

    ckpt = _resolve_ckpt(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}\nRun ./create_falcon_env.sh first.")

    audio_dir = args.audio_dir.expanduser().resolve()
    transcripts = _load_transcripts(args.input_json.expanduser().resolve())

    env = os.environ.copy()
    env["PYTHONPATH"] = str(VENDOR)
    env.setdefault("FDNFA_WORD_G2P", "espeak")

    aligned: dict[str, dict] = {}
    for name, text in transcripts.items():
        wav_path = audio_dir / name
        if not wav_path.is_file():
            raise FileNotFoundError(f"Segment wav not found: {wav_path}")
        print(f"FALCON align: {name}", flush=True)
        aligned[name] = _align_segment(
            wav_path,
            text,
            ckpt=ckpt,
            lang=args.lang,
            env=env,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(aligned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json} ({len(aligned)} segment(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
