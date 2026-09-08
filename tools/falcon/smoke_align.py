#!/usr/bin/env python3
"""Smoke test: word-level FALCON alignment -> Praat TextGrid.

Run inside falcon_env (after ./create_falcon_env.sh):

  conda run -n falcon_env python tools/falcon/smoke_align.py \\
    --wav examples/audio/Nikki3.wav \\
    --text "yeah i have been like ..."

Requires tools/falcon/vendor/FALCON (cloned by create_falcon_env.sh) and
checkpoints under tools/falcon/pretrained_models/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SCRIPT_DIR = Path(__file__).resolve().parent
FALCON_ROOT = SCRIPT_DIR
VENDOR = FALCON_ROOT / "vendor" / "FALCON"
CKPT_DIR = FALCON_ROOT / "pretrained_models"
TARGET_SR = 16_000

CKPT_BY_KEY = {
    "timit": CKPT_DIR / "falcon_timit_english.pt",
    "buckeye": CKPT_DIR / "falcon_buckeye_english.pt",
    "multilingual": CKPT_DIR / "falcon_joint_multilingual.pt",
}


def _prepare_wav(src: Path, dst: Path) -> None:
    wav, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR, quality="HQ").astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), mono, TARGET_SR, subtype="PCM_16")


def _parse_textgrid_phones(tg_path: Path, limit: int = 12) -> list[tuple[float, float, str]]:
    text = tg_path.read_text(encoding="utf-8")
    in_phones = False
    intervals: list[tuple[float, float, str]] = []
    xmin = xmax = label = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('name = "phones"'):
            in_phones = True
            continue
        if in_phones and line.startswith('name = "') and not line.startswith('name = "phones"'):
            break
        if not in_phones:
            continue
        if line.startswith("xmin ="):
            xmin = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            xmax = float(line.split("=", 1)[1].strip())
        elif line.startswith("text ="):
            label = line.split("=", 1)[1].strip().strip('"')
            if xmin is not None and xmax is not None and label:
                intervals.append((xmin, xmax, label))
            xmin = xmax = label = None
    return intervals[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True, help="Input wav (any rate/channels)")
    parser.add_argument(
        "--text",
        default=None,
        help="Plain transcript (words). Default: read <wav_stem>.txt beside --wav",
    )
    parser.add_argument(
        "--ckpt",
        choices=sorted(CKPT_BY_KEY),
        default="buckeye",
        help="Checkpoint preset (default: buckeye for spontaneous speech)",
    )
    parser.add_argument(
        "--lang",
        choices=("english", "multilingual"),
        default="english",
        help="FALCON --lang passed to generate_textgrids.py",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory (default: temp dir under tools/falcon/)",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete --work-dir when done",
    )
    args = parser.parse_args()

    if not VENDOR.is_dir():
        raise SystemExit(
            f"FALCON vendor missing: {VENDOR}\nRun ./create_falcon_env.sh first."
        )
    ckpt = CKPT_BY_KEY[args.ckpt]
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}\nRun ./create_falcon_env.sh first.")

    wav_in = args.wav.expanduser().resolve()
    if not wav_in.is_file():
        raise SystemExit(f"Input wav not found: {wav_in}")

    text = args.text
    if text is None:
        sidecar = wav_in.with_suffix(".txt")
        if not sidecar.is_file():
            raise SystemExit(f"Pass --text or provide {sidecar}")
        text = sidecar.read_text(encoding="utf-8").strip()
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        raise SystemExit("Empty transcript")

    cleanup = False
    if args.work_dir is None:
        work = Path(tempfile.mkdtemp(prefix="falcon_smoke_", dir=FALCON_ROOT))
        cleanup = not args.keep_work_dir
    else:
        work = args.work_dir.expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)

    stem = wav_in.stem
    wav_out = work / f"{stem}.wav"
    txt_out = work / f"{stem}.txt"
    tg_out = work / f"{stem}.TextGrid"

    try:
        _prepare_wav(wav_in, wav_out)
        txt_out.write_text(text + "\n", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(VENDOR)
        env.setdefault("FDNFA_WORD_G2P", "espeak")

        gen_script = VENDOR / "generate_textgrids.py"
        runner = FALCON_ROOT / "run_vendor_cli.py"
        if not gen_script.is_file():
            raise SystemExit(f"Missing {gen_script}")

        cmd = [
            sys.executable,
            str(runner),
            "generate_textgrids.py",
            "--wav", str(wav_out),
            "--mode", "word",
            "--lang", args.lang,
            "--annotation", "txt",
            "--ckpt", str(ckpt),
        ]
        subprocess.run(cmd, cwd=str(VENDOR), env=env, check=True)

        if not tg_out.is_file():
            raise SystemExit(f"Expected TextGrid not written: {tg_out}")

        phones = _parse_textgrid_phones(tg_out)
        print(f"\nOK: {tg_out} ({len(phones)} phone intervals shown, may be truncated)")
        for xmin, xmax, label in phones:
            print(f"  {xmin:7.3f} – {xmax:7.3f}  {label}")
        if len(phones) == 0:
            raise SystemExit("TextGrid phones tier is empty")
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
