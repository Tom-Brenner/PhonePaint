#!/usr/bin/env python3
"""Download FALCON checkpoints into tools/falcon/pretrained_models/."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "pretrained_models"
REPO = os.environ.get("FALCON_HF_REPO", "MLSpeech/FALCON-weights")
FILES = (
    "falcon_timit_english.pt",
    "falcon_buckeye_english.pt",
    "falcon_joint_multilingual.pt",
)


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = CKPT_DIR / name
        if dest.is_file():
            print(f"checkpoint present: {dest}")
            continue
        print(f"downloading {name} from {REPO} ...")
        path = hf_hub_download(repo_id=REPO, filename=name)
        dest.write_bytes(Path(path).read_bytes())
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
