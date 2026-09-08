#!/usr/bin/env bash
# Clone VoiceCraft sources + download encodec_4cb2048_giga.th for PhonePaint.
# Default location: <PhonePaint>/third_party/VoiceCraft (override with VOICECRAFT_ROOT).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/tools/repro_pins.env"

VOICECRAFT_ROOT="${VOICECRAFT_ROOT:-$ROOT/third_party/VoiceCraft}"
REPO="${VOICECRAFT_GIT_REPO}"
REF="${VOICECRAFT_GIT_REF}"
HF_REPO="${VOICECRAFT_ENCODEC_HF_REPO}"
CKPT_NAME="${VOICECRAFT_ENCODEC_FILE}"
CKPT_PATH="$VOICECRAFT_ROOT/pretrained_models/$CKPT_NAME"

echo "VoiceCraft root: $VOICECRAFT_ROOT"

if [[ -d "$VOICECRAFT_ROOT/.git" ]]; then
  git -C "$VOICECRAFT_ROOT" fetch --depth 1 origin "$REF"
  git -C "$VOICECRAFT_ROOT" checkout -q FETCH_HEAD
else
  mkdir -p "$(dirname "$VOICECRAFT_ROOT")"
  git clone --depth 1 --branch "$REF" "$REPO" "$VOICECRAFT_ROOT"
fi

mkdir -p "$VOICECRAFT_ROOT/pretrained_models"

if [[ -f "$CKPT_PATH" ]]; then
  echo "EnCodec checkpoint present: $CKPT_PATH"
else
  echo "Downloading $CKPT_NAME from Hugging Face ($HF_REPO) ..."
  download_py=(python3)
  if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '$1 == "PhonePaint" { found=1 } END { exit !found }'; then
      download_py=(conda run -n PhonePaint python)
    fi
  fi
  CKPT_PATH="$CKPT_PATH" HF_REPO="$HF_REPO" CKPT_NAME="$CKPT_NAME" \
    "${download_py[@]}" -c '
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
dest = Path(os.environ["CKPT_PATH"])
dest.parent.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=os.environ["HF_REPO"], filename=os.environ["CKPT_NAME"])
dest.write_bytes(Path(path).read_bytes())
print(f"  -> {dest}")
'
fi

AUDIOCRAFT_DIR="$VOICECRAFT_ROOT/src/audiocraft"
if [[ -d "$AUDIOCRAFT_DIR/.git" ]]; then
  echo "audiocraft sources present: $AUDIOCRAFT_DIR"
else
  echo "Cloning audiocraft ($AUDIOCRAFT_GIT_REF) into $AUDIOCRAFT_DIR ..."
  mkdir -p "$(dirname "$AUDIOCRAFT_DIR")"
  git clone --filter=blob:none "$AUDIOCRAFT_GIT_REPO" "$AUDIOCRAFT_DIR"
  git -C "$AUDIOCRAFT_DIR" checkout -q "$AUDIOCRAFT_GIT_REF"
fi

if [[ ! -f "$AUDIOCRAFT_DIR/audiocraft/modules/seanet.py" ]]; then
  echo "ERROR: audiocraft SEANet sources missing under $AUDIOCRAFT_DIR" >&2
  exit 1
fi

echo "VoiceCraft ready."
echo "  export VOICECRAFT_ROOT=$VOICECRAFT_ROOT"
echo "(PhonePaint defaults to <repo>/third_party/VoiceCraft when VOICECRAFT_ROOT is unset.)"
