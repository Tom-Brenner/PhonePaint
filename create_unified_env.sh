#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${PHONEPAINT_ENV_NAME:-PhonePaintUnified}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/tools/repro_pins.env"

if conda env list | awk -v name="$ENV_NAME" \
    '$1 == name { found=1 } END { exit !found }'; then
  conda env update -n "$ENV_NAME" -f "$ROOT/environment-unified.yml" --prune
else
  conda env create -n "$ENV_NAME" -f "$ROOT/environment-unified.yml"
fi

# BFA 1.1.5's core uses torchaudio.load and accepts waveform tensors. Its
# declared torchcodec>=0.9.1 dependency is not imported by that path and is
# ABI-incompatible with the Torch 2.5 stack required by PhonePaint.
conda run -n "$ENV_NAME" python -m pip install \
  --no-deps "bournemouth-forced-aligner==${BFA_VERSION}"

# The maintained fork adds an espeak data-path API needed by espeakng-loader.
# Keep phonemizer's distribution metadata (required by BFA) while replacing its
# import-compatible implementation with the fork.
conda run -n "$ENV_NAME" python -m pip install \
  --force-reinstall --no-deps "phonemizer-fork==${PHONEMIZER_FORK_VERSION}"

# FALCON vendor + checkpoints live under tools/falcon/ (align runs in this same env).
bash "$ROOT/tools/falcon/clone_vendor.sh"
conda run -n "$ENV_NAME" python "$ROOT/tools/falcon/patch_panphon.py"
mkdir -p "$ROOT/tools/falcon/pretrained_models"
FALCON_HF_REPO="${FALCON_HF_REPO}" \
  conda run -n "$ENV_NAME" python "$ROOT/tools/falcon/download_checkpoints.py"
ln -sfn "$ROOT/tools/falcon/pretrained_models" \
  "$ROOT/tools/falcon/vendor/FALCON/pretrained_models"

# MAPS Torch checkpoints (align runs in this same env via --maps).
bash "$ROOT/tools/maps/sync_checkpoints.sh" || true

# Scripts run this checkout directly. Remove stale editable ESPnet metadata,
# whose 2020-era strict pins do not describe the modern inference-only stack.
conda run -n "$ENV_NAME" python -m pip uninstall -y espnet >/dev/null 2>&1 || true

conda run -n "$ENV_NAME" python -c \
  'from unified_speech import configure_espeak; configure_espeak(); import bournemouth_aligner, torch, torchaudio, transformers, panphon, pydub, python_speech_features, natsort; from phonemizer import phonemize; assert phonemize("test", language="en-us", backend="espeak").strip(); print(f"ready: torch={torch.__version__}, torchaudio={torchaudio.__version__}, transformers={transformers.__version__}")'
