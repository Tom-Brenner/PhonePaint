#!/usr/bin/env bash
# Create/update the inference conda env: PhonePaint
#   - PhonePaint inpaint + ESPnet (editable from this checkout)
#   - Whisper ASR (transformers) on CUDA
#   - Torch MAPS (in-process; --maps)
# MFA is NOT installed here: --mfa shells out to the mfa_env sidecar
# (./create_mfa_envs.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ENV_NAME="${PHONEPAINT_ENV_NAME:-PhonePaint}"
WANT_PY="3.12"

env_exists() {
  conda env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'
}

# environment.yml pins python 3.12; `conda env update` cannot migrate an env
# built on another python, so recreate instead of failing mid-solve.
if env_exists; then
  cur_py="$(conda run -n "$ENV_NAME" python -c \
    'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "")"
  if [[ "$cur_py" != "$WANT_PY" ]]; then
    echo "Existing $ENV_NAME uses python ${cur_py:-unknown}; recreating for python $WANT_PY ..."
    conda env remove -n "$ENV_NAME" -y
  fi
fi

# Flexible priority: the pytorch-channel CUDA build is excluded under strict.
export CONDA_CHANNEL_PRIORITY=flexible

if env_exists; then
  echo "Updating $ENV_NAME from environment.yml ..."
  conda env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
else
  echo "Creating $ENV_NAME from environment.yml ..."
  conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi

# Editable ESPnet from this checkout (PyPI espnet differs).
conda run -n "$ENV_NAME" python -m pip uninstall -y espnet >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" python -m pip install -e "$ROOT" --no-deps

# Pip must not replace the CUDA pytorch-channel builds.
conda install -n "$ENV_NAME" -c pytorch -c nvidia -c conda-forge \
  'pytorch=2.5.1=py3.12_cuda12.4*' pytorch-cuda=12.4 torchaudio=2.5.1 -y

# Checkpoints are kept out of git. tools/fetch_assets.py restores them from the
# Hugging Face repo and covers both weights/best.pt and the MAPS torch_models.
# That repo is private for now, so it needs a read token; sync_checkpoints.sh is
# kept as a no-token fallback that covers the MAPS half from public TORCH-MAPS.
# Non-fatal: the env is still usable for --mfa. But the result is gated at the
# end of this script rather than swallowed, because a silent miss here leaves
# the smoke tests skipping while unittest still prints OK.
fetch_assets() { conda run -n "$ENV_NAME" python "$ROOT/tools/fetch_assets.py" "$@"; }

fetch_assets || true
if ! fetch_assets --check >/dev/null 2>&1; then
  echo "Assets still incomplete; trying the TORCH-MAPS fallback ..."
  bash "$ROOT/tools/maps/sync_checkpoints.sh" || true
fi

ASSETS_OK=1
fetch_assets --check || ASSETS_OK=0

conda run -n "$ENV_NAME" python -c "
import torch, torchaudio, transformers, natsort, python_speech_features, textgrid, pydub, soxr
print(
    f'ready: torch={torch.__version__} cuda_available={torch.cuda.is_available()} '
    f'transformers={transformers.__version__}'
)
"

echo ""
echo "$ENV_NAME ready (Whisper + MAPS + PhonePaint)."
echo "  conda activate $ENV_NAME"
echo "For --mfa, also run: ./create_mfa_envs.sh   # mfa_env sidecar (Kaldi MFA CLI)"
echo "EnCodec: ./tools/setup_voicecraft.sh"

if [[ "$ASSETS_OK" -eq 0 ]]; then
  echo ""
  echo "########################################################################"
  echo "## WARNING: model checkpoints are INCOMPLETE                          ##"
  echo "########################################################################"
  echo "The --check listing above names the missing files. Without them the"
  echo "pipeline cannot run: weights/best.pt is required by both aligners, and"
  echo "tools/maps/torch_models/ is required by --maps."
  echo "  Retry:  python tools/fetch_assets.py"
  echo "The Hugging Face repo is private, so that needs a read token:"
  echo "  export HF_TOKEN=hf_...      # or: hf auth login"
  echo "Offline alternative, copying from another checkout that has them:"
  echo "  python tools/fetch_assets.py --from /path/to/PhonePaint"
  echo "Smoke tests needing them will SKIP (and unittest still prints OK) unless"
  echo "you run them with PHONEPAINT_REQUIRE_ASSETS=1, which turns skips into"
  echo "failures."
  echo "########################################################################"
fi
