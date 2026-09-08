#!/usr/bin/env bash
# Create/update the conda envs used by: python inpaint_pipeline.py --mfa ...
#   mfa_env         — Montreal Forced Aligner CLI + Kaldi (sidecar binary for --mfa)
#   crisperWhisper  — optional CrisperWhisper ASR (--mfa --Crisper only)
#
# Default --mfa ASR is in-process Whisper in PhonePaintUnified; MFA align runs the
# JSON wrapper in that same interpreter and only shells out to mfa_env's ``mfa``.
# Inference still runs in the active interpreter (PhonePaint or PhonePaintUnified).
#
# Usage:
#   ./create_mfa_envs.sh              # mfa_env only (default)
#   ./create_mfa_envs.sh --crisper    # also create/update crisperWhisper
#   ./create_mfa_envs.sh --no-crisper # explicit mfa_env only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/tools/repro_pins.env"

WITH_CRISPER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --crisper) WITH_CRISPER=1; shift ;;
    --no-crisper) WITH_CRISPER=0; shift ;;
    -h|--help)
      cat <<'EOF'
Create/update the conda envs used by: python inpaint_pipeline.py --mfa ...

  ./create_mfa_envs.sh              # mfa_env only (default)
  ./create_mfa_envs.sh --crisper    # also create/update crisperWhisper
  ./create_mfa_envs.sh --no-crisper # explicit mfa_env only
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

create_or_update() {
  local name="$1"
  local yml="$2"
  if conda env list | awk -v n="$name" '$1 == n { found=1 } END { exit !found }'; then
    conda env update -n "$name" -f "$yml" --prune
  else
    conda env create -n "$name" -f "$yml"
  fi
}

create_or_update mfa_env "$ROOT/environment-mfa.yml"

# Prefetch English MFA models used by tools/mfa (fail loudly — required for --mfa).
conda run -n mfa_env mfa model download acoustic english_mfa
conda run -n mfa_env mfa model download dictionary english_mfa
# g2p has no "english_mfa"; tools/mfa defaults to english_us_mfa.
conda run -n mfa_env mfa model download g2p english_us_mfa
# Optional ARPAbet pair (still useful for some fallbacks / tooling).
conda run -n mfa_env mfa model download acoustic english_us_arpa || true
conda run -n mfa_env mfa model download dictionary english_us_arpa || true
conda run -n mfa_env mfa model download g2p english_us_arpa || true

conda run -n mfa_env python -c \
  'from textgrid import TextGrid; import montreal_forced_aligner; print("mfa_env ready")'
# Confirm sox is on the MFA env PATH (sidecar third-party check).
conda run -n mfa_env bash -lc 'command -v sox && sox --version | head -1'

if [[ "$WITH_CRISPER" -eq 1 ]]; then
  create_or_update crisperWhisper "$ROOT/environment-crisper.yml"
  # CrisperWhisper requires the nyrahealth transformers fork (pinned commit).
  conda run -n crisperWhisper python -m pip install --upgrade \
    "${CRISPER_TRANSFORMERS_GIT}"
  conda run -n crisperWhisper python -c \
    'import torch, transformers; print(f"crisperWhisper ready: torch={torch.__version__}, transformers={transformers.__version__}")'
else
  echo "Skipping crisperWhisper (pass --crisper to create it)."
fi

echo "MFA backends ready. Activate PhonePaint/PhonePaintUnified for inference, then:"
echo "  python inpaint_pipeline.py --mfa --input ... --output ... --checkpoint ..."
if [[ "$WITH_CRISPER" -eq 1 ]]; then
  # --Crisper exists only on the legacy CLI, which is no longer in the repo.
  echo "  crisperWhisper is wired into run_inpaint_pipeline.py (legacy, untracked)."
fi
