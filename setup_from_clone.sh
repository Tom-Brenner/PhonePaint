#!/usr/bin/env bash
# Bootstrap PhonePaint on a fresh machine after `git clone`.
#
#   ./setup_from_clone.sh                 # PhonePaint env + VoiceCraft EnCodec
#   ./setup_from_clone.sh --mfa           # also mfa_env sidecar (needed for --mfa)
#   ./setup_from_clone.sh --mfa --crisper
#   ./setup_from_clone.sh --skip-voicecraft
#
# System packages this script cannot install for you (must exist beforehand):
#   - conda / mamba (Miniconda or Miniforge)
#   - CUDA 12.x driver (for GPU Torch builds in environment.yml)
#
# PhonePaint model checkpoints (.pt) are not downloaded here; pass --checkpoint
# when running the pipeline. ffmpeg/sox come from the conda env.
# EnCodec weights: ./tools/setup_voicecraft.sh (unless --skip-voicecraft).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WITH_LEGACY_MFA=0
WITH_CRISPER=0
SKIP_VOICECRAFT=0

usage() {
  cat <<'EOF'
Bootstrap PhonePaint on a fresh machine after `git clone`.

  ./setup_from_clone.sh                 # PhonePaint (Whisper+MAPS+PP) + EnCodec
  ./setup_from_clone.sh --mfa [--crisper]   # also mfa_env sidecar for --mfa
  ./setup_from_clone.sh --skip-voicecraft

System packages this script cannot install for you (must exist beforehand):
  - conda / mamba (Miniconda or Miniforge)
  - CUDA 12.x driver

PhonePaint model checkpoints (.pt) are not downloaded here.
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mfa) WITH_LEGACY_MFA=1; shift ;;
    --crisper) WITH_CRISPER=1; WITH_LEGACY_MFA=1; shift ;;
    --skip-voicecraft) SKIP_VOICECRAFT=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    echo "See preamble of setup_from_clone.sh for system prerequisites." >&2
    exit 1
  fi
}

echo "== Checking system prerequisites =="
need_cmd conda
need_cmd git
echo "  conda=$(command -v conda)"

echo ""
echo "== Creating/updating PhonePaint =="
bash "$ROOT/create_phonepaint_env.sh"

if [[ "$WITH_LEGACY_MFA" -eq 1 ]]; then
  echo ""
  echo "== MFA sidecar env(s) for --mfa =="
  mfa_args=()
  if [[ "$WITH_CRISPER" -eq 1 ]]; then
    mfa_args+=(--crisper)
  else
    mfa_args+=(--no-crisper)
  fi
  bash "$ROOT/create_mfa_envs.sh" "${mfa_args[@]}"
fi

if [[ "$SKIP_VOICECRAFT" -eq 0 ]]; then
  echo ""
  echo "== VoiceCraft EnCodec (third_party/VoiceCraft) =="
  bash "$ROOT/tools/setup_voicecraft.sh"
  export VOICECRAFT_ROOT="${VOICECRAFT_ROOT:-$ROOT/third_party/VoiceCraft}"
fi

echo ""
echo "Code-ready. Next:"
echo "  conda activate PhonePaint"
echo "  export VOICECRAFT_ROOT=${VOICECRAFT_ROOT:-$ROOT/third_party/VoiceCraft}"
echo "  python inpaint_pipeline.py --mfa --input … --output … --checkpoint …"
echo "  # or: python inpaint_pipeline.py --maps ..."
