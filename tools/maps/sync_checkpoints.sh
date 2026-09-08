#!/usr/bin/env bash
# Sync Torch MAPS checkpoints into tools/maps/torch_models/.
#
# Preference order for source tree:
#   1. $TORCH_MAPS_SRC / $MAPS_SRC
#   2. $HOME/TORCH-MAPS or $HOME/MAPS (local checkouts)
#   3. shallow clone of github.com/Tom-Brenner/TORCH-MAPS into third_party/TORCH-MAPS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
DEST="$ROOT/torch_models"

pick_src() {
  local cand
  for cand in \
    "${TORCH_MAPS_SRC:-}" \
    "${MAPS_SRC:-}" \
    "$HOME/TORCH-MAPS" \
    "$HOME/MAPS" \
    "$REPO_ROOT/third_party/TORCH-MAPS"
  do
    [[ -n "$cand" && -d "$cand/torch_models" ]] || continue
    echo "$cand"
    return 0
  done
  return 1
}

SRC="$(pick_src || true)"
if [[ -z "${SRC}" ]]; then
  CLONE_DIR="$REPO_ROOT/third_party/TORCH-MAPS"
  echo "No local TORCH-MAPS/MAPS checkout found; cloning into $CLONE_DIR ..."
  mkdir -p "$(dirname "$CLONE_DIR")"
  if [[ -d "$CLONE_DIR/.git" ]]; then
    git -C "$CLONE_DIR" pull --ff-only || true
  else
    git clone --depth 1 https://github.com/Tom-Brenner/TORCH-MAPS.git "$CLONE_DIR"
  fi
  SRC="$CLONE_DIR"
fi

if [[ ! -f "$SRC/torch_models/timbuck_eng.pt" ]]; then
  echo "Missing $SRC/torch_models/timbuck_eng.pt" >&2
  exit 1
fi

mkdir -p "$DEST/ensemble_model"
cp -f "$SRC/torch_models/timbuck_eng.pt" "$DEST/"
if [[ -f "$SRC/torch_models/manifest.json" ]]; then
  cp -f "$SRC/torch_models/manifest.json" "$DEST/"
fi
if compgen -G "$SRC/torch_models/ensemble_model/"*.pt >/dev/null; then
  cp -f "$SRC/torch_models/ensemble_model/"*.pt "$DEST/ensemble_model/"
fi
echo "Synced MAPS checkpoints from $SRC → $DEST"
ls -la "$DEST/timbuck_eng.pt" "$DEST/ensemble_model" | head
