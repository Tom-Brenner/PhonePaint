#!/usr/bin/env python3
"""Fetch the model checkpoints that are deliberately kept out of git.

A `git clone` of PhonePaint is ~26 MB and contains no weights: `weights/best.pt`
is blocked by the top-level `.gitignore` `*.pt` rule and the MAPS checkpoints by
`tools/maps/.gitignore`. This script puts them back.

Two backends:

  --from DIR   copy from another PhonePaint checkout or asset tree. Offline, and
               useful when a machine already has the files somewhere.
  (default)    download from the Hugging Face model repo. Needs read access; the
               repo is private for now, so export HF_TOKEN or `hf auth login`.

Paths in the HF repo mirror the paths in this checkout exactly, so the mapping
is 1:1 in both directions and nothing has to be renamed on arrival.

Usage:
  python tools/fetch_assets.py                      # download whatever is missing
  python tools/fetch_assets.py --check              # report only, exit 1 if short
  python tools/fetch_assets.py --from ~/PhonePaint  # copy from a local checkout
  python tools/fetch_assets.py --force              # re-fetch even if present
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "Tom-Brenner/phonepaint-assets"

# Repo-root-relative. Same strings are used as path_in_repo on the Hub.
ASSETS: tuple[str, ...] = (
    "weights/best.pt",
    "tools/maps/torch_models/timbuck_eng.pt",
    *(
        f"tools/maps/torch_models/ensemble_model/timbuck_eng_{i}.pt"
        for i in range(1, 11)
    ),
)


def _present(path: Path) -> bool:
    """A truncated or zero-byte file counts as missing, not as present."""
    return path.is_file() and path.stat().st_size > 0


def _missing(dest_root: Path) -> list[str]:
    return [rel for rel in ASSETS if not _present(dest_root / rel)]


def _report(dest_root: Path) -> int:
    missing = _missing(dest_root)
    for rel in ASSETS:
        path = dest_root / rel
        if _present(path):
            print(f"  ok      {rel}  ({path.stat().st_size / 2**20:.1f} MiB)")
        else:
            print(f"  MISSING {rel}")
    if missing:
        print(f"\n{len(missing)} of {len(ASSETS)} assets missing under {dest_root}")
        return 1
    print(f"\nall {len(ASSETS)} assets present under {dest_root}")
    return 0


def _fetch_local(source_root: Path, dest_root: Path, wanted: list[str]) -> None:
    for rel in wanted:
        src = source_root / rel
        if not _present(src):
            raise SystemExit(f"source is missing {rel}: {src}")
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied  {rel}  ({dst.stat().st_size / 2**20:.1f} MiB)")


def _fetch_hf(
    repo_id: str,
    revision: str | None,
    token: str | None,
    dest_root: Path,
    wanted: list[str],
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "huggingface_hub is required for the download backend "
            "(it ships with the PhonePaint env). Use --from DIR to copy "
            f"from a local checkout instead. Import error: {exc}"
        ) from exc

    from huggingface_hub.errors import HfHubHTTPError

    for rel in wanted:
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                revision=revision,
                token=token,
                repo_type="model",
            )
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"failed to download {rel} from {repo_id}: {exc}\n"
                "If the repo is private, export HF_TOKEN or run `hf auth login` "
                "with a token that has read access."
            ) from exc
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy out of the shared HF cache: the pipeline expects real files at
        # these paths, and a cache entry can be garbage-collected underneath us.
        shutil.copy2(cached, dst)
        print(f"  fetched {rel}  ({dst.stat().st_size / 2**20:.1f} MiB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Report which assets are present and exit 1 if any are missing",
    )
    ap.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=None,
        metavar="DIR",
        help="Copy from this local checkout instead of downloading",
    )
    ap.add_argument(
        "--repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face model repo (default: {DEFAULT_HF_REPO})",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit on the Hub (default: the repo default branch)",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token; otherwise HF_TOKEN or a cached `hf auth login` is used",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT,
        help=f"Checkout to populate (default: {REPO_ROOT})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch assets that are already present",
    )
    args = ap.parse_args()

    dest_root = args.dest.expanduser().resolve()

    if args.check:
        return _report(dest_root)

    wanted = list(ASSETS) if args.force else _missing(dest_root)
    if not wanted:
        print(f"all {len(ASSETS)} assets already present under {dest_root}")
        return 0

    print(f"{len(wanted)} asset(s) to fetch into {dest_root}")
    if args.source is not None:
        source_root = args.source.expanduser().resolve()
        if source_root == dest_root:
            raise SystemExit("--from source and --dest are the same directory")
        print(f"source: {source_root} (local copy)\n")
        _fetch_local(source_root, dest_root, wanted)
    else:
        print(f"source: {args.repo} (Hugging Face)\n")
        _fetch_hf(args.repo, args.revision, args.token, dest_root, wanted)

    still_missing = _missing(dest_root)
    if still_missing:
        print("\nstill missing after fetch:", ", ".join(still_missing))
        return 1
    print(f"\nall {len(ASSETS)} assets present under {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
