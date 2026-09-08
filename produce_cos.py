#!/usr/bin/env python3
"""Cosine similarity of ECAPA speaker embeddings for original/inferred demo pairs.

For every demo clip produced by ``run_demo.py`` (job primary outputs + cuts) and
every VoiceCraft baseline named in that script, embed the original and inferred
wavs with SpeechBrain ECAPA-TDNN and write speaker cosine similarity under
``outreach/cos_per_file.json``, keyed by the inferred filename.

Kinds:
  ``16k``      — PhonePaint 16 kHz (``basename_{tag}.wav``)
  ``16kFull``  — PhonePaint 16 kHz + full phone conditioning (``…FULL.wav``)
  ``24k``      — PhonePaint 24 kHz (``basename{sep}24_{tag}.wav``)
  ``VC``       — VoiceCraft (``*_VoiceCraft.wav``)
  ``SeedVC``   — Seed-VC SVC (``*_SeedVC.wav``; demo.html originals only)

``--npz_path`` (default ``outreach/ecapa_embeddings.npz``) is a float32 embedding
cache keyed by wav filename. Existing entries are reused; missing ones are
computed and the archive is overwritten at the end.

After writing ``outreach/cos_per_file.json``, refreshes the ``const COS = {…}``
block in ``outreach/demo.html``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_demo import (  # noqa: E402
    DROPBOX_ROOT,
    ECAPA_EMBEDDINGS_PATH,
    ECAPA_SOURCE,
    INPUT_DIR,
    JOBS,
    VARIANTS,
    VOICECRAFT_RECIPES,
    Job,
    _patch_demo_html_cos,
    original_filename,
    seedvc_clips_from_demo_html,
    seedvc_output_name,
    variant_filename,
)

COS_OUT_DEFAULT = SCRIPT_DIR / "outreach" / "cos_per_file.json"


@dataclass(frozen=True)
class Pair:
    original: str  # filename under --input_dir
    inferred: str
    kind: str  # 16k | 16kFull | 24k | VC | SeedVC


def iter_pairs(jobs: Sequence[Job] = JOBS) -> List[Pair]:
    pairs: List[Pair] = []
    seen: set[Tuple[str, str, str]] = set()

    def add(original: str, inferred: str, kind: str) -> None:
        key = (original, inferred, kind)
        if key in seen:
            return
        seen.add(key)
        pairs.append(Pair(original=original, inferred=inferred, kind=kind))

    def add_clip(basename: str, tag: str, *, both_rates: bool, sep24: str) -> None:
        orig16 = original_filename(basename, rate24=False)
        add(orig16, variant_filename(basename, tag, VARIANTS["16k"]), "16k")
        add(orig16, variant_filename(basename, tag, VARIANTS["16k_fc"]), "16kFull")
        if both_rates:
            orig24 = original_filename(basename, rate24=True, sep24=sep24)
            add(
                orig24,
                variant_filename(basename, tag, VARIANTS["24k"], sep24=sep24),
                "24k",
            )

    for job in jobs:
        both_rates = bool(job.both_rates)
        # Match run_demo: job primary always uses sep24="_"; cuts use job.sep24.
        add_clip(job.basename, job.tag, both_rates=both_rates, sep24="_")
        for cut in job.cuts:
            add_clip(cut.basename, cut.tag, both_rates=both_rates, sep24=job.sep24)

    for vc_name in sorted(VOICECRAFT_RECIPES):
        if not vc_name.endswith("_VoiceCraft.wav"):
            continue
        stem = vc_name[: -len("_VoiceCraft.wav")]
        add(f"{stem}.wav", vc_name, "VC")

    for clip in seedvc_clips_from_demo_html():
        add(clip.original, seedvc_output_name(clip.original), "SeedVC")

    return pairs


def load_embedding_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    data = np.load(path)
    cache: Dict[str, np.ndarray] = {}
    for key in data.files:
        vec = np.asarray(data[key], dtype=np.float32).reshape(-1)
        cache[key] = vec
        # Accept stem keys written by run_demo (basename without .wav).
        if not key.endswith(".wav"):
            cache[f"{key}.wav"] = vec
    return cache


def save_embedding_cache(path: Path, cache: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer filename keys; drop redundant stem duplicates when both exist.
    arrays: Dict[str, np.ndarray] = {}
    for key, vec in sorted(cache.items()):
        if not key.endswith(".wav"):
            wav_key = f"{key}.wav"
            if wav_key in cache:
                continue
            arrays[wav_key] = np.asarray(vec, dtype=np.float32)
        else:
            arrays[key] = np.asarray(vec, dtype=np.float32)
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def embed_wav(classifier, path: Path) -> np.ndarray:
    signal = classifier.load_audio(str(path))
    emb = classifier.encode_batch(signal)
    return emb.squeeze().detach().cpu().numpy().astype(np.float32)


def ensure_embeddings(
    names: Iterable[str],
    *,
    input_dir: Path,
    cache: Dict[str, np.ndarray],
    classifier,
) -> List[str]:
    """Embed any missing filenames into ``cache``. Returns names that were missing on disk."""
    missing_files: List[str] = []
    for name in sorted(set(names)):
        if name in cache:
            continue
        path = input_dir / name
        if not path.is_file():
            missing_files.append(name)
            continue
        print(f"  embed {name}")
        cache[name] = embed_wav(classifier, path)
    return missing_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--npz_path",
        type=Path,
        default=ECAPA_EMBEDDINGS_PATH,
        help=(
            "Compressed ECAPA embedding cache (default: outreach/ecapa_embeddings.npz). "
            "Reused when keys match wav filenames; overwritten with embeddings used here."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=INPUT_DIR,
        help=(
            "Directory containing original/inferred wavs "
            f"(default: ~/Dropbox/datasets). Relative paths resolve under {DROPBOX_ROOT}."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=COS_OUT_DEFAULT,
        help=f"Output JSON path (default: {COS_OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="List pairs / cache hits without embedding or writing JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.npz_path = args.npz_path.expanduser()
    if not args.npz_path.is_absolute():
        args.npz_path = SCRIPT_DIR / args.npz_path
    args.input_dir = args.input_dir.expanduser()
    if not args.input_dir.is_absolute():
        args.input_dir = DROPBOX_ROOT / args.input_dir
    args.out = args.out.expanduser()
    if not args.out.is_absolute():
        args.out = SCRIPT_DIR / args.out

    pairs = iter_pairs(JOBS)
    print(f"pairs: {len(pairs)} (from run_demo JOBS + VoiceCraft + Seed-VC recipes)")

    needed: List[str] = []
    for p in pairs:
        needed.append(p.original)
        needed.append(p.inferred)

    cache = load_embedding_cache(args.npz_path)
    print(f"cache: {args.npz_path} ({len(cache)} entr{'y' if len(cache) == 1 else 'ies'})")

    if args.dry_run:
        present = {n for n in needed if (args.input_dir / n).is_file() or n in cache}
        for p in pairs:
            ok = p.original in present and p.inferred in present
            flag = "ok" if ok else "missing"
            print(f"  [{flag}] {p.kind:8} {p.original}  <->  {p.inferred}")
        print(f"+ would write {args.out}")
        return

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    to_compute = [n for n in sorted(set(needed)) if n not in cache]
    classifier = None
    missing_on_disk: List[str] = []
    if to_compute:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"loading ECAPA ({ECAPA_SOURCE}) on {device}")
        classifier = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            run_opts={"device": device},
        )
        missing_on_disk = ensure_embeddings(
            to_compute, input_dir=args.input_dir, cache=cache, classifier=classifier,
        )
    else:
        print("all needed embeddings already in cache")

    results: Dict[str, dict] = {}
    skipped: List[Pair] = []
    for p in pairs:
        if p.original not in cache or p.inferred not in cache:
            skipped.append(p)
            continue
        results[p.inferred] = {
            "original": p.original,
            "inferred": p.inferred,
            "kind": p.kind,
            "cos": cosine(cache[p.original], cache[p.inferred]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(results)} pair(s))")
    _patch_demo_html_cos()

    # Persist embeddings for files we successfully embedded / reused.
    used = {name: cache[name] for name in sorted(set(needed)) if name in cache}
    if used:
        save_embedding_cache(args.npz_path, used)
        print(f"wrote {args.npz_path} ({len(used)} embeddings; overwritten)")

    if missing_on_disk:
        print(f"missing wavs under {args.input_dir} ({len(missing_on_disk)}):")
        for name in missing_on_disk:
            print(f"  - {name}")
    if skipped:
        print(f"skipped pairs ({len(skipped)}) due to missing embeddings/wavs:")
        for p in skipped:
            print(f"  - {p.kind}: {p.original} <-> {p.inferred}")


if __name__ == "__main__":
    main()
