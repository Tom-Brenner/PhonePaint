#!/usr/bin/env python3
"""JSON-manifest MAPS alignment for inpaint_pipeline.py --maps.

Reads segment transcript JSON (``02_transcripts.json``), runs MAPS in the
active PhonePaint interpreter (no conda hop, no TensorFlow), and writes
per-segment alignments with PhonePaint ``canonical`` labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from textgrid import IntervalTier, TextGrid

MAPS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MAPS_ROOT.parents[1]
sys.path.insert(0, str(MAPS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from phone_labels import canonical_phone_label  # noqa: E402
from maps_torch.align_runtime import (  # noqa: E402
    align_wav,
    load_models,
    resolve_device,
)
from maps_torch.alignment import load_dictionary  # noqa: E402

DEFAULT_MAPS_CKPT = MAPS_ROOT / "torch_models" / "timbuck_eng.pt"
DEFAULT_MAPS_ENSEMBLE = MAPS_ROOT / "torch_models" / "ensemble_model"
_BUNDLED_CMUDICT = MAPS_ROOT / "nltk_data" / "corpora" / "cmudict" / "cmudict"
_NLTK_CMUDICT = Path.home() / "nltk_data" / "corpora" / "cmudict" / "cmudict"


def _default_cmudict() -> Path:
    env_value = os.environ.get("PHONEPAINT_CMUDICT")
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_file():
            return env_path
        # A stale env var (e.g. an unmounted drive) must not break a good checkout.
        print(
            f"PHONEPAINT_CMUDICT={env_value} is not a file — ignoring it.",
            file=sys.stderr,
        )
    for candidate in (_BUNDLED_CMUDICT, _NLTK_CMUDICT):
        if candidate.is_file():
            return candidate
    return _BUNDLED_CMUDICT


DEFAULT_MAPS_DICT = _default_cmudict()


def _pick_tiers(tg: TextGrid) -> tuple[IntervalTier | None, IntervalTier | None]:
    phones_tier = None
    words_tier = None
    for tier in tg.tiers:
        if not isinstance(tier, IntervalTier):
            continue
        lname = (tier.name or "").lower()
        if "phone" in lname or "segment" in lname:
            phones_tier = tier
        if "word" in lname:
            words_tier = tier
    return phones_tier, words_tier


def _tier_to_dict(tier: IntervalTier | None, *, phones: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if tier is None:
        return out
    for interval in tier:
        text = (interval.mark or "").strip()
        if not text:
            continue
        start = round(float(interval.minTime), 3)
        end = round(float(interval.maxTime), 3)
        if end <= start:
            continue
        item: dict = {"xmin": start, "xmax": end, "text": text}
        if phones:
            item["canonical"] = canonical_phone_label(text)
            item["inventory"] = "maps-timit61"
        out[str(len(out))] = item
    return out


def _textgrid_to_alignment(tg: TextGrid) -> dict:
    phones_tier, words_tier = _pick_tiers(tg)
    phones = _tier_to_dict(phones_tier, phones=True)
    if not phones:
        raise RuntimeError("MAPS produced no phone intervals")
    return {
        "words": _tier_to_dict(words_tier, phones=False),
        "phones": phones,
        "alignment": {
            "backend": "maps",
            "inventory": "maps-timit61",
        },
    }


def _load_transcripts(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected transcript object in {path}")
    texts: dict[str, str] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
        else:
            text = str(entry or "").strip()
        if not text:
            raise ValueError(f"Missing transcript text for {name} in {path}")
        texts[str(name)] = text
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio_dir", type=Path, required=True)
    ap.add_argument("--input_json", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Single .pt checkpoint or ensemble directory (default: timbuck_eng.pt)",
    )
    ap.add_argument(
        "--dict",
        dest="dictionary",
        type=Path,
        default=None,
        help=f"CMUdict path (default: PHONEPAINT_CMUDICT or {DEFAULT_MAPS_DICT})",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device for MAPS acoustic model (default: auto)",
    )
    ap.add_argument(
        "--no-sil",
        action="store_true",
        help="Disable silence padding at transcript ends",
    )
    ap.add_argument(
        "--no-interp",
        action="store_true",
        help="Disable boundary interpolation",
    )
    ap.add_argument(
        "--check-variants",
        action="store_true",
        help="Evaluate pronunciation variants (slow)",
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="Use tools/maps/torch_models/ensemble_model/ (10 checkpoints, median boundaries)",
    )
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_MAPS_ENSEMBLE if args.ensemble else DEFAULT_MAPS_CKPT
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(
            f"MAPS checkpoint not found: {model_path}\n"
            "Run tools/maps/sync_checkpoints.sh or copy torch_models/ from ~/MAPS."
        )

    dict_path = (args.dictionary or DEFAULT_MAPS_DICT).expanduser().resolve()
    if not dict_path.is_file():
        raise SystemExit(
            f"CMUdict not found: {dict_path}\n"
            "Set PHONEPAINT_CMUDICT or pass --dict."
        )

    device = resolve_device(args.device)
    print(f"MAPS device: {device}", flush=True)
    models = load_models(model_path, device)
    print(f"Loaded {len(models)} MAPS checkpoint(s) from {model_path}", flush=True)
    word2phone = load_dictionary(dict_path)

    audio_dir = args.audio_dir.expanduser().resolve()
    transcripts = _load_transcripts(args.input_json.expanduser().resolve())

    aligned: dict[str, dict] = {}
    for name, text in transcripts.items():
        wav_path = audio_dir / name
        if not wav_path.is_file():
            raise FileNotFoundError(f"Segment wav not found: {wav_path}")
        print(f"MAPS align: {name}", flush=True)
        tg = align_wav(
            wav_path,
            text,
            models=models,
            word2phone=word2phone,
            device=device,
            add_sil=not args.no_sil,
            use_interp=not args.no_interp,
            check_variants=args.check_variants,
        )
        aligned[name] = _textgrid_to_alignment(tg)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(aligned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json} ({len(aligned)} segment(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
