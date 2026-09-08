#!/usr/bin/env python3
"""
JSON-manifest driven MFA alignment using the "Libri" alignment pipeline.

MFA CLI resolution is delegated to ``mfa_align.resolve_mfa_bin`` (active conda env
first, then legacy ``mfa_env`` sidecar). Prefer installing MFA in the same env as
Whisper/PhonePaint when possible.

Why this exists:
- `mfa_align_libri.py` contains the correct model-pair + enhanced-dictionary (G2P) fallback logic.
- `mfa_align_cli.py` is a wrapper around `mfa_align.py` + `merge_mfa_json.py`, which does *not*
  use that newer logic.

This script reuses the core pipeline from `mfa_align_libri.py`, but swaps the input stage:
- instead of LibriSpeech tree + *.trans.txt, we accept a JSON manifest specifying audio + text.

Input JSON formats supported (auto-detected):
1) Dict mapping: { "audio.wav": {"text": "hello world"} , ... }
2) Dict mapping: { "audio.wav": "hello world", ... }
3) List of objects: [ {"audio": "audio.wav", "text": "hello world"}, ... ]
4) List of strings: [ "audio.wav", ... ]  (requires --sidecar_text_ext, default: .txt)
   In this mode, transcript is read from a sidecar file next to the audio, e.g. audio.wav + audio.txt

Typical usage:
  python3 mfa_align_json.py --audio_dir /path/to/audio --input_json manifest.json --output_json aligned.json \\
    --mfa_model_pairs english_mfa:english_mfa --use_g2p_fallback --dump_pre_g2p
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

from textgrid import IntervalTier, TextGrid  # pip install textgrid

from mfa_align import (
    _ffmpeg_convert,
    parse_model_pairs,
    run_incremental_alignment_with_model_pairs,
    detect_fail_words_from_textgrid,
    patch_fail_words_from_fallback,
    run_mfa,
    extract_words_from_texts,
    detect_oov_words,
    generate_g2p_pronunciations,
    create_enhanced_dictionary,
)


def _pick_tiers(tg: TextGrid) -> Tuple[IntervalTier | None, IntervalTier | None]:
    phones_tier = None
    words_tier = None
    for t in tg.tiers:
        if not isinstance(t, IntervalTier):
            continue
        lname = (t.name or "").lower()
        if ("phone" in lname) or ("segment" in lname):
            phones_tier = t
        if "word" in lname:
            words_tier = t
    return phones_tier, words_tier


def _build_flat_aligned(
    out_dir: Path, included_wavs: List[str], wav_to_key: Dict[str, str]
) -> Dict[str, dict]:
    """Create JSON with top-level words and phones (no nesting)."""
    aligned: Dict[str, dict] = {}

    for wav_name in included_wavs:
        tg_path = out_dir / Path(wav_name).with_suffix(".TextGrid")
        if not tg_path.exists():
            continue

        tg = TextGrid()
        tg.read(tg_path)
        phones_tier, words_tier = _pick_tiers(tg)
        if words_tier is None:
            continue

        words_dict: Dict[str, dict] = {}
        phones_dict: Dict[str, dict] = {}

        for w in words_tier:
            w_text = (w.mark or "").strip()
            if not w_text:
                continue
            words_dict[str(len(words_dict))] = {
                "xmin": round(w.minTime, 3),
                "xmax": round(w.maxTime, 3),
                "text": w_text,
            }

        if phones_tier is not None:
            for p in phones_tier:
                p_text = (p.mark or "").strip()
                if not p_text:
                    continue
                phones_dict[str(len(phones_dict))] = {
                    "xmin": round(p.minTime, 3),
                    "xmax": round(p.maxTime, 3),
                    "text": p_text,
                }

        top_key = wav_to_key.get(wav_name, wav_name)
        aligned[top_key] = {"words": words_dict, "phones": phones_dict}

    return aligned


def _normalize_manifest(
    manifest: Any,
    *,
    audio_dir: Path,
    sidecar_text_ext: str,
) -> Dict[str, Dict[str, str]]:
    """
    Normalize supported manifest formats into:
      { key: {"audio": str, "text": str} }

    Where key is what will appear in the output JSON (typically the audio filename).
    """
    def read_sidecar_text(audio_path: Path) -> str:
        sidecar = audio_path.with_suffix(sidecar_text_ext)
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing sidecar transcript: {sidecar}")
        return sidecar.read_text(encoding="utf-8").strip()

    out: Dict[str, Dict[str, str]] = {}

    if isinstance(manifest, dict):
        for k, v in manifest.items():
            if isinstance(v, dict):
                text = v.get("text", "")
                audio = v.get("audio", k)
            else:
                text = str(v)
                audio = k
            if not str(text).strip():
                # allow empty text entries but skip them
                continue
            out[str(k)] = {"audio": str(audio), "text": str(text)}
        return out

    if isinstance(manifest, list):
        for item in manifest:
            if isinstance(item, str):
                audio = item
                audio_path = Path(audio)
                if not audio_path.is_absolute():
                    audio_path = audio_dir / audio_path
                text = read_sidecar_text(audio_path)
                out[item] = {"audio": item, "text": text}
            elif isinstance(item, dict):
                audio = item.get("audio") or item.get("path") or item.get("file")
                text = item.get("text")
                if audio is None:
                    raise ValueError(f"Manifest object missing 'audio': {item}")
                if text is None:
                    audio_path = Path(str(audio))
                    if not audio_path.is_absolute():
                        audio_path = audio_dir / audio_path
                    text = read_sidecar_text(audio_path)
                key = str(item.get("key") or audio)
                if not str(text).strip():
                    continue
                out[key] = {"audio": str(audio), "text": str(text)}
            else:
                raise ValueError(f"Unsupported manifest list item type: {type(item)}")
        return out

    raise ValueError(f"Unsupported manifest type: {type(manifest)}")


def _iter_unique_ids(keys: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """
    Yield (key, utt_id) pairs where utt_id is safe/unique for MFA corpus filenames.
    Uses audio stem as base, disambiguates with _{n} if needed.
    """
    seen: Dict[str, int] = {}
    for key in keys:
        base = Path(key).stem or "utt"
        n = seen.get(base, 0)
        seen[base] = n + 1
        utt_id = base if n == 0 else f"{base}_{n}"
        yield key, utt_id


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Align audio specified by a JSON manifest using the mfa_align_libri pipeline."
    )
    ap.add_argument("--audio_dir", type=Path, required=True, help="Base directory for relative audio paths.")
    ap.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Manifest JSON (supported formats described in module docstring).",
    )
    ap.add_argument("--output_json", type=Path, required=True, help="Output alignment JSON path.")
    ap.add_argument(
        "--wav_output",
        type=Path,
        help="Optional directory to save converted 16k mono WAVs (mirrors relative path under audio_dir when possible).",
    )
    ap.add_argument(
        "--sidecar_text_ext",
        default=".txt",
        help="Used only when manifest entries do not contain text (default: .txt).",
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Optional volume gain (dB) applied when converting audio to 16k mono WAV.",
    )

    ap.add_argument("-j", "--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--omp_threads",
        type=int,
        default=1,
        help="OpenMP/BLAS threads to use inside MFA/ffmpeg subprocesses (default: 1).",
    )

    # Must match `mfa_align.py`'s argument shape: nargs='*', action='append'
    # so `parse_model_pairs()` receives a list-of-lists.
    ap.add_argument("--mfa_model_pairs", nargs="*", action="append")
    ap.add_argument("--mfa_dict_fallbacks", nargs="+", default=["english_mfa"])
    ap.add_argument("--mfa_acoustic_fallbacks", nargs="+", default=["english_mfa"])

    ap.add_argument("--use_g2p_fallback", action="store_true",
                    help="After english_mfa, G2P only for utterances with fail/OOV words.")
    ap.add_argument("--g2p_model", default="english_us_mfa")
    ap.add_argument("--pre_g2p_output_json", type=Path, help="Where to write a pre-G2P snapshot JSON.")
    ap.add_argument("--dump_pre_g2p", action="store_true", help="Enable pre-G2P snapshot emission.")

    ap.add_argument("--fallback_tolerance_ms", type=float, default=50.0)
    ap.add_argument("--fail_word_eps_ms", type=float, default=10.0)
    ap.add_argument("--debug_keep_tmp", action="store_true")
    ap.add_argument("--update", action="store_true", help="Skip entries already present in output_json.")

    args = ap.parse_args()

    args.audio_dir = args.audio_dir.resolve()
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"--audio_dir not found: {args.audio_dir}")

    existing_keys: set[str] = set()
    if args.update and args.output_json.exists():
        try:
            existing = json.loads(args.output_json.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing_keys = set(existing.keys())
        except Exception:
            existing_keys = set()

    manifest_raw = json.loads(args.input_json.read_text(encoding="utf-8"))
    norm = _normalize_manifest(
        manifest_raw, audio_dir=args.audio_dir, sidecar_text_ext=args.sidecar_text_ext
    )

    if args.update and existing_keys:
        norm = {k: v for k, v in norm.items() if k not in existing_keys}
        if not norm:
            print("All entries already present in output_json. Nothing to do.")
            return 0

    model_pairs = parse_model_pairs(args)
    print(f"Using model pairs: {' → '.join([f'{a}+{d}' for a, d in model_pairs])}")

    # Prepare temp corpus
    if args.debug_keep_tmp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mfa_json_"))
        cleanup = False
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mfa_json_"))
        cleanup = True

    text_data: Dict[str, dict] = {}
    included_wavs: List[str] = []
    wav_to_key: Dict[str, str] = {}

    try:
        # Isolate MFA working dirs per run, but reuse cached pretrained models via symlink
        os.environ["MFA_FALLBACK_TOL"] = f"{max(0.0, (args.fallback_tolerance_ms / 1000.0)):.3f}"

        mfa_tmp = tmp_dir / "mfa_tmp"
        mfa_root = tmp_dir / "mfa_root"
        mfa_cache = mfa_root / "cache"
        mfa_tmp.mkdir(parents=True, exist_ok=True)
        mfa_root.mkdir(parents=True, exist_ok=True)
        mfa_cache.mkdir(parents=True, exist_ok=True)

        os.environ["MFA_TEMP_DIR"] = str(mfa_tmp)
        os.environ["MFA_ROOT"] = str(mfa_root)
        os.environ["MFA_ROOT_DIR"] = str(mfa_root)
        os.environ["MFA_CACHE_DIR"] = str(mfa_cache)

        pretrained_candidates = [
            Path.home() / "Documents" / "MFA" / "pretrained_models",
            Path.home() / ".local" / "share" / "Montreal-Forced-Alignment" / "pretrained_models",
        ]
        for cand in pretrained_candidates:
            if cand.exists():
                target = mfa_root / "pretrained_models"
                if not target.exists():
                    try:
                        target.symlink_to(cand, target_is_directory=True)
                    except Exception:
                        pass
                break

        omp_threads = max(1, int(args.omp_threads))
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(omp_threads))

        # Build the MFA corpus (wav + lab files)
        keys = list(norm.keys())
        for key, utt_id in _iter_unique_ids(keys):
            entry = norm[key]
            audio_rel = Path(entry["audio"])
            audio_path = audio_rel if audio_rel.is_absolute() else (args.audio_dir / audio_rel)
            if not audio_path.exists():
                print(f"[warn] missing audio: {audio_path} (key={key}); skipping")
                continue

            text = (entry.get("text") or "").strip()
            if not text:
                print(f"[warn] empty transcript for key={key}; skipping")
                continue

            wav_name = f"{utt_id}.wav"
            dst_wav = tmp_dir / wav_name
            _ffmpeg_convert(audio_path, dst_wav, gain_db=args.gain)

            (tmp_dir / f"{utt_id}.lab").write_text(text, encoding="utf-8")

            # Optional WAV export: mirror relative path under audio_dir when possible
            if args.wav_output:
                args.wav_output.mkdir(parents=True, exist_ok=True)
                try:
                    rel = audio_path.resolve().relative_to(args.audio_dir)
                    export = args.wav_output / rel
                    export = export.with_suffix(".wav")
                except Exception:
                    export = args.wav_output / f"{utt_id}.wav"
                export.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst_wav, export)

            text_data[wav_name] = {"text": text}
            included_wavs.append(wav_name)
            wav_to_key[wav_name] = key

        if not included_wavs:
            raise RuntimeError("No valid (audio,text) entries found after filtering.")

        pre_g2p_path = None
        if args.use_g2p_fallback and args.dump_pre_g2p:
            if args.pre_g2p_output_json is None:
                pre_g2p_path = args.output_json.with_name(args.output_json.stem + "_pre_g2p.json")
            else:
                pre_g2p_path = args.pre_g2p_output_json

        out_dir, _stats = run_incremental_alignment_with_model_pairs(
            tmp_dir,
            model_pairs,
            args.threads,
            text_data=text_data,
            use_g2p=False,
            g2p_model=args.g2p_model,
            pre_g2p_json_path=pre_g2p_path,
            included_fns=included_wavs,
        )

        # Always dump a fresh pre-G2P snapshot derived from TextGrids (same behavior as mfa_align_libri)
        if pre_g2p_path is not None:
            pre_aligned = _build_flat_aligned(out_dir, included_wavs, wav_to_key)
            pre_g2p_path.parent.mkdir(parents=True, exist_ok=True)
            pre_g2p_path.write_text(
                json.dumps(pre_aligned, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[pre-G2P] JSON snapshot written to: {pre_g2p_path}")

        # Detect fail_words per file, optionally run G2P-only fallback on a subset, and patch only fail segments
        eps_s = max(0.0, float(args.fail_word_eps_ms) / 1000.0)
        fail_words_by_wav: Dict[str, List[int]] = {}
        fail_wavs: List[str] = []
        for wav_name in included_wavs:
            tg_path = out_dir / Path(wav_name).with_suffix(".TextGrid")
            if not tg_path.exists():
                continue
            idxs = detect_fail_words_from_textgrid(tg_path, eps_s=eps_s)
            if idxs:
                fail_words_by_wav[wav_name] = idxs
                fail_wavs.append(wav_name)

        if args.use_g2p_fallback and fail_wavs:
            g2p_corpus = tmp_dir / "corpus_g2p_fail_only"
            g2p_corpus.mkdir(parents=True, exist_ok=True)

            text_data_fail: Dict[str, dict] = {}
            for wav_name in fail_wavs:
                src_wav = tmp_dir / wav_name
                dst_wav = g2p_corpus / wav_name
                src_lab = tmp_dir / Path(wav_name).with_suffix(".lab").name
                dst_lab = g2p_corpus / Path(wav_name).with_suffix(".lab").name

                if not dst_wav.exists():
                    try:
                        os.symlink(src_wav, dst_wav)
                    except Exception:
                        shutil.copy2(src_wav, dst_wav)
                if src_lab.exists() and not dst_lab.exists():
                    try:
                        os.symlink(src_lab, dst_lab)
                    except Exception:
                        shutil.copy2(src_lab, dst_lab)
                text_data_fail[wav_name] = text_data[wav_name]

            primary_acoustic, primary_dict = model_pairs[0]
            tmp_g2p_dir = g2p_corpus / "_g2p"
            tmp_g2p_dir.mkdir(exist_ok=True)

            all_words = extract_words_from_texts(text_data_fail)
            oov_words = detect_oov_words(all_words, primary_dict)
            g2p_out_dir = None
            if oov_words and generate_g2p_pronunciations(oov_words, args.g2p_model, tmp_g2p_dir):
                enhanced = create_enhanced_dictionary(
                    primary_dict,
                    tmp_g2p_dir / "g2p_pronunciations.dict",
                    tmp_g2p_dir,
                )
                if enhanced is not None and enhanced.exists():
                    g2p_out_dir = run_mfa(
                        g2p_corpus,
                        primary_acoustic,
                        str(enhanced),
                        args.threads,
                        out_dir=(g2p_corpus / "out_g2p"),
                        tag="g2p",
                    )

            if g2p_out_dir is not None:
                patch_fail_words_from_fallback(
                    out_dir,
                    g2p_out_dir,
                    fail_words_by_wav,
                    eps_s=eps_s,
                    convert_arpabet=False,
                )

        aligned = _build_flat_aligned(out_dir, included_wavs, wav_to_key)

        # Merge with existing output if --update was provided
        if args.update and args.output_json.exists():
            try:
                existing = json.loads(args.output_json.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing.update(aligned)
                    aligned = existing
            except Exception:
                pass

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(aligned, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] wrote {len(aligned)} entries to {args.output_json}")
        return 0
    finally:
        if cleanup:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"[keep] temp at {tmp_dir}")


if __name__ == "__main__":
    raise SystemExit(main())


