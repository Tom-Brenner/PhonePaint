#!/usr/bin/env python3
"""crisper_transcribe.py – Produce verbatim segments + per‑word confidences
with CrisperWhisper.

BUGFIX 2025‑08‑06
-----------------
Previous patch used `task="transcribe"` directly in the pipeline call; the
forked `AutomaticSpeechRecognitionPipeline` does **not** accept that kwarg and
raised `TypeError: _sanitize_parameters() got an unexpected keyword argument
'task'`.

Fix: pass the directive inside **`generate_kwargs`** (exactly what the original
script did), and keep the `Path → str` cast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import pipeline  # forked version from nyrahealth
# --- Whisper tuple-safe split_by_batch_index ---------------------------------
from transformers.models.whisper import generation_whisper as _gw

def _patched_split_by_batch_index(values, key, batch_idx):
    """
    Clone of the original HF helper with added tuple handling.

    • key == 'scores'   → list of tensors (same as upstream)
    • values is tuple   → tuple of tensors, each sent to CPU
    • fallback          → single tensor path (upstream behaviour)
    """
    # original special-case: list of per-step score tensors
    if key == "scores":
        return [v[batch_idx].cpu() for v in values]

    # new: some outputs (e.g. past_key_values, logits) come back as tuples
    if isinstance(values, tuple):
        return tuple(
            v[batch_idx].cpu() if hasattr(v, "cpu") else v[batch_idx]
            for v in values
        )

    # default path – single tensor → CPU
    return values[batch_idx].cpu()

# hot-swap the helper inside transformers
_gw.split_by_batch_index = _patched_split_by_batch_index
# -----------------------------------------------------------------------------



###############################################################################
# Helpers
###############################################################################

def create_audio_dataset(wavs: List[Path]):
    """
    Create a generator that yields audio file paths.
    This helps the pipeline optimize batching and reduces the sequential processing warning.
    """
    for wav in wavs:
        yield {"path": str(wav), "filename": wav.name}

def transcribe_optimized(
    wavs: List[Path],
    model_id: str,
    lang: str | None,
    device: int | str,
    batch_size: int = 8,
) -> Dict[str, Dict]:
    """
    Optimized transcription using dataset iteration for better GPU utilization.
    This approach should eliminate the sequential processing warning.
    """
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16,
        chunk_length_s=8,
        stride_length_s=1,
        return_timestamps=False,
        batch_size=batch_size,
    )

    gen_kwargs = {"task": "transcribe"}
    if lang:
        gen_kwargs["language"] = lang

    results: Dict[str, Dict] = {}
    
    # Create dataset-like iterator
    audio_dataset = list(create_audio_dataset(wavs))
    
    with torch.inference_mode():
        # Process using the dataset approach with progress bar
        for i in tqdm(range(0, len(audio_dataset), batch_size), desc="Processing batches"):
            batch = audio_dataset[i:i+batch_size]
            audio_paths = [item["path"] for item in batch]
            
            # Process batch
            batch_results = asr(audio_paths, generate_kwargs=gen_kwargs)
            
            # Handle results (single or multiple)
            if isinstance(batch_results, list):
                for item, result in zip(batch, batch_results):
                    txt = result["text"].strip() if isinstance(result, dict) else str(result).strip()
                    results[item["filename"]] = {"text": txt}
            else:
                # Single result case
                txt = batch_results["text"].strip()
                results[batch[0]["filename"]] = {"text": txt}
    
    return results

def transcribe(
    wavs: List[Path],
    model_id: str,
    lang: str | None,
    device: int | str,
    batch_size: int = 8,  # Configurable batch size
) -> Dict[str, Dict]:
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,              # 0 for GPU, -1 for CPU
        torch_dtype=torch.float16,
        chunk_length_s=8,
        stride_length_s=1,
        return_timestamps=False,
        #return_segments=False,
        batch_size=batch_size,      # Use configurable batch size
    )

    gen_kwargs = {"task": "transcribe"}
    if lang:
        gen_kwargs["language"] = lang

    # Create dataset-like structure for efficient batching
    # Convert paths to strings as required by the pipeline
    audio_paths = [str(wav) for wav in wavs]
    
    results: Dict[str, Dict] = {}
    with torch.inference_mode():                   # one global context
        # Process in batches for GPU parallelization
        batch_results = asr(
            audio_paths, 
            generate_kwargs=gen_kwargs,
            batch_size=batch_size
        )
        
        # Handle both single result and batch results
        if isinstance(batch_results, list):
            # Multiple files processed in batch
            for wav_path, result in zip(wavs, batch_results):
                txt = result["text"].strip() if isinstance(result, dict) else str(result).strip()
                results[wav_path.name] = {"text": txt}
        else:
            # Single file result
            txt = batch_results["text"].strip()
            results[wavs[0].name] = {"text": txt}

    return results


###############################################################################
# Main CLI
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--model", default="nyrahealth/CrisperWhisper")
    ap.add_argument("--language", "-l", default="en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for GPU parallelization (default: 8)")
    ap.add_argument("--use_optimized", action="store_true",
                    help="Use optimized dataset-based processing (recommended for eliminating warnings)")
    args = ap.parse_args()

    device_idx = args.device_idx if args.device.startswith("cuda") else -1
    # (-1 tells transformers to run on CPU)
    # ─────────────────────────────────────────────────────────────────────────

    wavs = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        raise SystemExit("[!] No .wav files found in input_dir")

    # Use optimized version if requested, otherwise use standard version
    # The optimized version processes audio files in batches and should eliminate
    # the "running tasks sequentially" warning by properly preparing the dataset
    if args.use_optimized:
        print(f"[i] Using optimized batched processing (batch_size={args.batch_size})")
        res = transcribe_optimized(wavs, args.model, args.language, device_idx, args.batch_size)
    else:
        print(f"[i] Using standard processing (batch_size={args.batch_size})")
        res = transcribe(wavs, args.model, args.language, device_idx, args.batch_size)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("[✓] Wrote", args.output_json)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
