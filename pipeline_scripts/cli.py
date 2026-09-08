"""Command-line parsing and shared inference configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    CONTEXT_FRAMES,
    DEFAULT_ASR_MODEL,
    ENCODEC_16K_CKPT,
    OUTPUT_FS,
    STAGES,
)
from .speech import ALIGN_THREADS_SEGMENTS, parse_align_threads
from .spans import add_phone_map_args, validate_phone_map

PIPELINE_DESCRIPTION = """
PhonePaint inpaint pipeline (entry: inpaint_pipeline.py)

16 kHz conditional inpainting only (infer_phone_ec_cond.py + VoiceCraft EnCodec).

Stages (always this order):
  segment → transcribe (Whisper) → align (MFA|MAPS) → inpaint → stitch

Required speech backend (pick one):
  --mfa     Whisper + Montreal Forced Aligner
  --maps    Whisper + MAPS (optional --maps-ensemble)

Segments are only for ASR/align. Spans are remapped onto each original source
wav; inpainting runs once per source file (not per segment).

Workspace (default work/<input_stem>/):
  00_input/  01_segments/  02_transcripts.json  03_aligned.json
  04_spans/  05_inpainted/  06_stitched.wav  manifest.json  stage_stats.json
"""

PIPELINE_EPILOG = """
examples:
  # List every flag (this help text):
  python inpaint_pipeline.py --help

  # MFA, single file:
  python inpaint_pipeline.py --mfa \\
    --input in.wav --output out.wav --checkpoint model.pt \\
    --phones_in s --phones_out th

  # MAPS, resume from align, stop before inpaint:
  python inpaint_pipeline.py --maps --resume --from-stage align --to-stage align \\
    --input in.wav --output out.wav --checkpoint model.pt \\
    --phones_in s --phones_out th

  # MFA with one Kaldi job (often faster on short single files):
  python inpaint_pipeline.py --mfa --align-threads 1 \\
    --input in.wav --output out.wav --checkpoint model.pt \\
    --phones_in s --phones_out th

code map:
  pipeline_scripts/cli.py           flags (this --help)
  pipeline_scripts/coordinator.py   runs the stages
  pipeline_scripts/speech.py        segment / Whisper / MFA / MAPS
  pipeline_scripts/inference.py     spans + infer + stitch
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inpaint_pipeline.py",
        description=PIPELINE_DESCRIPTION,
        epilog=PIPELINE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Input WAV file or directory of WAVs "
             "(ignored / not required with --batch_manifest)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final stitched output WAV (16 kHz) "
             "(ignored / not required with --batch_manifest)",
    )
    p.add_argument("--work-dir", type=Path, default=None, help="Workspace (default: work/<input_stem>)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint (.pt) from train_phone_ec_cond.py",
    )
    add_phone_map_args(p, required=False)
    p.add_argument(
        "--batch_manifest", type=Path, default=None,
        help='JSON with a top-level "pipeline_jobs" list of '
             '{"input", "output", "phones_in", "phones_out"[, "ext", "force_ext", '
             '"work_dir", "resume", "from_stage", "force"]} dicts. When set, every '
             "job shares --checkpoint/--full_phone_conditioning/"
             "--windowed_attention/etc; segment/transcribe/align run per work "
             "dir as needed, then all inpaint jobs share one infer subprocess "
             "(checkpoint + EnCodec loaded once).",
    )
    p.add_argument(
        "--ext",
        type=float,
        default=0.02,
        metavar="SEC",
        help=(
            "Extend each inpainted span by SEC on both sides (default: 0.02). "
            "Spans immediately following a /t/ also get --ext by default; "
            "pass --no-force_ext to skip those."
        ),
    )
    p.add_argument(
        "--force_ext",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply --ext even to spans immediately following a /t/ (default: on).",
    )
    p.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        help=(
            "First stage to (re)run. Always regenerates this stage and everything "
            "after it (including stitch), even with --resume. Default: full run "
            "from segment, or continue after completed stages when --resume is set alone."
        ),
    )
    p.add_argument(
        "--to-stage",
        choices=STAGES,
        default=None,
        help=(
            "Last stage to run (inclusive). Use e.g. --to-stage align to stop after "
            "forced alignment without inpaint/stitch. Default: run through stitch."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep completed stages before --from-stage (or all completed stages "
            "when --from-stage is omitted). Default without --resume: regenerate "
            "from --from-stage (or segment)."
        ),
    )
    p.add_argument("--force", action="store_true", help="Re-run stages even when --resume is set")
    p.add_argument(
        "--rewrite-transcription",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Replace ASR texts before align: pass one quoted string per segment "
            "(in segment order), or a single .json path with a list of strings / "
            "name→text object. Errors if the count does not match the number of "
            "segment files. Implies re-running align even under --resume."
        ),
    )

    backend = p.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--mfa",
        action="store_const",
        const="mfa",
        dest="speech_backend",
        help="In-process Whisper + MFA",
    )
    backend.add_argument(
        "--maps",
        action="store_const",
        const="maps",
        dest="speech_backend",
        help="In-process Whisper + MAPS forced aligner in the active env "
             "(PhonePaint; no conda hop)",
    )

    p.add_argument("--silence-level", type=float, default=-40.0, help="Segmenter silence threshold (dBFS)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="ASR batch size (default: 4)",
    )
    p.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help=f"Hugging Face Whisper model (default: {DEFAULT_ASR_MODEL})",
    )
    p.add_argument("--language", default="en")
    p.add_argument("--asr-device", default="cuda")
    p.add_argument(
        "--asr-device-idx",
        type=int,
        default=None,
        help="CUDA device for ASR (default: GPU with most free VRAM)",
    )
    p.add_argument(
        "--env-mfa",
        default="mfa_env",
        help="--mfa only: sidecar conda env holding the MFA CLI, used when the "
             "active env has no in-env MFA (or set MFA_BIN).",
    )
    p.add_argument(
        "--align-threads",
        type=parse_align_threads,
        default=ALIGN_THREADS_SEGMENTS,
        metavar="N|segments",
        help="--mfa only: MFA --num_jobs / -j (default: segments = one job "
             "per segment wav for each input). Pass an integer to override; "
             "use 1 for short single-segment files. With 'segments', batch "
             "paths run each stem serially and never merge corpora.",
    )
    p.add_argument(
        "--max-parallel-align-jobs",
        type=int,
        default=0,
        help="--mfa only (batch paths): how to batch MFA across workspaces. "
             "0 (default) = merge all pending stems into one MFA corpus "
             "(amortize cold-start; uses --align-threads as MFA --num_jobs; "
             "ignored when --align-threads segments). "
             "N>=1 = up to N concurrent MFA processes; each keeps "
             "--align-threads as -j (segments → -j per stem's segment count).",
    )
    p.add_argument(
        "--maps-model",
        type=Path,
        default=None,
        help="--maps only: .pt checkpoint or ensemble directory "
             "(default: tools/maps/torch_models/timbuck_eng.pt or ensemble_model/)",
    )
    p.add_argument(
        "--maps-dict",
        type=Path,
        default=None,
        help="--maps only: CMUdict path (default: PHONEPAINT_CMUDICT or NLTK cmudict)",
    )
    p.add_argument(
        "--maps-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="--maps only: Torch device for MAPS acoustic model (default: auto)",
    )
    p.add_argument(
        "--maps-ensemble",
        action="store_true",
        help="--maps only: use 10-model ensemble (median boundaries; slower)",
    )

    infer = p.add_argument_group("Inpainting (infer_phone_ec_cond.py, 16 kHz)")
    infer.add_argument("--save-resampled", dest="save_resampled", type=Path, default=None)
    infer.add_argument("--device", dest="infer_device", default=None)
    infer.add_argument("--max_utt_sec", type=float, default=float("inf"))
    infer.add_argument(
        "--ola",
        action="store_true",
        help="Enable OLA synthesis (default: disabled; single full-sequence decode)",
    )
    infer.add_argument("--ola_chunk_frames", type=int, default=32)
    infer.add_argument("--ola_hop_factor", type=int, default=4, choices=[2, 4])
    infer.add_argument("--ola_ctx_frames", type=int, default=8)
    infer.add_argument("--normalize-vol", "--normalize_vol", dest="normalize_vol", type=float, default=None)
    infer.add_argument(
        "--windowed_attention", action="store_true",
        help="Encode/inpaint/(partial-)decode only a local context window around each "
             "span (passed through to infer_phone_ec_cond.py; default: off). "
             "Incompatible with --full-decode.",
    )
    infer.add_argument(
        "--context_frames", type=int, default=None,
        help="Context window size when --windowed_attention is set "
             f"(default: {CONTEXT_FRAMES} = ±{CONTEXT_FRAMES // 2} "
             "frames ≈ ±0.50s @ 50 fps).",
    )
    infer.add_argument(
        "--full-decode", "--full_decode", dest="full_decode", action="store_true",
        help="Pass --full-decode to infer_phone_ec_cond.py (single full-sequence "
             "EnCodec decode). Incompatible with --windowed_attention.",
    )
    infer.add_argument(
        "--full_phone_conditioning", action="store_true",
        help="Condition on remapped full utterance phone strings "
             "(passed through to infer_phone_ec_cond.py; requires a checkpoint "
             "trained with train_phone_ec_cond.py --full_phone_conditioning).",
    )
    infer.add_argument(
        "--encodec-ckpt",
        dest="encodec_ckpt",
        type=Path,
        default=None,
        help=(
            "Override VoiceCraft 16 kHz EnCodec checkpoint "
            f"(default: {ENCODEC_16K_CKPT})."
        ),
    )

    p.add_argument("--crossfade-ms", type=float, default=0.0, help="Optional crossfade at stitch joins")
    args = p.parse_args()
    maps_only = (
        args.maps_ensemble
        or args.maps_model is not None
        or args.maps_dict is not None
        or args.maps_device != "auto"
    )
    if maps_only and args.speech_backend != "maps":
        raise SystemExit("--maps-model/--maps-dict/--maps-device/--maps-ensemble require --maps")
    if args.maps_ensemble and args.maps_model is not None:
        raise SystemExit(
            "--maps-ensemble and --maps-model are mutually exclusive "
            "(ensemble uses tools/maps/torch_models/ensemble_model/ by default)"
        )
    return args


def _resolve_shared_runtime(args: argparse.Namespace) -> int:
    """Resolve built-in 16 kHz EnCodec path and context_frames; return output sample rate."""
    if args.full_decode and args.windowed_attention:
        raise SystemExit("--full-decode and --windowed_attention cannot be combined")

    encodec_ckpt = args.encodec_ckpt or ENCODEC_16K_CKPT
    if not encodec_ckpt.is_file():
        raise SystemExit(
            f"16 kHz EnCodec checkpoint not found: {encodec_ckpt}\n"
            "Install VoiceCraft pretrained_models/encodec_4cb2048_giga.th, "
            "set VOICECRAFT_ROOT, or pass --encodec-ckpt."
        )
    args.encodec_ckpt = encodec_ckpt

    if args.context_frames is None:
        args.context_frames = CONTEXT_FRAMES
    return OUTPUT_FS


def _overlay_job_args(
    shared: argparse.Namespace,
    job: dict,
) -> argparse.Namespace:
    """Copy *shared* and overlay per-pipeline-job fields from a batch manifest entry."""
    args = argparse.Namespace(**vars(shared))
    args.input = Path(job["input"])
    args.output = Path(job["output"])
    args.phones_in = list(job["phones_in"])
    args.phones_out = list(job["phones_out"])
    args.ext = float(job.get("ext", getattr(shared, "ext", 0.02)))
    args.force_ext = bool(job.get("force_ext", getattr(shared, "force_ext", True)))
    work_dir = job.get("work_dir")
    args.work_dir = Path(work_dir) if work_dir else None
    if "resume" in job:
        args.resume = bool(job["resume"])
    if "from_stage" in job:
        args.from_stage = job["from_stage"]
    if "force" in job:
        args.force = bool(job["force"])
    if "rewrite_transcription" in job:
        args.rewrite_transcription = job["rewrite_transcription"]
    validate_phone_map(args.phones_in, args.phones_out)
    return args
