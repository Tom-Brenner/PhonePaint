"""Top-level stage coordination for single and batch inpainting runs."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from pipeline_utils import deploy_stitched_output, invalidate_from_stage, load_rewrite_texts, prepare_stage_start, rewrite_transcriptions

from .cli import _overlay_job_args, _resolve_shared_runtime, parse_args
from .constants import REPO_ROOT, STAGES
from .inference import collect_inpaint_jobs, execute_infer_batch, run_inpaint, source_order_from_segments, stitch_wavs
from .speech import MfaAlignJob, prepare_input, run_align, run_align_mfa_many, run_segment, run_transcribe
from .spans import validate_phone_map
from .timing import _fmt_elapsed, load_stage_stats_state, reset_stage_stats, stage_timing, vram_peak_mb, wallclock, write_stage_stats
from .workspace import load_manifest, mark_stage, save_manifest, sorted_segment_wavs, stage_done, workspace_paths

log = logging.getLogger(__name__)

def should_run(stage: str, from_stage: str) -> bool:
    order = {s: i for i, s in enumerate(STAGES)}
    return order[stage] >= order[from_stage]


def stage_should_run(stage: str, args: argparse.Namespace, manifest: dict) -> bool:
    if not should_run(stage, args.from_stage):
        return False
    to_stage = getattr(args, "to_stage", None)
    if to_stage is not None:
        order = {s: i for i, s in enumerate(STAGES)}
        if order[stage] > order[to_stage]:
            return False
    if args.resume and not args.force and stage_done(manifest, stage):
        return False
    return True


def prepare_pipeline_through_align(
    args: argparse.Namespace,
    *,
    out_fs: int,
    defer_align: bool = False,
) -> Tuple[dict[str, Path], dict, List[str], Path, bool]:
    """Run segment/transcribe/align as needed.

    Returns ``(ws, manifest, segment_names, input_dir, align_pending)``.
    When ``defer_align`` is true and the align stage would run under ``--mfa``,
    skip per-job MFA and set ``align_pending`` so the caller can batch-align.
    """
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    stem = input_path.stem if input_path.is_file() else input_path.name
    work_dir = (args.work_dir or (REPO_ROOT / "work" / stem)).resolve()
    ws = workspace_paths(work_dir)
    manifest = load_manifest(ws["manifest"])
    manifest.setdefault("input", str(input_path))
    manifest.setdefault("stem", stem)

    if args.speech_backend == "maps":
        speech_runtime = {
            "transcriber": "transformers-whisper",
            "asr_model": args.asr_model,
            "language": args.language,
            "aligner": "maps",
            "maps_model": str(args.maps_model) if args.maps_model else None,
            "maps_dict": str(args.maps_dict) if args.maps_dict else None,
            "maps_device": args.maps_device,
            "maps_ensemble": args.maps_ensemble,
            "phone_inventory": "phonepaint-arpa-v1",
        }
    else:
        speech_runtime = {
            "transcriber": "transformers-whisper",
            "asr_model": args.asr_model,
            "language": args.language,
            "aligner": "mfa",
            "env_mfa": args.env_mfa,
            "align_threads": args.align_threads,
            "phone_inventory": "phonepaint-arpa-v1",
        }
    previous_runtime = manifest.get("speech_runtime")
    if previous_runtime != speech_runtime:
        if previous_runtime is None and stage_done(manifest, "align"):
            invalidate_from_stage(ws, "align", manifest, stages=STAGES)
        elif previous_runtime is not None:
            transcription_keys = ("transcriber", "asr_model", "language")
            transcription_changed = any(
                previous_runtime.get(key) != speech_runtime.get(key)
                for key in transcription_keys
            )
            invalidate_from_stage(
                ws,
                "transcribe" if transcription_changed else "align",
                manifest,
                stages=STAGES,
            )
        manifest["speech_runtime"] = speech_runtime

    args.from_stage = prepare_stage_start(
        ws,
        manifest,
        stages=STAGES,
        from_stage=args.from_stage,
        resume=args.resume,
    )
    # Batch/resume may request inpaint while a backend switch invalidated align.
    if args.from_stage in ("inpaint", "stitch") and not ws["aligned"].is_file():
        log.warning(
            "%s: aligned JSON missing; re-running from align (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "align"
    if args.from_stage in ("align", "inpaint", "stitch") and not ws["transcripts"].is_file():
        log.warning(
            "%s: transcripts missing; re-running from transcribe (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "transcribe"
    if args.from_stage != "segment" and not any(ws["segments"].glob("*.wav")):
        log.warning(
            "%s: segments missing; re-running from segment (requested --from-stage %s)",
            stem,
            args.from_stage,
        )
        args.from_stage = "segment"
    save_manifest(ws["manifest"], manifest)

    input_dir = prepare_input(input_path, ws)

    if stage_should_run("segment", args, manifest):
        log.info("=== Stage: segment (%s) ===", stem)
        with wallclock(f"stage segment ({stem})"):
            segment_names = run_segment(
                input_dir, ws, silence_level=args.silence_level,
            )
        manifest["segments"] = segment_names
        mark_stage(manifest, "segment")
        save_manifest(ws["manifest"], manifest)
    else:
        segment_names = manifest.get("segments") or [
            p.name for p in sorted_segment_wavs(ws["segments"])
        ]
        if not segment_names:
            log.warning(
                "No cached segments for %s; running from segment "
                "(requested --from-stage %s)",
                stem, args.from_stage,
            )
            args.from_stage = "segment"
            log.info("=== Stage: segment (%s) ===", stem)
            with wallclock(f"stage segment ({stem})"):
                segment_names = run_segment(
                    input_dir, ws, silence_level=args.silence_level,
                )
            manifest["segments"] = segment_names
            mark_stage(manifest, "segment")
            save_manifest(ws["manifest"], manifest)

    if not segment_names:
        raise SystemExit(f"No segments found for {stem}; run segment stage first.")

    rewrite_texts = load_rewrite_texts(args.rewrite_transcription)
    if rewrite_texts is not None:
        order = {s: i for i, s in enumerate(STAGES)}
        if order[args.from_stage] > order["align"]:
            raise SystemExit(
                "--rewrite-transcription requires re-running align; "
                "use --from-stage align (or earlier), typically with --resume."
            )

    if stage_should_run("transcribe", args, manifest):
        if rewrite_texts is not None:
            raise SystemExit(
                "--rewrite-transcription replaces ASR output; do not also run the "
                "transcribe stage. Use --from-stage align (typically with --resume)."
            )
        log.info("=== Stage: transcribe (%s) ===", stem)
        with wallclock(f"stage transcribe ({stem})"):
            run_transcribe(ws, args)
        mark_stage(manifest, "transcribe")
        save_manifest(ws["manifest"], manifest)

    if rewrite_texts is not None:
        log.info(
            "=== Rewrite transcriptions (%s, %d segment(s)) ===",
            stem, len(rewrite_texts),
        )
        rewrite_transcriptions(ws["transcripts"], segment_names, rewrite_texts)
        invalidate_from_stage(ws, "align", manifest, stages=STAGES)
        mark_stage(manifest, "transcribe")
        save_manifest(ws["manifest"], manifest)

    need_align = stage_should_run("align", args, manifest) or rewrite_texts is not None
    align_pending = bool(
        need_align and defer_align and args.speech_backend == "mfa"
    )
    if need_align and not align_pending:
        log.info("=== Stage: align (%s) ===", stem)
        with wallclock(f"stage align ({stem})"):
            run_align(ws, segment_names, args)
        mark_stage(manifest, "align")
        save_manifest(ws["manifest"], manifest)
    elif align_pending:
        log.info(
            "=== Stage: align (%s) deferred for MFA batch ===", stem,
        )

    return ws, manifest, segment_names, input_dir, align_pending


def finish_pipeline_after_inpaint(
    args: argparse.Namespace,
    *,
    ws: dict[str, Path],
    manifest: dict,
    source_files: Sequence[str],
    out_fs: int,
) -> None:
    """Stitch inpainted sources and deploy to --output."""
    if stage_should_run("stitch", args, manifest):
        log.info("=== Stage: stitch (%s) ===", Path(args.output).name)
        inpainted_paths = [ws["inpainted"] / f"{stem}.wav" for stem in source_files]
        missing = [str(p) for p in inpainted_paths if not p.is_file()]
        if missing:
            raise SystemExit(
                f"Missing inpainted source files: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )
        with wallclock(f"stage stitch ({Path(args.output).name})"):
            stitch_wavs(
                inpainted_paths, ws["stitched"],
                crossfade_ms=args.crossfade_ms, out_fs=out_fs,
            )
        mark_stage(manifest, "stitch")
        save_manifest(ws["manifest"], manifest)

    deploy_stitched_output(ws["stitched"], args.output)
    if ws["stitched"].is_file():
        manifest["output"] = str(args.output.resolve())
        save_manifest(ws["manifest"], manifest)


def run_pipeline_single(args: argparse.Namespace) -> None:
    """Original single --input/--output pipeline path."""
    validate_phone_map(args.phones_in, args.phones_out)
    out_fs = _resolve_shared_runtime(args)
    pipeline_t0 = time.perf_counter()
    reset_stage_stats()

    ws, manifest, segment_names, input_dir, _align_pending = prepare_pipeline_through_align(
        args, out_fs=out_fs,
    )
    stem = Path(args.input).stem

    if args.to_stage is not None and STAGES.index(args.to_stage) < STAGES.index("inpaint"):
        write_stage_stats(ws["root"], stem=stem)
        log.info(
            "[timing] pipeline total (through %s) — %s",
            args.to_stage,
            _fmt_elapsed(time.perf_counter() - pipeline_t0),
        )
        log.info("Pipeline stopped after --to-stage %s: %s", args.to_stage, ws["root"])
        return

    if stage_should_run("inpaint", args, manifest):
        log.info("=== Stage: inpaint ===")
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        with wallclock("stage inpaint"):
            source_files = run_inpaint(
                ws, infer_args=args,
                segment_names=segment_names, input_dir=input_dir, out_fs=out_fs,
            )
        manifest["source_files"] = source_files
        mark_stage(manifest, "inpaint")
        save_manifest(ws["manifest"], manifest)
    else:
        source_files = manifest.get("source_files") or source_order_from_segments(
            manifest.get("segments")
            or [p.name for p in sorted_segment_wavs(ws["segments"])]
        )

    finish_pipeline_after_inpaint(
        args, ws=ws, manifest=manifest, source_files=source_files, out_fs=out_fs,
    )
    write_stage_stats(ws["root"], stem=stem)
    log.info(
        "[timing] pipeline total — %s",
        _fmt_elapsed(time.perf_counter() - pipeline_t0),
    )
    log.info("Pipeline complete: %s", args.output)


def run_pipeline_batch(args: argparse.Namespace) -> None:
    """Multi-input batch: prepare each workspace, one shared infer, then stitch each."""
    out_fs = _resolve_shared_runtime(args)
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    raw = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    pipeline_jobs = raw["pipeline_jobs"] if isinstance(raw, dict) else raw
    if not pipeline_jobs:
        raise SystemExit(f"--batch_manifest {args.batch_manifest} has no pipeline_jobs")

    pipeline_t0 = time.perf_counter()
    prepared: List[dict] = []
    pending_mfa: List[MfaAlignJob] = []
    all_infer_jobs: List[dict] = []
    defer_mfa = args.speech_backend == "mfa"

    for i, job in enumerate(pipeline_jobs):
        job_args = _overlay_job_args(args, job)
        log.info(
            "=== Batch pipeline job %d/%d: %s -> %s ===",
            i + 1, len(pipeline_jobs), job_args.input.name, job_args.output.name,
        )
        reset_stage_stats()
        ws, manifest, segment_names, input_dir, align_pending = (
            prepare_pipeline_through_align(
                job_args, out_fs=out_fs, defer_align=defer_mfa,
            )
        )
        stem = Path(job_args.input).stem
        if align_pending:
            pending_mfa.append(
                MfaAlignJob(stem=stem, ws=ws, segment_names=list(segment_names))
            )

        # Align may still be pending; inpaint collection needs 03_aligned.json.
        # Defer collect_inpaint_jobs until after the MFA batch below.
        prepared.append(dict(
            args=job_args, ws=ws, manifest=manifest,
            segment_names=segment_names, input_dir=input_dir,
            stem=stem, align_pending=align_pending,
            pre_inpaint_timing=stage_timing(),
            pre_inpaint_vram=vram_peak_mb(),
            needs_inpaint_mark=False,
            source_files=[],
        ))

    if pending_mfa:
        log.info(
            "=== Stage: align (MFA batch, %d workspace(s), max_parallel=%d) ===",
            len(pending_mfa),
            int(getattr(args, "max_parallel_align_jobs", 0) or 0),
        )
        reset_stage_stats()
        with wallclock("stage align (MFA batch)"):
            run_align_mfa_many(
                pending_mfa,
                env_mfa=args.env_mfa,
                threads=args.align_threads,
                max_parallel_align_jobs=int(
                    getattr(args, "max_parallel_align_jobs", 0) or 0
                ),
            )
        shared_align_t = float(stage_timing().get("align", 0.0))
        shared_align_vram = vram_peak_mb()
        n_pending = max(1, len(pending_mfa))
        share = shared_align_t / float(n_pending)
        for entry in prepared:
            if not entry["align_pending"]:
                continue
            mark_stage(entry["manifest"], "align")
            save_manifest(entry["ws"]["manifest"], entry["manifest"])
            # Attribute shared MFA wall evenly across deferred stems.
            merged = dict(entry["pre_inpaint_timing"])
            merged["align"] = share
            load_stage_stats_state(
                merged,
                max(float(entry["pre_inpaint_vram"]), shared_align_vram),
            )
            entry["pre_inpaint_timing"] = stage_timing()
            entry["pre_inpaint_vram"] = vram_peak_mb()

    # Collect inpaint jobs now that alignments exist.
    for entry in prepared:
        job_args = entry["args"]
        ws = entry["ws"]
        manifest = entry["manifest"]
        segment_names = entry["segment_names"]
        input_dir = entry["input_dir"]
        if stage_should_run("inpaint", job_args, manifest):
            source_files, batch_jobs, _counts = collect_inpaint_jobs(
                ws,
                infer_args=job_args,
                segment_names=segment_names,
                input_dir=input_dir,
                out_fs=out_fs,
            )
            all_infer_jobs.extend(batch_jobs)
            manifest["source_files"] = source_files
            entry["source_files"] = source_files
            entry["needs_inpaint_mark"] = True
        else:
            source_files = list(
                manifest.get("source_files") or source_order_from_segments(
                    manifest.get("segments")
                    or [p.name for p in sorted_segment_wavs(ws["segments"])]
                )
            )
            write_stage_stats(ws["root"], stem=entry["stem"])
            entry["source_files"] = source_files
            entry["needs_inpaint_mark"] = False

    if all_infer_jobs:
        # Prefer a stable temp path under the first workspace's spans dir.
        first_ws = prepared[0]["ws"]
        first_ws["spans"].mkdir(parents=True, exist_ok=True)
        manifest_path = first_ws["spans"] / "_demo_pipeline_batch_manifest.json"
        log.info("=== Stage: inpaint (shared batch, %d source file(s)) ===", len(all_infer_jobs))
        reset_stage_stats()
        with wallclock("stage inpaint (shared batch)"):
            execute_infer_batch(
                infer_args=args,
                batch_jobs=all_infer_jobs,
                manifest_path=manifest_path,
            )
        shared_inpaint_t = float(stage_timing().get("inpaint", 0.0))
        shared_vram = vram_peak_mb()
    else:
        shared_inpaint_t = 0.0
        shared_vram = 0.0

    for entry in prepared:
        job_args = entry["args"]
        ws = entry["ws"]
        manifest = entry["manifest"]
        if entry["needs_inpaint_mark"]:
            mark_stage(manifest, "inpaint")
            save_manifest(ws["manifest"], manifest)
        finish_pipeline_after_inpaint(
            job_args,
            ws=ws,
            manifest=manifest,
            source_files=entry["source_files"],
            out_fs=out_fs,
        )
        # Merge pre-inpaint stage times with this job's share of batch inpaint.
        merged = dict(entry["pre_inpaint_timing"])
        if entry["needs_inpaint_mark"]:
            merged["inpaint"] = shared_inpaint_t
        load_stage_stats_state(
            merged, max(float(entry["pre_inpaint_vram"]), shared_vram),
        )
        write_stage_stats(ws["root"], stem=entry["stem"])
        log.info("Pipeline complete: %s", job_args.output)

    log.info(
        "[timing] pipeline batch total — %d job(s), %s",
        len(pipeline_jobs),
        _fmt_elapsed(time.perf_counter() - pipeline_t0),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.batch_manifest is not None:
        if args.input is not None or args.output is not None:
            raise SystemExit(
                "--batch_manifest cannot be combined with --input/--output"
            )
        if args.phones_in is not None or args.phones_out is not None:
            raise SystemExit(
                "--batch_manifest cannot be combined with --phones_in/--phones_out "
                "(supply them per pipeline_job in the manifest)"
            )
        run_pipeline_batch(args)
        return

    if args.input is None or args.output is None:
        raise SystemExit(
            "--input and --output are required unless --batch_manifest is given"
        )
    if args.phones_in is None or args.phones_out is None:
        raise SystemExit(
            "--phones_in and --phones_out are required unless --batch_manifest is given"
        )
    run_pipeline_single(args)


if __name__ == "__main__":
    main()

