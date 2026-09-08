"""Segmentation, transcription, and MFA/MAPS alignment backends."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pipeline_utils import all_inputs_shorter_than, copy_as_single_segments, pick_asr_device_idx, write_aligned_json
from unified_speech import segment_wav_directory

from .constants import MAPS_ALIGN_SCRIPT, MFA_ALIGN_SCRIPT, MIN_SPLIT_SEC, REPO_ROOT, UNIFIED_SPEECH_SCRIPT
from .runtime import resolve_mfa_bin, run_python

log = logging.getLogger(__name__)

def prepare_input(input_path: Path, ws: dict[str, Path]) -> Path:
    input_path = input_path.resolve()
    if input_path.is_dir():
        return input_path
    ws["input"].mkdir(parents=True, exist_ok=True)
    dst = ws["input"] / input_path.name
    if not dst.exists() or dst.stat().st_mtime < input_path.stat().st_mtime:
        shutil.copy2(input_path, dst)
    return ws["input"]


def run_segment(
    input_dir: Path,
    ws: dict[str, Path],
    *,
    silence_level: float,
) -> List[str]:
    """Applio-identical silence split in-process."""
    ws["segments"].mkdir(parents=True, exist_ok=True)
    if all_inputs_shorter_than(input_dir, MIN_SPLIT_SEC):
        log.info(
            "All inputs shorter than %.1fs — skipping silence-based split",
            MIN_SPLIT_SEC,
        )
        return copy_as_single_segments(input_dir, ws["segments"])
    return segment_wav_directory(
        input_dir,
        ws["segments"],
        silence_level=silence_level,
        min_split_sec=MIN_SPLIT_SEC,
    )


def run_transcribe_whisper(
    ws: dict[str, Path],
    *,
    model_name: str,
    batch_size: int,
    language: str,
    device: str,
    device_idx: Optional[int] = None,
) -> None:
    cmd = [
        "python",
        str(UNIFIED_SPEECH_SCRIPT),
        "transcribe",
        "--input-dir",
        str(ws["segments"]),
        "--output-json",
        str(ws["transcripts"]),
        "--model",
        model_name,
        "--language",
        language,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    if device.startswith("cuda"):
        idx = pick_asr_device_idx(device_idx)
        cmd.extend(["--device-idx", str(idx)])
    run_python(cmd, cwd=REPO_ROOT)



def run_transcribe(ws: dict[str, Path], args: argparse.Namespace) -> None:
    run_transcribe_whisper(
        ws,
        model_name=args.asr_model,
        batch_size=args.batch_size,
        language=args.language,
        device=args.asr_device,
        device_idx=args.asr_device_idx,
    )



@dataclass
class MfaAlignJob:
    """One MFA unit of work (usually one ``work/<stem>/`` tree)."""

    stem: str
    ws: dict[str, Path]
    segment_names: List[str]


# Special ``--align-threads`` value: MFA -j = number of segment wavs for that input.
ALIGN_THREADS_SEGMENTS = "segments"


def parse_align_threads(value: str):
    """Argparse type: positive int, or ``segments`` / ``auto`` (= per-input segment count)."""
    key = str(value).strip().lower().replace("_", "-")
    if key in {
        "segments", "auto", "nseg", "n-segments",
        "per-segment", "per-segments", "per-file",
    }:
        return ALIGN_THREADS_SEGMENTS
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--align-threads must be a positive int or 'segments', got {value!r}"
        ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError("--align-threads must be >= 1")
    return n


def resolve_mfa_align_threads(
    align_threads: object,
    segment_names: Sequence[str],
) -> int:
    """Map ``--align-threads`` to MFA ``--num_jobs`` for one input's segments."""
    if align_threads == ALIGN_THREADS_SEGMENTS:
        return max(1, len(segment_names))
    return max(1, int(align_threads))  # type: ignore[arg-type]


def _transcript_text(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("text") or "").strip()
    return str(entry or "").strip()


def _mfa_align_env(env_mfa: str, *, threads_per_job: int) -> dict[str, str]:
    mfa_bin = resolve_mfa_bin(env_mfa)
    env = os.environ.copy()
    env["MFA_BIN"] = mfa_bin
    env["MFA_CONDA_ENV"] = env_mfa
    if threads_per_job <= 1:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "1"
    return env


def _mfa_align_cmd(
    *,
    audio_dir: Path,
    input_json: Path,
    output_json: Path,
    threads: int,
) -> List[str]:
    return [
        sys.executable,
        str(MFA_ALIGN_SCRIPT),
        "--audio_dir",
        str(audio_dir),
        "--input_json",
        str(input_json),
        "--output_json",
        str(output_json),
        # english_mfa only; G2P runs only for files that still have fail/OOV words
        # after the primary dictionary pass (no english_us_arpa fallback pair).
        "--use_g2p_fallback",
        "--dump_pre_g2p",
        "--mfa_model_pairs",
        "english_mfa:english_mfa",
        "-j",
        str(max(1, int(threads))),
    ]


def _finalize_mfa_workspace(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    per_segment: dict,
    *,
    pre_g2p: Optional[dict] = None,
) -> None:
    """Write ``03_aligned*.json`` contracts from a per-segment MFA dict."""
    per_segment_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    per_segment_path.parent.mkdir(parents=True, exist_ok=True)
    per_segment_path.write_text(
        json.dumps(per_segment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing = [name for name in segment_names if name not in per_segment]
    if missing:
        raise RuntimeError(f"MFA alignment omitted segments: {missing}")
    write_aligned_json(per_segment, segment_names, ws["segments"], ws["aligned"])
    if pre_g2p is not None:
        pre_path = per_segment_path.with_name("03_aligned_per_segment_pre_g2p.json")
        pre_path.write_text(
            json.dumps(pre_g2p, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_aligned_json(
            pre_g2p,
            segment_names,
            ws["segments"],
            ws["aligned"].with_name("03_aligned_pre_g2p.json"),
        )


def run_align_mfa(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    *,
    env_mfa: str,
    threads: int,
) -> None:
    """MFA JSON wrapper in-process; Kaldi work via MFA CLI (same env or fallback)."""
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")
    mfa_bin = resolve_mfa_bin(env_mfa)
    per_segment_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    env = _mfa_align_env(env_mfa, threads_per_job=threads)
    cmd = _mfa_align_cmd(
        audio_dir=ws["segments"],
        input_json=ws["transcripts"],
        output_json=per_segment_path,
        threads=threads,
    )
    log.info("Running MFA CLI (%s): %s", mfa_bin, " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=str(MFA_ALIGN_SCRIPT.parent),
        check=True,
        env=env,
    )
    per_segment = json.loads(per_segment_path.read_text(encoding="utf-8"))
    pre_g2p_src = per_segment_path.with_name("03_aligned_per_segment_pre_g2p.json")
    pre = (
        json.loads(pre_g2p_src.read_text(encoding="utf-8"))
        if pre_g2p_src.is_file()
        else None
    )
    _finalize_mfa_workspace(ws, segment_names, per_segment, pre_g2p=pre)


def run_align_mfa_batch_merged(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    threads: int,
) -> None:
    """Strategy 1: one ``mfa align`` over a merged corpus, then scatter JSON.

    Amortizes MFA cold-start across many short PhonePaint workspaces. Unique
    utterance keys encode ``stem`` + segment name so results can be remapped
    back to each ``03_aligned.json`` contract.
    """
    if not jobs:
        return
    if len(jobs) == 1:
        job = jobs[0]
        run_align_mfa(
            job.ws, job.segment_names, env_mfa=env_mfa, threads=threads,
        )
        return
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")

    # key → (stem, original segment wav name)
    key_map: Dict[str, Tuple[str, str]] = {}
    merged_manifest: Dict[str, dict] = {}
    for job in jobs:
        transcripts = json.loads(job.ws["transcripts"].read_text(encoding="utf-8"))
        if not isinstance(transcripts, dict):
            raise RuntimeError(
                f"Expected dict transcripts for stem={job.stem}; got {type(transcripts)}"
            )
        for name in job.segment_names:
            audio_path = job.ws["segments"] / name
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing segment wav: {audio_path}")
            text = _transcript_text(transcripts.get(name))
            if not text:
                raise RuntimeError(
                    f"Empty transcript for stem={job.stem} segment={name}"
                )
            # Avoid path separators / collisions across stems.
            key = f"{job.stem}__{name}"
            if key in key_map:
                raise RuntimeError(f"Duplicate MFA batch key: {key}")
            key_map[key] = (job.stem, name)
            merged_manifest[key] = {"audio": str(audio_path.resolve()), "text": text}

    mfa_bin = resolve_mfa_bin(env_mfa)
    with tempfile.TemporaryDirectory(prefix="mfa_batch_merged_") as tmp:
        scratch = Path(tmp)
        audio_dir = scratch / "audio"
        audio_dir.mkdir()
        input_json = scratch / "merged_transcripts.json"
        output_json = scratch / "merged_aligned.json"
        input_json.write_text(
            json.dumps(merged_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        env = _mfa_align_env(env_mfa, threads_per_job=threads)
        cmd = _mfa_align_cmd(
            audio_dir=audio_dir,
            input_json=input_json,
            output_json=output_json,
            threads=threads,
        )
        log.info(
            "Running merged MFA for %d workspace(s) (%s): %s",
            len(jobs), mfa_bin, " ".join(cmd),
        )
        subprocess.run(
            cmd,
            cwd=str(MFA_ALIGN_SCRIPT.parent),
            check=True,
            env=env,
        )
        aligned_all = json.loads(output_json.read_text(encoding="utf-8"))
        pre_path = output_json.with_name(output_json.stem + "_pre_g2p.json")
        pre_all = (
            json.loads(pre_path.read_text(encoding="utf-8"))
            if pre_path.is_file()
            else None
        )

    by_stem: Dict[str, dict] = {job.stem: {} for job in jobs}
    pre_by_stem: Dict[str, dict] = {job.stem: {} for job in jobs}
    for key, payload in aligned_all.items():
        mapped = key_map.get(key)
        if mapped is None:
            continue
        stem, name = mapped
        by_stem[stem][name] = payload
    if pre_all is not None:
        for key, payload in pre_all.items():
            mapped = key_map.get(key)
            if mapped is None:
                continue
            stem, name = mapped
            pre_by_stem[stem][name] = payload

    for job in jobs:
        pre = pre_by_stem.get(job.stem) or None
        if pre is not None and not pre:
            pre = None
        _finalize_mfa_workspace(
            job.ws, job.segment_names, by_stem[job.stem], pre_g2p=pre,
        )


def run_align_mfa_jobs_parallel(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    max_parallel: int,
    threads_per_job: object = 1,
    poll_interval: float = 0.05,
) -> None:
    """Strategy 2: slot pool of MFA subprocesses (Applio libri launcher pattern).

    Uses ``Popen`` + poll (not ProcessPool) so this stays safe after the parent
    has already initialized CUDA (e.g. warm Whisper in ``run_demo_batched``).
    ``threads_per_job`` is MFA ``--num_jobs`` (-j) for each process.
    Use ``"segments"`` for per-stem ``-j = len(segments)``. Concurrent
    stems (``max_parallel > 1``) no longer override -j.
    """
    if not jobs:
        return
    if not MFA_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MFA align script not found: {MFA_ALIGN_SCRIPT}")

    workers = max(1, min(int(max_parallel), len(jobs)))
    per_segments = threads_per_job == ALIGN_THREADS_SEGMENTS
    if not per_segments:
        threads_fixed = max(1, int(threads_per_job))  # type: ignore[arg-type]
        if workers > 1 and threads_fixed > 1:
            log.warning(
                "max_parallel_align_jobs=%d with align_threads=%d "
                "(-j kept as requested; watch CPU oversubscription)",
                workers, threads_fixed,
            )
    else:
        threads_fixed = None

    def _threads_for(job: MfaAlignJob) -> int:
        if per_segments:
            return resolve_mfa_align_threads(ALIGN_THREADS_SEGMENTS, job.segment_names)
        assert threads_fixed is not None
        return threads_fixed

    if workers <= 1:
        for job in jobs:
            j = _threads_for(job)
            log.info(
                "MFA serial stem=%s segments=%d -j %d",
                job.stem, len(job.segment_names), j,
            )
            run_align_mfa(
                job.ws, job.segment_names, env_mfa=env_mfa, threads=j,
            )
        return

    queue = list(jobs)
    processes: List[Tuple[subprocess.Popen, MfaAlignJob]] = []
    log.info(
        "Running parallel MFA pool: %d job(s), max_parallel=%d, threads_per_job=%s",
        len(queue), workers,
        "segments" if per_segments else str(threads_fixed),
    )
    while queue or processes:
        while queue and len(processes) < workers:
            job = queue.pop(0)
            j = _threads_for(job)
            per_segment_path = job.ws["aligned"].with_name(
                "03_aligned_per_segment.json"
            )
            cmd = _mfa_align_cmd(
                audio_dir=job.ws["segments"],
                input_json=job.ws["transcripts"],
                output_json=per_segment_path,
                threads=j,
            )
            env = _mfa_align_env(env_mfa, threads_per_job=j)
            log.info(
                "MFA pool start stem=%s segments=%d -j %d",
                job.stem, len(job.segment_names), j,
            )
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(MFA_ALIGN_SCRIPT.parent),
            )
            processes.append((proc, job))

        alive: List[Tuple[subprocess.Popen, MfaAlignJob]] = []
        for proc, job in processes:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, job))
            elif ret != 0:
                raise RuntimeError(
                    f"MFA align failed for stem={job.stem} exit={ret}"
                )
            else:
                per_segment_path = job.ws["aligned"].with_name(
                    "03_aligned_per_segment.json"
                )
                per_segment = json.loads(
                    per_segment_path.read_text(encoding="utf-8")
                )
                pre_src = per_segment_path.with_name(
                    "03_aligned_per_segment_pre_g2p.json"
                )
                pre = (
                    json.loads(pre_src.read_text(encoding="utf-8"))
                    if pre_src.is_file()
                    else None
                )
                _finalize_mfa_workspace(
                    job.ws, job.segment_names, per_segment, pre_g2p=pre,
                )
        processes = alive
        if processes or queue:
            time.sleep(poll_interval)


def run_align_mfa_many(
    jobs: Sequence[MfaAlignJob],
    *,
    env_mfa: str,
    threads: object,
    max_parallel_align_jobs: int = 0,
) -> None:
    """Batch MFA for multiple workspaces.

    * ``max_parallel_align_jobs <= 0`` (default): Strategy 1 — merge corpora,
      one MFA invocation with ``--num_jobs=threads`` (best cold-start amortization).
      If ``threads == "segments"``, skip merge and run each stem serially with
      ``-j = len(segments)`` for that stem.
    * ``max_parallel_align_jobs >= 1``: Strategy 2 — up to N concurrent MFA
      processes. Each process keeps ``--align-threads`` as MFA ``-j``
      (``segments`` → ``-j = len(that stem's segments)``).
    """
    if not jobs:
        return
    per_segments = threads == ALIGN_THREADS_SEGMENTS
    if len(jobs) == 1:
        j = resolve_mfa_align_threads(threads, jobs[0].segment_names)
        run_align_mfa(
            jobs[0].ws, jobs[0].segment_names, env_mfa=env_mfa, threads=j,
        )
        return
    if per_segments:
        # Never merge when -j tracks per-file segment count.
        max_p = int(max_parallel_align_jobs)
        if max_p <= 0:
            max_p = 1
        run_align_mfa_jobs_parallel(
            jobs,
            env_mfa=env_mfa,
            max_parallel=max_p,
            threads_per_job=ALIGN_THREADS_SEGMENTS,
        )
        return
    if int(max_parallel_align_jobs) >= 1:
        # N>=1 → process pool (N=1 is serial); always honor --align-threads.
        run_align_mfa_jobs_parallel(
            jobs,
            env_mfa=env_mfa,
            max_parallel=int(max_parallel_align_jobs),
            threads_per_job=max(1, int(threads)),  # type: ignore[arg-type]
        )
    else:
        run_align_mfa_batch_merged(
            jobs, env_mfa=env_mfa, threads=max(1, int(threads)),  # type: ignore[arg-type]
        )


def run_align_maps(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    *,
    model: Optional[Path],
    dictionary: Optional[Path],
    device: str,
    ensemble: bool,
) -> None:
    """MAPS align in the active interpreter (no conda hop)."""
    if not MAPS_ALIGN_SCRIPT.is_file():
        raise SystemExit(f"MAPS align script not found: {MAPS_ALIGN_SCRIPT}")
    per_segment_path = ws["aligned"].with_name("03_aligned_per_segment.json")
    cmd = [
        "python",
        str(MAPS_ALIGN_SCRIPT),
        "--audio_dir",
        str(ws["segments"]),
        "--input_json",
        str(ws["transcripts"]),
        "--output_json",
        str(per_segment_path),
        "--device",
        device,
    ]
    if model is not None:
        cmd.extend(["--model", str(model)])
    if dictionary is not None:
        cmd.extend(["--dict", str(dictionary)])
    if ensemble:
        cmd.append("--ensemble")
    run_python(cmd, cwd=MAPS_ALIGN_SCRIPT.parent)
    per_segment = json.loads(per_segment_path.read_text(encoding="utf-8"))
    missing = [name for name in segment_names if name not in per_segment]
    if missing:
        raise RuntimeError(f"MAPS alignment omitted segments: {missing}")
    write_aligned_json(
        per_segment,
        segment_names,
        ws["segments"],
        ws["aligned"],
    )



def run_align(
    ws: dict[str, Path],
    segment_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    if args.speech_backend == "maps":
        run_align_maps(
            ws,
            segment_names,
            model=args.maps_model,
            dictionary=args.maps_dict,
            device=args.maps_device,
            ensemble=args.maps_ensemble,
        )
    else:
        run_align_mfa(
            ws,
            segment_names,
            env_mfa=args.env_mfa,
            threads=resolve_mfa_align_threads(args.align_threads, segment_names),
        )
