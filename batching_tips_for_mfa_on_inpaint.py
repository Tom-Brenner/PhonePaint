#!/usr/bin/env python3
"""Batching tips for MFA multi-file alignment in PhonePaint (brief for another LLM).

This file is **guidance + stubs**, not a runnable aligner. Implement changes in
``run_inpaint_pipeline.py`` / ``run_demo.py`` (and optionally
``tools/mfa/mfa_align_json.py``) using the patterns below.

=============================================================================
Problem (current PhonePaint path)
=============================================================================

``run_inpaint_pipeline.run_align_mfa`` (≈840–882) spawns a **new** Python
process for ``tools/mfa/mfa_align_json.py`` per workspace. That wrapper then
calls ``mfa align`` via subprocess (see ``tools/mfa/mfa_align.py::run_mfa``,
≈235–301) with ``--clean``, ``--num_jobs`` / ``-j``, ``--use_mp``.

``run_demo.py`` already **batches inpaint** across jobs via
``--batch_manifest`` (checkpoint/EnCodec loaded once), but **align** still
tends to run **per stem / per pipeline invocation**, so MFA pays cold-start
(model load + ``--clean`` workspace rebuild) repeatedly.

G2P is already skipped when no OOV/``spn`` leftovers remain
(``mfa_align.py`` incremental path); that is **not** the main multi-file cost.

=============================================================================
Reference implementations (read these first)
=============================================================================

1. ``/media/tom/SATAM/Applio41/mfa_align_libri_batch.py``
   - ``ProcessPoolExecutor`` over **roots**
   - Each worker runs one ``mfa_align_libri.py`` subprocess
   - Isolates outputs by ``output_dir / root.name``
   - Relies on **threads=1 / omp_threads=1 inside** the child (no nested MFA job
     storm)

2. ``/media/tom/SATAM/Applio41/mfa_align_libri_launcher.py``
   - Same idea with an explicit **slot pool**: ``Popen`` + poll loop
   - ``--max_parallel_terminals`` caps concurrent MFA processes
   - Prefer this style when you need fail-soft per root and dynamic refill

3. ``/media/tom/SATAM/Applio41/mfa_align_libri.py`` (supporting)
   - Forces ``args.threads = 1`` and ``args.omp_threads = 1``
   - Optional ``--max_parallel_subdirs`` = ProcessPool **inside** one root
   - Outer launcher/batch × inner MFA ``--num_jobs`` must not both be large

4. PhonePaint touch points
   - ``~/PhonePaint/run_inpaint_pipeline.py``: ``run_align_mfa``, ``run_align``,
     ``run_pipeline_batch``, ``--align-threads``
   - ``~/PhonePaint/run_demo.py``: ``build_pipeline_batch_cmd``,
     ``run_pipeline_batches``, ``force_align_demo_sources``, ``--realign``
   - ``~/PhonePaint/tools/mfa/mfa_align_json.py`` / ``mfa_align.py::run_mfa``

=============================================================================
Design rules
=============================================================================

A. **Two levels of parallelism — pick one dominant level**

   - **Level 1 (preferred for many short PhonePaint segment dirs):**
     N concurrent processes, each running MFA on **one** corpus/work-dir with
     ``--num_jobs 1`` and ``OMP_NUM_THREADS=1`` (Applio libri pattern).

   - **Level 2 (preferred for one big corpus):**
     Single ``mfa align`` over a merged corpus with ``--num_jobs N``
     (native MFA worker pool; models loaded once).

   Do **not** run 8 outer MFA processes each with ``--num_jobs 8``.

B. **Amortize cold start**
   - Prefer one MFA invocation per many wavs, or a small pool of long-lived
     workers — not one ``mfa align --clean`` per demo stem.
   - Gate ``--clean``: use it for first run / ``--realign``; skip when the MFA
     job dir is intentionally reused.

C. **Isolate artifacts**
   - Per-worker: history dir, corpus temp dir, output TextGrids
     (Applio appends ``root.name`` under shared parents).

D. **Keep pipeline contracts**
   - Downstream still expects ``work/<stem>/03_aligned.json`` (and optional
     ``03_aligned_per_segment.json``). Parallel align may write a shared
     scratch corpus, then **scatter** TextGrids/JSON back into each workspace.

E. **Demo timing**
   - ``--force-alignment`` sidecar timing must remain meaningful: either time
     the shared batch and attribute roughly, or keep sidecar single-stem runs
     for RT measurement only.

=============================================================================
Recommended strategies for PhonePaint
=============================================================================

Strategy 1 — Shared MFA corpus (best wall-clock when many stems need align)
  Collect all pending ``segments/*.wav`` + transcripts from multiple workspaces
  into one temp corpus → one ``mfa align`` with ``--num_jobs`` → map TextGrids
  back → build each ``03_aligned.json``.

Strategy 2 — Applio root pool (best when workspaces must stay isolated)
  ``run_demo`` / batch_manifest path: queue stems that need align; run up to
  ``max_parallel_align_jobs`` subprocesses of ``mfa_align_json.py`` (or a thin
  wrapper) with ``-j 1``; then proceed to the existing shared inpaint batch.

Strategy 3 — Hybrid
  Within one pipeline job (many segments): one MFA corpus (Level 2).
  Across demo jobs: pool of jobs (Level 1) with ``-j 1`` each.

=============================================================================
Code stubs (implement elsewhere; do not execute this file as a CLI)
=============================================================================
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Stub types mirroring PhonePaint workspaces
# ---------------------------------------------------------------------------

@dataclass
class AlignJob:
    """One MFA unit of work (usually one ``work/<stem>/`` tree)."""

    stem: str
    work_dir: Path
    segments_dir: Path
    transcripts_json: Path
    aligned_json: Path  # final PhonePaint contract path


# ---------------------------------------------------------------------------
# Pattern from mfa_align_libri_launcher.py — slot pool of subprocesses
# ---------------------------------------------------------------------------

def align_jobs_with_popen_pool(
    jobs: Sequence[AlignJob],
    *,
    mfa_align_json: Path,
    max_parallel: int,
    threads_per_job: int = 1,
    poll_interval: float = 0.05,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    """Launch up to ``max_parallel`` ``mfa_align_json.py`` processes at a time.

    Reference: ``/media/tom/SATAM/Applio41/mfa_align_libri_launcher.py``
    (queue + ``Popen`` + ``poll``; refill when slots free).

    Wire from ``run_demo.run_pipeline_batches`` *before* shared inpaint, for
    stems missing ``03_aligned.json`` (or after ``--realign`` invalidation).
    """
    queue = list(jobs)
    processes: list[tuple[subprocess.Popen, AlignJob]] = []
    extra = list(extra_args or [])

    while queue or processes:
        while queue and len(processes) < max_parallel:
            job = queue.pop(0)
            cmd = [
                os.environ.get("PYTHON", "python"),
                str(mfa_align_json),
                "--audio_dir",
                str(job.segments_dir),
                "--input_json",
                str(job.transcripts_json),
                "--output_json",
                str(job.aligned_json.with_name("03_aligned_per_segment.json")),
                "-j",
                str(threads_per_job),
                *extra,
            ]
            # IMPORTANT: set OMP/MKL to 1 in env when threads_per_job == 1
            env = os.environ.copy()
            if threads_per_job <= 1:
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    env[k] = "1"
            proc = subprocess.Popen(cmd, env=env, cwd=str(mfa_align_json.parent))
            processes.append((proc, job))

        alive: list[tuple[subprocess.Popen, AlignJob]] = []
        for proc, job in processes:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, job))
            elif ret != 0:
                raise RuntimeError(f"MFA align failed for stem={job.stem} exit={ret}")
            else:
                # TODO: write_aligned_json(per_segment → job.aligned_json)
                pass
        processes = alive
        if processes or queue:
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Pattern from mfa_align_libri_batch.py — ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _align_one_job_subprocess(payload: dict) -> str:
    """Worker entry: must be top-level picklable for ProcessPoolExecutor."""
    cmd = payload["cmd"]
    cwd = payload["cwd"]
    env = payload["env"]
    stem = payload["stem"]
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"MFA align failed for stem={stem} exit={proc.returncode}")
    return stem


def align_jobs_with_process_pool(
    jobs: Sequence[AlignJob],
    *,
    mfa_align_json: Path,
    max_parallel: int,
    threads_per_job: int = 1,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    """ProcessPool over stems; each child runs one MFA wrapper subprocess.

    Reference: ``/media/tom/SATAM/Applio41/mfa_align_libri_batch.py``
    """
    extra = list(extra_args or [])
    payloads = []
    for job in jobs:
        env = os.environ.copy()
        if threads_per_job <= 1:
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[k] = "1"
        payloads.append(
            {
                "stem": job.stem,
                "cwd": str(mfa_align_json.parent),
                "env": env,
                "cmd": [
                    os.environ.get("PYTHON", "python"),
                    str(mfa_align_json),
                    "--audio_dir",
                    str(job.segments_dir),
                    "--input_json",
                    str(job.transcripts_json),
                    "--output_json",
                    str(job.aligned_json.with_name("03_aligned_per_segment.json")),
                    "-j",
                    str(threads_per_job),
                    *extra,
                ],
            }
        )

    workers = min(max_parallel, max(1, len(payloads)))
    if workers <= 1:
        for p in payloads:
            _align_one_job_subprocess(p)
        return

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_align_one_job_subprocess, p) for p in payloads]
        for f in as_completed(futs):
            f.result()


# ---------------------------------------------------------------------------
# Strategy 1 — merge corpora, one MFA, scatter (sketch)
# ---------------------------------------------------------------------------

def align_many_workspaces_single_mfa_corpus(
    jobs: Sequence[AlignJob],
    *,
    scratch_dir: Path,
    num_jobs: int,
) -> None:
    """Best amortization: one ``mfa align`` for all pending wavs.

    Steps for the implementer:
      1. scratch_dir/corpus/{unique_id}.wav + .lab from each job.segments_dir
         (unique_id must encode stem + segment name to scatter later).
      2. Call ``tools/mfa/mfa_align.py::run_mfa(scratch_dir/corpus, …,
         threads=num_jobs)`` **once**. Consider omitting ``--clean`` on reuse;
         today ``run_mfa`` always passes ``--clean`` — add a flag.
      3. Parse TextGrids → per-stem aligned dicts → write each
         ``job.aligned_json`` using existing ``write_aligned_json``.

    Use this from ``run_inpaint_pipeline.run_pipeline_batch`` when
    ``speech_backend == "mfa"`` and multiple pipeline_jobs need align.
    """
    raise NotImplementedError("sketch only — see docstring steps")


# ---------------------------------------------------------------------------
# Integration hooks (where to edit)
# ---------------------------------------------------------------------------

def run_demo_align_then_inpaint_hook() -> None:
    """Suggested ``run_demo.py`` control flow change (pseudocode).

    Current (approx):
      for variant_batch in pipeline_batches:
          run_inpaint_pipeline --batch_manifest  # segment/transcribe/align/inpaint
                                                # align still per-job inside

    Target:
      jobs_needing_align = [j for j in pipeline_jobs if not aligned_or_realign]
      align_jobs_with_popen_pool(jobs_needing_align, max_parallel=..., threads_per_job=1)
      # or align_many_workspaces_single_mfa_corpus(...)
      run_pipeline_batches(...)  # existing shared inpaint; skip align stage
                                 # via --from-stage inpaint / cached 03_aligned.json

    Also update ``force_align_demo_sources``: either call the pooled aligner on
    sidecars ``work/<stem>_rt_align/`` or document that force-align RT is
    single-stem by design.
    """
    raise NotImplementedError


def run_inpaint_pipeline_batch_align_hook() -> None:
    """Suggested ``run_inpaint_pipeline.py`` change inside ``run_pipeline_batch``.

    Before per-job stage loops (or as a prelude):
      1. Build AlignJob list from ``pipeline_jobs`` work dirs that lack
         ``03_aligned.json`` or whose stage plan includes align.
      2. Run Strategy 1 or 2 once for the whole batch.
      3. Mark align complete so individual job loops start at inpaint.

    Replace or bypass ``run_align_mfa``'s per-call ``subprocess.run(mfa_align_json)``
    when ``len(jobs) > 1``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Checklist for the implementing LLM
# ---------------------------------------------------------------------------

IMPLEMENTATION_CHECKLIST: List[str] = [
    "Read Applio launcher + batch + mfa_align_libri.py thread pinning.",
    "Read PhonePaint run_align_mfa / run_mfa (--clean, --num_jobs).",
    "Add CLI: --max-parallel-align-jobs (outer) vs --align-threads (inner MFA).",
    "Default: outer>1 ⇒ force align-threads=1 and OMP=1; document the tradeoff.",
    "Optional: run_mfa(..., clean: bool = True) to skip --clean on warm reuse.",
    "Preserve 03_aligned.json / per_segment JSON contracts for inpaint + VoiceCraft.",
    "Do not double-parallelize MFA num_jobs with a large process pool.",
    "Keep G2P-as-needed behavior; batching does not require disabling G2P.",
    "Update run_demo batch path: align pool (or merged corpus) then existing inpaint batch.",
    "Decide force-alignment timing policy (sidecar single-stem vs attributed batch).",
]


if __name__ == "__main__":
    print(__doc__.split("=============================================================================")[0])
    print("This module is documentation/stubs only.")
    print("Checklist:")
    for item in IMPLEMENTATION_CHECKLIST:
        print(f"  - {item}")
