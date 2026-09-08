#!/usr/bin/env python3
"""PhonePaint production pipeline — command-line entry point.

How to use
----------
List every flag (this is the docs)::

    python inpaint_pipeline.py --help

Minimal single-file run (MFA)::

    conda activate PhonePaint
    python inpaint_pipeline.py \\
      --mfa \\
      --input /path/to/input.wav \\
      --output /path/to/output.wav \\
      --checkpoint /path/to/model.pt \\
      --phones_in s \\
      --phones_out th

What runs (in order)
--------------------
  1. segment     — silence-split long audio (skipped if short)
  2. transcribe  — Whisper ASR on segments
  3. align       — MFA (--mfa) or MAPS (--maps) forced alignment
  4. inpaint     — 16 kHz phone replacement via infer_phone_ec_cond.py
  5. stitch      — write final --output wav (16 kHz)

Where the code lives
--------------------
  inpaint_pipeline.py              this file — run me; ``--help`` lists flags
  pipeline_scripts/cli.py          argparse definitions
  pipeline_scripts/coordinator.py  stage order / single + batch runners
  pipeline_scripts/speech.py       segment, Whisper, MFA/MAPS
  pipeline_scripts/inference.py    spans, infer, stitch helpers
  pipeline_scripts/spans.py        phone-map → time spans
  pipeline_scripts/workspace.py    work/<stem>/ layout + manifest
  pipeline_scripts/timing.py       stage wall-clock / VRAM stats

Do not run pipeline_scripts/cli.py alone — it only parses args.
This script calls coordinator.main(), which parses then runs.
"""

from __future__ import annotations

# Re-export the package API so ``from inpaint_pipeline import …`` still works.
from pipeline_scripts.cli import *  # noqa: F401,F403
from pipeline_scripts.constants import *  # noqa: F401,F403
from pipeline_scripts.coordinator import *  # noqa: F401,F403
from pipeline_scripts.inference import *  # noqa: F401,F403
from pipeline_scripts.runtime import *  # noqa: F401,F403
from pipeline_scripts.speech import *  # noqa: F401,F403
from pipeline_scripts.spans import *  # noqa: F401,F403
from pipeline_scripts.timing import *  # noqa: F401,F403
from pipeline_scripts.workspace import *  # noqa: F401,F403

from pipeline_scripts.coordinator import main


if __name__ == "__main__":
    main()
