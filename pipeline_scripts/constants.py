"""Shared paths and immutable pipeline defaults."""

from __future__ import annotations

import re
from pathlib import Path

from phone_labels import canonical_user_phone
from voicecraft_encodec import DEFAULT_CHECKPOINT as ENCODEC_16K_CKPT

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_SPEECH_SCRIPT = REPO_ROOT / "unified_speech.py"
MFA_ALIGN_SCRIPT = REPO_ROOT / "tools" / "mfa" / "mfa_align_json.py"
MAPS_ALIGN_SCRIPT = REPO_ROOT / "tools" / "maps" / "maps_align_json.py"
INFER_SCRIPT = REPO_ROOT / "infer_phone_ec_cond.py"

MIN_SPLIT_SEC = 6.0
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
CONTEXT_FRAMES = 50  # ±25 frames @ 50 fps ≈ ±0.50 s
OUTPUT_FS = 16_000
_SEGMENT_RE = re.compile(r"_segment_(\d+)\.wav$", re.IGNORECASE)
STAGES = ("segment", "transcribe", "align", "inpaint", "stitch")
_T_ALIGN_LABELS = frozenset({canonical_user_phone("t")})
_SHORT_SZ_PHONES = frozenset({"S", "Z"})
