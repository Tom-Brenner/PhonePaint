"""Integration smoke: inpaint_pipeline.py --maps --maps-ensemble on a VCTK clip."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
# VCTK p225_001 "Please call Stella." — 2.05 s, one /s/ for the run to act on.
FIXTURE_WAV = Path(__file__).resolve().parent / "p225_001_mic2.wav"
CHECKPOINT = REPO_ROOT / "weights" / "best.pt"
MAPS_ENSEMBLE = REPO_ROOT / "tools" / "maps" / "torch_models" / "ensemble_model"
PIPELINE = REPO_ROOT / "inpaint_pipeline.py"


def _missing_reasons() -> list[str]:
    reasons: list[str] = []
    if not FIXTURE_WAV.is_file():
        reasons.append(f"missing fixture {FIXTURE_WAV}")
    if not PIPELINE.is_file():
        reasons.append(f"missing {PIPELINE}")
    if not CHECKPOINT.is_file():
        reasons.append(f"missing checkpoint {CHECKPOINT}")
    if not MAPS_ENSEMBLE.is_dir():
        reasons.append(f"missing MAPS ensemble dir {MAPS_ENSEMBLE}")
    return reasons


def _require_assets() -> bool:
    """True when a skipped smoke test should be treated as a failure.

    Skipping keeps this usable on a laptop without checkpoints, but on CI or a
    remote box a silent skip still reports OK, which is indistinguishable from
    a real pass. Set PHONEPAINT_REQUIRE_ASSETS=1 there.
    """
    value = os.environ.get("PHONEPAINT_REQUIRE_ASSETS", "").strip().lower()
    return value not in ("", "0", "false", "no")


class InpaintPipelineMapsTest(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_reasons()
        if not missing:
            return
        detail = "inpaint_pipeline MAPS smoke prerequisites: " + "; ".join(missing)
        if _require_assets():
            self.fail(f"{detail} [PHONEPAINT_REQUIRE_ASSETS is set]")
        self.skipTest(detail)

    def test_maps_ensemble(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phonepaint_maps_") as tmp:
            tmp_path = Path(tmp)
            work_dir = tmp_path / "work"
            output = tmp_path / f"{FIXTURE_WAV.stem}_out.wav"
            cmd = [
                sys.executable,
                str(PIPELINE),
                "--maps",
                "--maps-ensemble",
                "--input",
                str(FIXTURE_WAV),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--checkpoint",
                str(CHECKPOINT),
                "--phones_in",
                "s",
                "--phones_out",
                "s",
            ]
            env = os.environ.copy()
            # Prefer repo cwd so relative tools/maps paths resolve.
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(
                    "inpaint_pipeline.py failed:\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout[-4000:]}\n"
                    f"stderr:\n{proc.stderr[-4000:]}"
                )
            self.assertTrue(output.is_file(), f"missing output wav: {output}")
            info = sf.info(str(output))
            self.assertGreater(info.duration, 0.5)
            self.assertEqual(info.samplerate, 16_000)


if __name__ == "__main__":
    unittest.main()
