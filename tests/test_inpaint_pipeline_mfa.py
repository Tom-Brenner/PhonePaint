"""Integration smoke: inpaint_pipeline.py --mfa on a VCTK clip."""

from __future__ import annotations

import json
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
MFA_ALIGN_SCRIPT = REPO_ROOT / "tools" / "mfa" / "mfa_align_json.py"
PIPELINE = REPO_ROOT / "inpaint_pipeline.py"

# pipeline_scripts/speech.py pins --mfa_model_pairs english_mfa:english_mfa.
MFA_MODEL = "english_mfa"
# Sidecar env built by ./create_mfa_envs.sh (the --env-mfa default).
ENV_MFA = "mfa_env"
# Roots mfa_align_json.py links into its temp MFA_ROOT_DIR.
PRETRAINED_DIRS = (
    Path.home() / "Documents" / "MFA" / "pretrained_models",
    Path.home() / ".local" / "share" / "Montreal-Forced-Alignment" / "pretrained_models",
)


def _find_mfa_bin() -> Path | None:
    """Locate the MFA CLI the way pipeline_scripts.runtime.resolve_mfa_bin does.

    Duplicated rather than imported because ``pipeline_scripts/__init__`` pulls
    in the whole inference stack, which is far more than a skip guard needs.
    """
    explicit = os.environ.get("MFA_BIN")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mfa")
        candidates.append(Path(conda_prefix).parent / ENV_MFA / "bin" / "mfa")
    home = Path.home()
    for base in ("miniconda3", "anaconda3", "mambaforge", "miniforge3"):
        candidates.append(home / base / "envs" / ENV_MFA / "bin" / "mfa")
    which = shutil.which("mfa")
    if which:
        candidates.append(Path(which))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_pretrained_dir() -> Path | None:
    for base in PRETRAINED_DIRS:
        acoustic = base / "acoustic" / f"{MFA_MODEL}.zip"
        dictionary = base / "dictionary" / f"{MFA_MODEL}.dict"
        if acoustic.is_file() and dictionary.is_file():
            return base
    return None


def _missing_reasons() -> list[str]:
    reasons: list[str] = []
    if not FIXTURE_WAV.is_file():
        reasons.append(f"missing fixture {FIXTURE_WAV}")
    if not PIPELINE.is_file():
        reasons.append(f"missing {PIPELINE}")
    if not CHECKPOINT.is_file():
        reasons.append(f"missing checkpoint {CHECKPOINT}")
    if not MFA_ALIGN_SCRIPT.is_file():
        reasons.append(f"missing MFA align script {MFA_ALIGN_SCRIPT}")
    if _find_mfa_bin() is None:
        reasons.append("no MFA CLI (set MFA_BIN or run ./create_mfa_envs.sh)")
    if _find_pretrained_dir() is None:
        reasons.append(f"no {MFA_MODEL} acoustic model + dictionary downloaded")
    return reasons


def _require_assets() -> bool:
    """True when a skipped smoke test should be treated as a failure.

    Skipping keeps this usable on a laptop without MFA installed, but on CI or
    a remote box a silent skip still reports OK, which is indistinguishable
    from a real pass. Set PHONEPAINT_REQUIRE_ASSETS=1 there.
    """
    value = os.environ.get("PHONEPAINT_REQUIRE_ASSETS", "").strip().lower()
    return value not in ("", "0", "false", "no")


class InpaintPipelineMfaTest(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_reasons()
        if not missing:
            return
        detail = "inpaint_pipeline MFA smoke prerequisites: " + "; ".join(missing)
        if _require_assets():
            self.fail(f"{detail} [PHONEPAINT_REQUIRE_ASSETS is set]")
        self.skipTest(detail)

    def test_mfa(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phonepaint_mfa_") as tmp:
            tmp_path = Path(tmp)
            work_dir = tmp_path / "work"
            output = tmp_path / f"{FIXTURE_WAV.stem}_out.wav"
            cmd = [
                sys.executable,
                str(PIPELINE),
                "--mfa",
                # One Kaldi job: the clip is short enough to stay a single segment.
                "--align-threads",
                "1",
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
            # Prefer repo cwd so relative tools/mfa paths resolve.
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

            # Align is the stage this test exists for, so assert it produced
            # phones rather than only checking that the run exited 0.
            aligned_path = work_dir / "03_aligned.json"
            self.assertTrue(aligned_path.is_file(), f"missing {aligned_path}")
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
            entry = aligned.get(FIXTURE_WAV.name)
            self.assertIsNotNone(
                entry, f"no {FIXTURE_WAV.name} entry in {sorted(aligned)}"
            )
            self.assertTrue(entry.get("phones"), "MFA returned no phones")
            self.assertTrue(entry.get("words"), "MFA returned no words")

            self.assertTrue(output.is_file(), f"missing output wav: {output}")
            info = sf.info(str(output))
            self.assertGreater(info.duration, 0.5)
            self.assertEqual(info.samplerate, 16_000)


if __name__ == "__main__":
    unittest.main()
