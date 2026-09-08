"""Subprocess execution and MFA executable discovery."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)

def run_python(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    """Run a stage with this environment's interpreter."""
    cmd = list(cmd)
    if cmd and cmd[0] in {"python", Path(sys.executable).name}:
        cmd[0] = sys.executable
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)



def active_conda_env_name() -> Optional[str]:
    """Return the active conda env name, or None if not in a conda env."""
    name = (os.environ.get("CONDA_DEFAULT_ENV") or "").strip()
    if name:
        return name
    prefix = (os.environ.get("CONDA_PREFIX") or "").strip()
    if prefix:
        return Path(prefix).name
    return None


def is_phonepaint_mfa_env() -> bool:
    """True when the active interpreter is the legacy in-env MFA+MAPS stack."""
    return active_conda_env_name() == "PhonePaintMFA"


def resolve_mfa_bin(env_mfa: str) -> str:
    """Absolute path to the MFA CLI (subprocess; not imported in-process).

    Prefers MFA in the active conda env. The shipped ``PhonePaint`` env has no
    in-env MFA, so it falls back to ``env_mfa`` (default ``mfa_env``), the
    sidecar built by ``./create_mfa_envs.sh``. Legacy ``PhonePaintMFA`` keeps
    MFA in-env and never hops to the sidecar.
    """
    explicit = os.environ.get("MFA_BIN")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        raise SystemExit(f"MFA_BIN is set but not a file: {explicit}")

    candidates: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            which_path = Path(which).resolve()
            prefix_path = Path(conda_prefix).resolve()
            try:
                which_path.relative_to(prefix_path)
                candidates.append(which_path)
            except ValueError:
                pass

    allow_sidecar = not is_phonepaint_mfa_env()
    if allow_sidecar:
        if conda_prefix:
            candidates.append(Path(conda_prefix).parent / env_mfa / "bin" / "mfa")
        home = Path.home()
        for base in (home / "miniconda3", home / "anaconda3", home / "mambaforge", home / "miniforge3"):
            candidates.append(base / "envs" / env_mfa / "bin" / "mfa")
        which = shutil.which("mfa")
        if which:
            candidates.append(Path(which))
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return str(cand)
    if is_phonepaint_mfa_env():
        env = active_conda_env_name() or "PhonePaint"
        raise SystemExit(
            f"{env} is active but MFA CLI was not found in this env "
            f"($CONDA_PREFIX/bin/mfa). Install MFA in-env or set MFA_BIN; "
            f"the mfa_env sidecar is not used while {env} is active."
        )
    raise SystemExit(
        f"MFA CLI not found in the active env or fallback {env_mfa!r}. "
        "Install MFA in-env, run ./create_mfa_envs.sh, or set MFA_BIN."
    )

