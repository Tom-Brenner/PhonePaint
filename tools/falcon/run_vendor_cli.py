#!/usr/bin/env python3
"""Run FALCON vendor CLI with inference-only stubs for unused training imports."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

VENDOR = Path(__file__).resolve().parent / "vendor" / "FALCON"


def _stub_module(name: str, **attrs: object) -> None:
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


def bootstrap() -> None:
    def _hydra_main(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    _stub_module("wandb")
    _stub_module("memory_profiler", profile=lambda fn: fn)
    _stub_module("hydra", main=_hydra_main)
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_vendor_cli.py <vendor_script.py> [args...]")
    script = VENDOR / sys.argv[1]
    if not script.is_file():
        raise SystemExit(f"vendor script not found: {script}")
    bootstrap()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
