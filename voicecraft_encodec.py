"""
Shared loader for the VoiceCraft 16 kHz SEANet EnCodec.

Used by both prep_encodec_cond.py (encoder-only) and infer_phone_ec.py
(encoder + decoder).  Import this module instead of duplicating the loading
logic.

Architecture: encodec_4cb2048_giga.th
  - SEANetEncoder / SEANetDecoder, dimension=128, ratios=[8,5,4,2]
  - 16 kHz audio, hop=320 samples -> 50 fps
  - VoiceCraft audiocraft fork under VOICECRAFT_ROOT/src/audiocraft
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Iterator

import os

import torch

_REPO_ROOT = Path(__file__).resolve().parent


def _default_voicecraft_root() -> Path:
    """Prefer env, then repo third_party/, then legacy home paths."""
    env = os.environ.get("VOICECRAFT_ROOT")
    if env:
        return Path(env).expanduser()
    candidates = (
        _REPO_ROOT / "third_party" / "VoiceCraft",
        Path.home() / "VoiceCraft",
        Path("/home/tom/VoiceCraft"),  # legacy machine-local default
    )
    for cand in candidates:
        if (cand / "src" / "audiocraft").is_dir():
            return cand
    return candidates[0]


VOICECRAFT_ROOT = _default_voicecraft_root()
AUDIOCRAFT_SRC = VOICECRAFT_ROOT / "src" / "audiocraft"
DEFAULT_CHECKPOINT = VOICECRAFT_ROOT / "pretrained_models" / "encodec_4cb2048_giga.th"

SAMPLE_RATE = 16_000
FRAME_RATE = 50.0       # 16000 / 320
HOP_LENGTH = 320        # samples per frame
DIMENSION = 128

# SEANet architecture kwargs shared by both encoder and decoder.
_SEANET_KWARGS = dict(
    dimension=128,
    channels=1,
    causal=False,
    n_filters=64,
    n_residual_layers=1,
    ratios=[8, 5, 4, 2],
    activation="ELU",
    activation_params={"alpha": 1.0},
    norm="weight_norm",
    norm_params={},
    kernel_size=7,
    residual_kernel_size=3,
    last_kernel_size=7,
    dilation_base=2,
    pad_mode="constant",
    true_skip=True,
    compress=2,
    lstm=2,
    disable_norm_outer_blocks=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_audiocraft_modules():
    """Import SEANet from VoiceCraft audiocraft without pulling in xformers etc."""
    if str(AUDIOCRAFT_SRC) not in sys.path:
        sys.path.insert(0, str(AUDIOCRAFT_SRC))
    if "audiocraft" not in sys.modules:
        pkg = types.ModuleType("audiocraft")
        pkg.__path__ = [str(AUDIOCRAFT_SRC / "audiocraft")]
        sys.modules["audiocraft"] = pkg
    if "audiocraft.modules" not in sys.modules:
        mod = types.ModuleType("audiocraft.modules")
        mod.__path__ = [str(AUDIOCRAFT_SRC / "audiocraft" / "modules")]
        sys.modules["audiocraft.modules"] = mod
    importlib.import_module("audiocraft.modules.conv")
    importlib.import_module("audiocraft.modules.lstm")
    return importlib.import_module("audiocraft.modules.seanet")


def _stub_omegaconf_for_checkpoint_unpickle():
    """VoiceCraft checkpoints pickle OmegaConf objects; stub them for torch.load."""
    if "omegaconf.dictconfig" in sys.modules:
        return

    class DictConfig(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key) from None

        def __setattr__(self, key, value):
            self[key] = value

    class ListConfig(list):
        pass

    def _cls(name):
        return type(name, (), {})

    modules = {
        "omegaconf.base": {
            "Metadata": _cls("Metadata"),
            "ContainerMetadata": _cls("ContainerMetadata"),
            "Node": _cls("Node"),
        },
        "omegaconf.nodes": {
            "ValueNode": _cls("ValueNode"),
            "AnyNode": _cls("AnyNode"),
        },
        "omegaconf.dictconfig": {"DictConfig": DictConfig},
        "omegaconf.listconfig": {"ListConfig": ListConfig},
        "omegaconf": {
            "DictConfig": DictConfig,
            "ListConfig": ListConfig,
            "OmegaConf": _cls("OmegaConf"),
        },
    }
    for path, attrs in modules.items():
        mod = types.ModuleType(path)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[path] = mod


def _load_model_state(checkpoint: Path) -> dict:
    _stub_omegaconf_for_checkpoint_unpickle()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "best_state" not in state or "model" not in state["best_state"]:
        raise ValueError(f"Not a VoiceCraft audiocraft checkpoint: {checkpoint}")
    return state["best_state"]["model"]


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VoiceCraftEncodec:
    """
    Thin wrapper around the SEANet encoder (and optionally decoder) extracted
    from a VoiceCraft checkpoint.

    Attributes
    ----------
    encoder : SEANetEncoder
    decoder : SEANetDecoder or None   (None when loaded with decoder=False)
    sample_rate : int
    frame_rate  : float
    """

    def __init__(self, encoder, decoder, sample_rate: int, frame_rate: float):
        self.encoder = encoder
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

    def eval(self) -> "VoiceCraftEncodec":
        self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        return self

    def to(self, device) -> "VoiceCraftEncodec":
        self.encoder.to(device)
        if self.decoder is not None:
            self.decoder.to(device)
        return self

    def parameters(self) -> Iterator:
        yield from self.encoder.parameters()
        if self.decoder is not None:
            yield from self.decoder.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def load_voicecraft_encodec(
    device,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    load_decoder: bool = True,
) -> VoiceCraftEncodec:
    """
    Load the VoiceCraft 16 kHz SEANet EnCodec from *checkpoint*.

    Parameters
    ----------
    device      : torch device or str
    checkpoint  : path to encodec_4cb2048_giga.th (defaults to the canonical location)
    load_decoder: if False, skip loading the decoder (saves memory during prep)
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"VoiceCraft EnCodec checkpoint not found: {checkpoint}\n"
            f"Expected: {DEFAULT_CHECKPOINT}"
        )

    seanet_mod = _ensure_audiocraft_modules()
    model_state = _load_model_state(checkpoint)

    enc_state = {
        k[len("encoder."):]: v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    if not enc_state:
        raise ValueError(f"No encoder weights found in checkpoint: {checkpoint}")
    encoder = seanet_mod.SEANetEncoder(**_SEANET_KWARGS)
    encoder.load_state_dict(enc_state)

    decoder = None
    if load_decoder:
        dec_state = {
            k[len("decoder."):]: v
            for k, v in model_state.items()
            if k.startswith("decoder.")
        }
        if not dec_state:
            raise ValueError(f"No decoder weights found in checkpoint: {checkpoint}")
        decoder = seanet_mod.SEANetDecoder(**_SEANET_KWARGS)
        decoder.load_state_dict(dec_state)

    frame_rate = SAMPLE_RATE / encoder.hop_length
    model = VoiceCraftEncodec(encoder, decoder, SAMPLE_RATE, frame_rate)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
