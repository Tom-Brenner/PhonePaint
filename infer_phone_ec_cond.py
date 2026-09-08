#!/usr/bin/env python3
"""
Inpaint / replace acoustic spans using a phone-conditioned A3T model
(train_phone_ec_cond.py checkpoints, 16 kHz VoiceCraft EnCodec).

Standalone: does not import or inherit behavior from infer_phone_ec.py /
infer_phone_ec_lip.py.  Pass a spans JSON (start/end[/phone][/follows_t]);
``--ext`` widens each span, including spans with ``follows_t: true``
(default ``--force_ext``). Pass ``--no-force_ext`` to keep ext=0 on those
spans (no bleed into a preceding /t/).

Example:
    python infer_phone_ec_cond.py \\
        --phone s \\
        --wav input.wav --spans spans.json \\
        --checkpoint exp/cond_pretrain/best.pt \\
        --output output.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from espnet.nets.pytorch_backend.conformer.encoder import MLMDecoder, MLMEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
from phone_labels import encode_label_to_ids
from train_phone_ec_cond import (
    build_model,
    build_segment_pos_from_alignment,
    trim_phone_alignment,
)
from voicecraft_encodec import DEFAULT_CHECKPOINT, load_voicecraft_encodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Codec config ──────────────────────────────────────────────────────────────
FS          = 16_000
HOP_SAMP    = 320        # samples per VoiceCraft EnCodec frame (16000 / 50)
ENCODEC_FPS = 50.0

MIN_CONTEXT_SEC = 0.15
MAX_UTT_SEC     = 8.0
RMS_EPS         = 1e-10
# Default context window: ±25 frames ≈ ±0.5 s @ 50 fps.
CONTEXT_FRAMES  = 50
# Extra frames encoded on each side of a crop so SEANet edges match full-file encode.
ENCODER_CTX_PAD_FRAMES = 8
DEFAULT_INFER_BATCH_SIZE = 32

# Keys tried (in order) for per-span conditioning phone in pipeline JSON.
SPAN_PHONE_KEYS = ("phone", "phone_out", "target")

# Full-utterance phone intervals for --full_phone_conditioning (seconds).
PhoneInterval = Tuple[float, float, str]


def apply_ext(
    spans: Sequence[Tuple[float, float]],
    *,
    ext_sec: float,
    max_end: float,
) -> List[Tuple[float, float]]:
    """Extend each span by ext_sec on both sides, clipped to max_end."""
    out: List[Tuple[float, float]] = []
    for start, end in spans:
        s = max(0.0, start - ext_sec)
        e = min(max_end, end + ext_sec)
        if e > s:
            out.append((s, e))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EnCodec helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_encodec(device: torch.device, ckpt_path: Optional[Path] = None):
    checkpoint = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CHECKPOINT
    return load_voicecraft_encodec(device, checkpoint, load_decoder=True)


@torch.no_grad()
def _ec_decode(ec_model, z_lat: torch.Tensor) -> torch.Tensor:
    try:
        wav = ec_model.decoder(z_lat)
    except TypeError:
        scale = torch.ones(z_lat.size(0), device=z_lat.device, dtype=z_lat.dtype)
        wav = ec_model.decoder(z_lat, scale)
    return wav.squeeze(1) if wav.dim() == 3 else wav


@torch.no_grad()
def extract_ze_t(
    ec_model,
    wav_np: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Encode wav to z_e latent frames [T, D] on device."""
    pcm = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0).to(device)
    z_e = ec_model.encoder(pcm)  # [1, D, T]
    return z_e.squeeze(0).t().contiguous()


def expected_ze_len(n_samples: int) -> int:
    """Frame count for a wav of length n_samples (hop-aligned)."""
    return n_samples // HOP_SAMP


def merge_context_frame_windows(
    spans_sec: Sequence[Tuple[float, float]],
    ze_len: int,
    context_frames: int,
) -> List[Tuple[int, int]]:
    """Merge overlapping MLM context windows into (f0, f1) frame ranges."""
    half = context_frames // 2
    regions: List[List[int]] = []
    for start, end in spans_sec:
        paste_f0 = max(0, int(start * ENCODEC_FPS))
        paste_f1 = min(ze_len, int(np.ceil(end * ENCODEC_FPS)))
        if paste_f1 <= paste_f0:
            continue
        regions.append([max(0, paste_f0 - half), min(ze_len, paste_f1 + half)])
    if not regions:
        raise ValueError("No encode regions after span quantisation")
    regions.sort(key=lambda r: r[0])
    merged: List[List[int]] = [regions[0]]
    for f0, f1 in regions[1:]:
        prev = merged[-1]
        if f0 <= prev[1]:
            prev[1] = max(prev[1], f1)
        else:
            merged.append([f0, f1])
    return [(r[0], r[1]) for r in merged]


@torch.no_grad()
def extract_ze_windowed_t(
    ec_model,
    wav_np: np.ndarray,
    device: torch.device,
    regions: Sequence[Tuple[int, int]],
    ze_len: int,
    *,
    encoder_pad_frames: int = ENCODER_CTX_PAD_FRAMES,
) -> torch.Tensor:
    """Encode only listed frame windows; return full-length [T, D] (zeros elsewhere)."""
    if not regions:
        raise ValueError("extract_ze_windowed_t requires at least one region")
    z_e: Optional[torch.Tensor] = None
    n_encoded = 0
    for f0, f1 in regions:
        need = f1 - f0
        if need <= 0:
            continue
        enc_f0 = max(0, f0 - encoder_pad_frames)
        enc_f1 = min(ze_len, f1 + encoder_pad_frames)
        s0 = enc_f0 * HOP_SAMP
        s1 = min(enc_f1 * HOP_SAMP, len(wav_np))
        z_crop = extract_ze_t(ec_model, wav_np[s0:s1], device)
        off = f0 - enc_f0
        if z_crop.shape[0] < off + need:
            pad_n = off + need - z_crop.shape[0]
            z_crop = torch.nn.functional.pad(z_crop, (0, 0, 0, pad_n))
        z_mid = z_crop[off:off + need]
        if z_e is None:
            z_e = torch.zeros(
                ze_len, z_mid.shape[1], device=device, dtype=z_mid.dtype,
            )
        z_e[f0:f1] = z_mid
        n_encoded += need
    if z_e is None:
        raise ValueError("No frames encoded in windowed extract")
    log.info(
        "Windowed encode: %d region(s), %d / %d frames (%.1f%%)",
        len(regions), n_encoded, ze_len, 100.0 * n_encoded / max(ze_len, 1),
    )
    return z_e


@torch.no_grad()
def decode_ze_crop_t(ec_model, z_e_crop: torch.Tensor) -> torch.Tensor:
    """Decode a z_e crop [T, D] on device → mono wav [N_samples] on device."""
    z_lat = z_e_crop.unsqueeze(0).transpose(1, 2).contiguous()
    return _ec_decode(ec_model, z_lat).squeeze(0)


def build_decode_regions(
    spans_sec: Sequence[Tuple[float, float]],
    ze_len: int,
    decode_ctx_frames: int,
) -> List[Tuple[int, int, int, int]]:
    """Merge overlapping decode windows.

    Returns (dec_f0, dec_f1, paste_f0, paste_f1) in EnCodec frame indices.
    [dec_f0:dec_f1) is decoded; [paste_f0:paste_f1) is pasted into the output wav.
    """
    regions: List[List[int]] = []
    for start, end in spans_sec:
        paste_f0 = max(0, int(start * ENCODEC_FPS))
        paste_f1 = min(ze_len, int(np.ceil(end * ENCODEC_FPS)))
        if paste_f1 <= paste_f0:
            continue
        dec_f0 = max(0, paste_f0 - decode_ctx_frames)
        dec_f1 = min(ze_len, paste_f1 + decode_ctx_frames)
        regions.append([dec_f0, dec_f1, paste_f0, paste_f1])

    if not regions:
        raise ValueError("No decode regions after span quantisation")

    regions.sort(key=lambda r: r[0])
    merged: List[List[int]] = [regions[0]]
    for dec_f0, dec_f1, paste_f0, paste_f1 in regions[1:]:
        prev = merged[-1]
        if dec_f0 <= prev[1]:
            prev[1] = max(prev[1], dec_f1)
            prev[2] = min(prev[2], paste_f0)
            prev[3] = max(prev[3], paste_f1)
        else:
            merged.append([dec_f0, dec_f1, paste_f0, paste_f1])
    return [tuple(r) for r in merged]


def _crossfade_ramp(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.linspace(0.0, 1.0, length, dtype=np.float64)


def paste_patch(
    out_wav: np.ndarray,
    s0: int,
    s1: int,
    patch: np.ndarray,
    *,
    crossfade_samples: int,
) -> None:
    """Paste decoded patch into out_wav[s0:s1], optionally crossfading at boundaries."""
    n = s1 - s0
    if n <= 0:
        return
    patch = patch[:n].astype(np.float64, copy=False)
    fade = min(crossfade_samples, n // 2, s0, len(out_wav) - s1)
    if fade <= 0:
        out_wav[s0:s1] = patch
        return

    ramp = _crossfade_ramp(fade)
    out_wav[s0:s0 + fade] = (
        out_wav[s0:s0 + fade].astype(np.float64) * (1.0 - ramp) + patch[:fade] * ramp
    )
    if n > 2 * fade:
        out_wav[s0 + fade:s1 - fade] = patch[fade:n - fade]
    out_wav[s1 - fade:s1] = (
        patch[n - fade:n] * (1.0 - ramp) + out_wav[s1 - fade:s1].astype(np.float64) * ramp
    )


@torch.no_grad()
def stitch_decoded_patches(
    out_wav: np.ndarray,
    z_e_t: torch.Tensor,
    regions: Sequence[Tuple[int, int, int, int]],
    ec_model,
    *,
    crossfade_samples: int,
) -> None:
    """Decode only the listed z_e regions and paste into out_wav (in-place)."""
    for dec_f0, dec_f1, paste_f0, paste_f1 in regions:
        z_crop = z_e_t[0, dec_f0:dec_f1, :]
        wav_dec = decode_ze_crop_t(ec_model, z_crop).detach().cpu().numpy()

        s0 = paste_f0 * HOP_SAMP
        s1 = min(paste_f1 * HOP_SAMP, len(out_wav))
        if s1 <= s0:
            continue

        paste_off0 = (paste_f0 - dec_f0) * HOP_SAMP
        need = s1 - s0
        patch = wav_dec[paste_off0:paste_off0 + need]
        if len(patch) < need:
            patch = np.pad(patch, (0, need - len(patch)))
        elif len(patch) > need:
            patch = patch[:need]

        paste_patch(
            out_wav, s0, s1, patch,
            crossfade_samples=crossfade_samples,
        )
        log.info(
            "Patched samples %d–%d (frames %d–%d, decoded %d frames)",
            s0, s1, paste_f0, paste_f1, dec_f1 - dec_f0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Span parsing and edge padding
# ─────────────────────────────────────────────────────────────────────────────

LabeledSpan = Tuple[float, float, Optional[str], bool]


def _parse_span_item(item) -> LabeledSpan:
    phone: Optional[str] = None
    follows_t = False
    if isinstance(item, (list, tuple)):
        if len(item) < 2:
            raise ValueError(f"Span list/tuple needs at least start/end: {item!r}")
        start, end = float(item[0]), float(item[1])
        if len(item) >= 3:
            phone = str(item[2])
        if len(item) >= 4:
            follows_t = bool(item[3])
    elif isinstance(item, dict):
        if "start" in item and "end" in item:
            start, end = float(item["start"]), float(item["end"])
        elif "xmin" in item and "xmax" in item:
            start, end = float(item["xmin"]), float(item["xmax"])
        else:
            raise ValueError(f"Span dict needs start/end or xmin/xmax: {item!r}")
        for key in SPAN_PHONE_KEYS:
            if key in item and item[key] is not None:
                phone = str(item[key])
                break
        follows_t = bool(item.get("follows_t", False))
    else:
        raise ValueError(f"Unsupported span entry: {item!r}")
    if end <= start:
        raise ValueError(f"Span end must be > start: ({start}, {end})")
    return start, end, phone, follows_t


def parse_labeled_spans(data: dict) -> List[LabeledSpan]:
    if "spans" not in data:
        raise ValueError('JSON must contain a top-level "spans" list')
    spans = [_parse_span_item(item) for item in data["spans"]]
    if not spans:
        raise ValueError("No spans found in JSON")
    return spans


def parse_spans(data: dict) -> List[Tuple[float, float]]:
    return [(start, end) for start, end, _, _ in parse_labeled_spans(data)]


def _parse_phone_interval_item(item) -> PhoneInterval:
    if isinstance(item, (list, tuple)):
        if len(item) < 3:
            raise ValueError(
                f"phone_intervals entry needs [start, end, text], got: {item!r}"
            )
        start, end, text = float(item[0]), float(item[1]), str(item[2])
    elif isinstance(item, dict):
        if "start" in item and "end" in item:
            start, end = float(item["start"]), float(item["end"])
        elif "xmin" in item and "xmax" in item:
            start, end = float(item["xmin"]), float(item["xmax"])
        else:
            raise ValueError(
                f"phone_intervals dict needs start/end or xmin/xmax: {item!r}"
            )
        text = item.get("text")
        if text is None:
            raise ValueError(f"phone_intervals dict missing text: {item!r}")
        text = str(text)
    else:
        raise ValueError(f"Unsupported phone_intervals entry: {item!r}")
    if end <= start:
        raise ValueError(f"phone interval end must be > start: ({start}, {end})")
    return start, end, text


def parse_phone_intervals(data: dict) -> List[PhoneInterval]:
    """Parse full-utterance phone intervals from spans JSON."""
    raw = data.get("phone_intervals")
    if raw is None:
        raw = data.get("alignment")
    if raw is None:
        raise ValueError(
            'JSON must contain "phone_intervals" (or "alignment") when '
            "using --full_phone_conditioning"
        )
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("phone_intervals must be a non-empty list")
    intervals = [_parse_phone_interval_item(item) for item in raw]
    intervals.sort(key=lambda x: x[0])
    return intervals


def build_phone_seq_tensors(
    phone_intervals: Sequence[PhoneInterval],
    ze_len: int,
    char2id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Char-tokenize phone intervals into SEGA alignment tensors (frame indices)."""
    phone_ids: List[int] = []
    phone_starts: List[int] = []
    phone_ends: List[int] = []
    phone_nchars: List[int] = []

    for start, end, label in phone_intervals:
        char_ids = encode_label_to_ids(label, char2id)
        if not char_ids:
            continue
        f0 = max(0, int(start * ENCODEC_FPS))
        f1 = min(ze_len, int(np.ceil(end * ENCODEC_FPS)))
        if f1 <= f0:
            continue
        phone_starts.append(f0)
        phone_ends.append(f1)
        phone_nchars.append(len(char_ids))
        phone_ids.extend(char_ids)

    if not phone_ids:
        raise ValueError(
            "phone_intervals produced no valid char tokens "
            "(check labels against checkpoint char vocab)"
        )
    return (
        torch.tensor(phone_ids, dtype=torch.long),
        torch.tensor(phone_starts, dtype=torch.long),
        torch.tensor(phone_ends, dtype=torch.long),
        torch.tensor(phone_nchars, dtype=torch.long),
    )


def apply_edge_padding(
    wav: np.ndarray,
    spans_sec: List[Tuple[float, float]],
    min_context: float = MIN_CONTEXT_SEC,
    max_utt_sec: float = MAX_UTT_SEC,
) -> Tuple[np.ndarray, List[Tuple[float, float]], int, int]:
    utt_dur = len(wav) / FS
    utt_end = min(utt_dur, max_utt_sec)
    clipped = [(s, e) for s, e in spans_sec if s < utt_end]
    if not clipped:
        raise ValueError("All spans fall outside the usable utterance window")
    wav = wav[: int(round(utt_end * FS))]
    first_start = min(s for s, _ in clipped)
    last_end    = max(e for _, e in clipped)
    pad_pre  = max(0.0, min_context - first_start)
    pad_post = max(0.0, min_context - (utt_end - last_end))
    pre_samp  = int(round(pad_pre * FS))
    post_samp = int(round(pad_post * FS))
    if pre_samp > 0:
        wav = np.concatenate([np.zeros(pre_samp, np.float32), wav])
    if post_samp > 0:
        wav = np.concatenate([wav, np.zeros(post_samp, np.float32)])
    shifted = [(s + pad_pre, e + pad_pre) for s, e in clipped]
    return wav, shifted, pre_samp, post_samp


def spans_to_frame_mask(
    spans_sec: Sequence[Tuple[float, float]],
    ze_len: int,
) -> np.ndarray:
    mask = np.zeros(ze_len, dtype=bool)
    for t_start, t_end in spans_sec:
        f_start = max(0, int(t_start * ENCODEC_FPS))
        f_end   = min(ze_len, int(np.ceil(t_end * ENCODEC_FPS)))
        if f_end > f_start:
            mask[f_start:f_end] = True
    if not mask.any():
        raise ValueError("Spans did not cover any EnCodec frames after quantisation")
    return mask


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + RMS_EPS))


def spans_to_sample_mask(
    spans_sec: Sequence[Tuple[float, float]],
    n_samples: int,
    pad_pre_samp: int,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for t_start, t_end in spans_sec:
        s = max(0, int(t_start * FS) - pad_pre_samp)
        e = min(n_samples, int(np.ceil(t_end * FS)) - pad_pre_samp)
        if e > s:
            mask[s:e] = True
    return mask


def normalize_inpaint_volume(
    wav: np.ndarray,
    spans_sec: Sequence[Tuple[float, float]],
    pad_pre_samp: int,
    vol_scale: float,
) -> np.ndarray:
    mask = spans_to_sample_mask(spans_sec, len(wav), pad_pre_samp)
    if not mask.any():
        log.warning("normalize_vol: no inpainted samples in output; skipping")
        return wav
    context = ~mask
    if not context.any():
        log.warning("normalize_vol: no context samples in output; skipping")
        return wav
    ref_rms     = _rms(wav[context])
    inpaint_rms = _rms(wav[mask])
    gain        = (ref_rms / inpaint_rms) * vol_scale
    out = wav.copy()
    out[mask] *= gain
    log.info(
        "normalize_vol: context_rms=%.6f  inpaint_rms=%.6f  scale=%.3f  gain=%.3f",
        ref_rms, inpaint_rms, vol_scale, gain,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[object, Dict[str, int], bool]:
    """Load a train_phone_ec_cond.py checkpoint.

    Returns (model, token→id, full_phone_conditioning_from_ckpt).
    """
    ckpt = torch.load(path, map_location=device)

    token_list = ckpt.get("token_list")
    if not token_list:
        raise ValueError(
            f"Checkpoint {path} has no token_list — "
            "was it saved by train_phone_ec_cond.py?"
        )

    model_config = ckpt.get("model_config")
    if not model_config:
        raise ValueError(
            f"Checkpoint {path} has no model_config — "
            "was it saved by train_phone_ec_cond.py?"
        )

    fake_args = types.SimpleNamespace(
        attn_dim       = model_config["attn_dim"],
        attn_heads     = model_config["attn_heads"],
        ffn_units      = model_config["ffn_units"],
        enc_blocks     = model_config["enc_blocks"],
        dec_blocks     = model_config["dec_blocks"],
        dropout        = model_config.get("dropout", 0.0),
        postnet_layers = model_config.get("postnet_layers", 3),
        postnet_chans  = model_config.get("postnet_chans", 128),
        postnet_filts  = model_config.get("postnet_filts", 5),
    )

    model = build_model(token_list, fake_args)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    phone2id = {t: i for i, t in enumerate(token_list)}
    ckpt_full_phone = bool(model_config.get("full_phone_conditioning", False))
    log.info(
        "Loaded checkpoint %s  (stage=%s  epoch=%s  val_loss=%s  full_phone=%s)",
        path,
        ckpt.get("stage", "?"),
        ckpt.get("epoch", "?"),
        ckpt.get("val_loss", "?"),
        ckpt_full_phone,
    )
    return model, phone2id, ckpt_full_phone


def cond_token_id_for_phone(
    phone: str,
    phone2id: Dict[str, int],
    *,
    log_choice: bool = True,
) -> int:
    if phone in phone2id:
        if log_choice:
            log.info("Conditioning on phone %r -> token id %d", phone, phone2id[phone])
        return phone2id[phone]
    if log_choice:
        log.warning(
            "Phone %r not in checkpoint token_list %s; using <any> (id 0)",
            phone, list(phone2id),
        )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inpainting forward pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def inpaint_ze(
    model: nn.Module,
    z_e_t: torch.Tensor,            # [B, T, D]
    masked_position: torch.Tensor,  # [B, T] bool
    device: torch.device,
    cond_token_id: Union[int, torch.Tensor] = 0,
    feat_lens: Optional[torch.Tensor] = None,
    *,
    text_pad: Optional[torch.Tensor] = None,
    text_lengths: Optional[torch.Tensor] = None,
    speech_segment_pos: Optional[torch.Tensor] = None,
    text_segment_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inpaint masked z_e frames.

    Single-token mode (default): pass ``cond_token_id``.
    Full-phone mode: pass ``text_pad`` / ``text_lengths`` /
    ``speech_segment_pos`` / ``text_segment_pos`` (SEGA alignment).
    """
    model.eval()
    z_e_t = z_e_t.to(device)
    masked_position = masked_position.to(device)

    b, ze_len, _ = z_e_t.shape
    if feat_lens is None:
        feat_lens = torch.full((b,), ze_len, dtype=torch.long, device=device)
    else:
        feat_lens = feat_lens.to(device)

    speech_mask = make_non_pad_mask(
        feat_lens.tolist(), z_e_t[:, :, 0], length_dim=1
    ).unsqueeze(-2)
    masked_position = masked_position & speech_mask.squeeze(1)

    if text_pad is not None:
        if (
            text_lengths is None
            or speech_segment_pos is None
            or text_segment_pos is None
        ):
            raise ValueError(
                "full-phone inpaint requires text_pad, text_lengths, "
                "speech_segment_pos, and text_segment_pos"
            )
        text_pad = text_pad.to(device)
        text_lengths = text_lengths.to(device)
        speech_seg_pos = speech_segment_pos.to(device)
        text_seg_pos = text_segment_pos.to(device)
        text_mask = make_non_pad_mask(
            text_lengths.tolist(), text_pad, length_dim=1,
        ).unsqueeze(-2)
    else:
        if isinstance(cond_token_id, int):
            text_pad = torch.full((b, 1), cond_token_id, dtype=torch.long, device=device)
        else:
            text_pad = cond_token_id.to(device).view(b, 1)
        text_mask = torch.ones(b, 1, 1, dtype=torch.bool, device=device)
        speech_seg_pos = torch.zeros(b, ze_len, dtype=torch.long, device=device)
        text_seg_pos = torch.ones(b, 1, dtype=torch.long, device=device)

    batch = dict(
        speech_pad         = z_e_t,
        text_pad           = text_pad,
        masked_position    = masked_position,
        speech_mask        = speech_mask,
        text_mask          = text_mask,
        speech_segment_pos = speech_seg_pos,
        text_segment_pos   = text_seg_pos,
    )

    before_outs, after_outs, _, _ = model._forward(batch, speech_seg_pos)
    pred = after_outs if after_outs is not None else before_outs

    out = z_e_t.clone()
    out[masked_position] = pred[masked_position]
    return out


def _full_phone_batch_tensors(
    phone_ids: torch.Tensor,
    phone_starts: torch.Tensor,
    phone_ends: torch.Tensor,
    phone_nchars: torch.Tensor,
    speech_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build batched SEGA tensors for a single sample (B=1)."""
    sp_seg, tx_seg = build_segment_pos_from_alignment(
        speech_len, phone_starts, phone_ends, phone_nchars,
    )
    text_pad = phone_ids.unsqueeze(0).to(device)
    text_lengths = torch.tensor([phone_ids.numel()], dtype=torch.long, device=device)
    return (
        text_pad,
        text_lengths,
        sp_seg.unsqueeze(0).to(device),
        tx_seg.unsqueeze(0).to(device),
    )


def context_slice_for_mask(
    mask_np: np.ndarray,
    length: int,
    context_frames: int,
) -> Tuple[int, int]:
    """Match train_phone_ec_cond MultiPhoneNpzDataset context trimming."""
    half = context_frames // 2
    nz = np.flatnonzero(mask_np)
    if nz.size == 0:
        return 0, length
    span_start = int(nz[0])
    span_end   = int(nz[-1]) + 1
    ctx_start  = max(0, span_start - half)
    ctx_end    = min(length, span_end + half)
    return ctx_start, ctx_end


@torch.no_grad()
def inpaint_ze_windowed(
    model: nn.Module,
    z_e_t: torch.Tensor,
    masked_position: torch.Tensor,
    device: torch.device,
    cond_token_id: int,
    context_frames: int,
    *,
    phone_ids: Optional[torch.Tensor] = None,
    phone_starts: Optional[torch.Tensor] = None,
    phone_ends: Optional[torch.Tensor] = None,
    phone_nchars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run inpaint_ze on a trimmed window and paste the result back."""
    ze_len = z_e_t.shape[1]
    mask_np = masked_position[0].cpu().numpy()
    ctx_start, ctx_end = context_slice_for_mask(mask_np, ze_len, context_frames)
    z_crop = z_e_t[:, ctx_start:ctx_end, :].clone()
    mask_crop = masked_position[:, ctx_start:ctx_end].clone()
    if not mask_crop.any():
        return z_e_t
    wlen = ctx_end - ctx_start
    if phone_ids is not None:
        assert phone_starts is not None and phone_ends is not None and phone_nchars is not None
        ids_t, starts_t, ends_t, nchars_t = trim_phone_alignment(
            phone_ids, phone_starts, phone_ends, phone_nchars, ctx_start, ctx_end,
        )
        text_pad, text_lengths, sp_seg, tx_seg = _full_phone_batch_tensors(
            ids_t, starts_t, ends_t, nchars_t, wlen, device,
        )
        z_out_crop = inpaint_ze(
            model, z_crop, mask_crop, device,
            text_pad=text_pad,
            text_lengths=text_lengths,
            speech_segment_pos=sp_seg,
            text_segment_pos=tx_seg,
        )
    else:
        z_out_crop = inpaint_ze(model, z_crop, mask_crop, device, cond_token_id)
    out = z_e_t.clone()
    out[:, ctx_start:ctx_end, :] = z_out_crop
    return out


@torch.no_grad()
def inpaint_windowed_spans(
    model: nn.Module,
    z_e_t: torch.Tensor,
    labeled_shifted: Sequence[Tuple[float, float, str]],
    *,
    ze_len: int,
    phone2id: Dict[str, int],
    device: torch.device,
    context_frames: int,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    full_phone: bool = False,
    phone_ids: Optional[torch.Tensor] = None,
    phone_starts: Optional[torch.Tensor] = None,
    phone_ends: Optional[torch.Tensor] = None,
    phone_nchars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inpaint each span in its own context window; batch forwards when possible."""
    if not labeled_shifted:
        return z_e_t

    if full_phone:
        if phone_ids is None or phone_starts is None or phone_ends is None or phone_nchars is None:
            raise ValueError("full_phone windowed inpaint requires phone alignment tensors")
    else:
        phone_token_ids = {
            phone: cond_token_id_for_phone(phone, phone2id, log_choice=True)
            for phone in {p for _, _, p in labeled_shifted}
        }

    jobs: List[dict] = []
    for start, end, phone_label in labeled_shifted:
        mask_np = spans_to_frame_mask([(start, end)], ze_len)
        ctx_start, ctx_end = context_slice_for_mask(mask_np, ze_len, context_frames)
        job = dict(
            ctx_start=ctx_start,
            ctx_end=ctx_end,
            local_mask_np=mask_np[ctx_start:ctx_end],
            phone_label=phone_label,
        )
        if full_phone:
            ids_t, starts_t, ends_t, nchars_t = trim_phone_alignment(
                phone_ids, phone_starts, phone_ends, phone_nchars, ctx_start, ctx_end,
            )
            job.update(
                phone_ids=ids_t,
                phone_starts=starts_t,
                phone_ends=ends_t,
                phone_nchars=nchars_t,
            )
        else:
            job["cond_token_id"] = phone_token_ids[phone_label]
        jobs.append(job)

    batch_size = max(1, infer_batch_size)
    n_batches = (len(jobs) + batch_size - 1) // batch_size
    log.info(
        "Windowed inpaint: %d span(s) in %d batch(es) of up to %d%s",
        len(jobs), n_batches, batch_size,
        " (full_phone)" if full_phone else "",
    )

    out = z_e_t.clone()
    for batch_idx in range(0, len(jobs), batch_size):
        chunk = jobs[batch_idx: batch_idx + batch_size]
        crops: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for job in chunk:
            crops.append(out[0, job["ctx_start"]:job["ctx_end"], :].clone())
            masks.append(torch.from_numpy(job["local_mask_np"]).to(device))

        z_batch = pad_list(crops, 0.0)
        mask_batch = pad_list(masks, False)
        lens = torch.tensor([c.size(0) for c in crops], dtype=torch.long, device=device)

        if full_phone:
            text_list = [job["phone_ids"] for job in chunk]
            text_lengths = torch.tensor(
                [t.numel() for t in text_list], dtype=torch.long, device=device,
            )
            text_pad = pad_list(text_list, 0).to(device)
            max_tlen = int(text_lengths.max().item()) if text_lengths.numel() else 1
            max_slen = int(lens.max().item())
            speech_seg = torch.zeros(len(chunk), max_slen, dtype=torch.long, device=device)
            text_seg = torch.zeros(len(chunk), max_tlen, dtype=torch.long, device=device)
            for i, job in enumerate(chunk):
                slen = int(lens[i].item())
                sp_seg, tx_seg = build_segment_pos_from_alignment(
                    slen, job["phone_starts"], job["phone_ends"], job["phone_nchars"],
                )
                speech_seg[i, :slen] = sp_seg.to(device)
                text_seg[i, :text_lengths[i]] = tx_seg.to(device)
            z_pred = inpaint_ze(
                model, z_batch, mask_batch, device,
                feat_lens=lens,
                text_pad=text_pad,
                text_lengths=text_lengths,
                speech_segment_pos=speech_seg,
                text_segment_pos=text_seg,
            )
        else:
            cond_tensor = torch.tensor(
                [job["cond_token_id"] for job in chunk],
                dtype=torch.long, device=device,
            )
            z_pred = inpaint_ze(
                model, z_batch, mask_batch, device,
                cond_token_id=cond_tensor,
                feat_lens=lens,
            )

        for i, job in enumerate(chunk):
            wlen = job["ctx_end"] - job["ctx_start"]
            cs, ce = job["ctx_start"], job["ctx_end"]
            local_mask = mask_batch[i, :wlen]
            region = out[0, cs:ce, :].clone()
            region[local_mask] = z_pred[i, :wlen, :][local_mask]
            out[0, cs:ce, :] = region

        log.info(
            "  batch %d/%d — %d span(s), window lengths %s",
            batch_idx // batch_size + 1,
            n_batches,
            len(chunk),
            [job["ctx_end"] - job["ctx_start"] for job in chunk],
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Top-level inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_cond_models(
    checkpoint: Path,
    device_t: torch.device,
    encodec_ckpt: Optional[Path] = None,
) -> Tuple[object, nn.Module, Dict[str, int], bool]:
    """Load the VoiceCraft EnCodec + phone-EC checkpoint once.

    Returns (ec_model, model, phone2id, ckpt_full_phone) for reuse across any
    number of inpaint jobs (see ``run_batch_inference``).
    """
    log.info("Loading VoiceCraft EnCodec…")
    ec_model = load_encodec(device_t, ckpt_path=encodec_ckpt)
    model, phone2id, ckpt_full_phone = load_checkpoint(checkpoint, device_t)
    return ec_model, model, phone2id, ckpt_full_phone


def _infer_one_cond(
    ec_model,
    model: nn.Module,
    phone2id: Dict[str, int],
    ckpt_full_phone: bool,
    *,
    phone: Optional[str],
    wav_path: Path,
    spans_path: Path,
    output_path: Path,
    device_t: torch.device,
    resampled_path: Optional[Path] = None,
    max_utt_sec: float = float("inf"),
    ola: bool = False,
    ola_chunk_frames: int = 32,
    ola_hop_factor: int = 4,
    ola_ctx_frames: int = 8,
    normalize_vol: Optional[float] = None,
    ext_sec: float = 0.0,
    force_ext: bool = True,
    windowed_attention: bool = False,
    context_frames: int = CONTEXT_FRAMES,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    full_phone_conditioning: bool = False,
    full_decode: bool = False,
) -> Path:
    """Run one inpaint job against already-loaded EnCodec + phone-EC models.

    This is the per-job core shared by ``run_inference`` (single job; loads
    models itself) and ``run_batch_inference`` (many jobs; loads models once
    and calls this in a loop) — behavior for a single job is identical
    either way.
    """
    if full_decode and windowed_attention:
        raise ValueError("--full-decode and --windowed_attention cannot be combined")
    wav_raw, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav_raw.ndim > 1:
        wav_raw = wav_raw.mean(axis=1)
    if sr != FS:
        log.info("Resampling %d Hz → %d Hz (soxr VHQ)", sr, FS)
        wav_raw = soxr.resample(wav_raw, sr, FS, quality="VHQ").astype(np.float32)
    if resampled_path is not None:
        resampled_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(resampled_path), wav_raw, FS)
        log.info("Saved resampled input: %s", resampled_path)

    with open(spans_path) as fh:
        spans_data = json.load(fh)
    labeled_raw = parse_labeled_spans(spans_data)
    phone_intervals_raw: Optional[List[PhoneInterval]] = None
    if full_phone_conditioning:
        phone_intervals_raw = parse_phone_intervals(spans_data)

    labeled: List[Tuple[float, float, str]] = []
    follows_t_flags: List[bool] = []
    for start, end, span_phone, follows_t in labeled_raw:
        target = span_phone or phone
        if target is None:
            raise ValueError(
                "Span missing phone label; set --phone or add \"phone\" to each span"
            )
        labeled.append((start, end, target))
        follows_t_flags.append(follows_t)

    clip_end = min(len(wav_raw) / FS, max_utt_sec) if max_utt_sec != float("inf") else len(wav_raw) / FS
    if ext_sec > 0:
        extended: List[Tuple[float, float, str]] = []
        n_applied = 0
        n_skipped = 0
        for (start, end, p), follows_t in zip(labeled, follows_t_flags):
            use_ext = ext_sec if (force_ext or not follows_t) else 0.0
            if use_ext > 0:
                start, end = apply_ext([(start, end)], ext_sec=use_ext, max_end=clip_end)[0]
                n_applied += 1
            else:
                n_skipped += 1
            extended.append((start, end, p))
        labeled = extended
        log.info(
            "Applied --ext %.3fs to %d span(s)%s",
            ext_sec, n_applied,
            f" ({n_skipped} skipped: follow /t/; use --force_ext to override)"
            if n_skipped else
            (" (--force_ext)" if force_ext else ""),
        )

    phones_used = sorted({p for _, _, p in labeled})
    log.info(
        "Inpainting %d span(s) for phone(s): %s%s",
        len(labeled), phones_used,
        "  [full_phone_conditioning]" if full_phone_conditioning else "",
    )

    utt_end = min(len(wav_raw) / FS, max_utt_sec) if max_utt_sec != float("inf") else len(wav_raw) / FS
    labeled_clipped = [(s, e, p) for s, e, p in labeled if s < utt_end]
    wav_padded, shifted, pad_pre_samp, pad_post_samp = apply_edge_padding(
        wav_raw, [(s, e) for s, e, _ in labeled_clipped], max_utt_sec=max_utt_sec
    )
    pad_pre_sec = pad_pre_samp / FS
    labeled_shifted = [
        (shifted[i][0], shifted[i][1], labeled_clipped[i][2])
        for i in range(len(shifted))
    ]
    spans = [(s, e) for s, e, _ in labeled_shifted]
    phone_intervals_shifted: Optional[List[PhoneInterval]] = None
    if phone_intervals_raw is not None:
        phone_intervals_shifted = [
            (s + pad_pre_sec, e + pad_pre_sec, lab)
            for s, e, lab in phone_intervals_raw
            if s < utt_end
        ]
    log.info(
        "Padded wav %.2fs (pad_pre=%.3fs, pad_post=%.3fs)",
        len(wav_padded) / FS,
        pad_pre_samp / FS,
        pad_post_samp / FS,
    )

    ze_len = expected_ze_len(len(wav_padded))
    if windowed_attention:
        encode_regions = merge_context_frame_windows(spans, ze_len, context_frames)
        z_e_t = extract_ze_windowed_t(
            ec_model, wav_padded, device_t, encode_regions, ze_len,
        ).unsqueeze(0)  # [1, T, D]
    else:
        z_e_t = extract_ze_t(ec_model, wav_padded, device_t).unsqueeze(0)  # [1, T, D]
        ze_len = z_e_t.shape[1]
    log.info("z_e: [%d, %d]  (%.2f s @ %.0f fps)", ze_len, z_e_t.shape[2], ze_len / ENCODEC_FPS, ENCODEC_FPS)

    if full_phone_conditioning and not ckpt_full_phone:
        log.warning(
            "Checkpoint model_config.full_phone_conditioning is false; "
            "proceeding with --full_phone_conditioning anyway"
        )
    if ckpt_full_phone and not full_phone_conditioning:
        log.warning(
            "Checkpoint was trained with full_phone_conditioning but the flag "
            "is not set; using single-token conditioning"
        )

    phone_ids_t = phone_starts_t = phone_ends_t = phone_nchars_t = None
    if full_phone_conditioning:
        assert phone_intervals_shifted is not None
        phone_ids_t, phone_starts_t, phone_ends_t, phone_nchars_t = build_phone_seq_tensors(
            phone_intervals_shifted, ze_len, phone2id,
        )
        log.info(
            "Full-phone conditioning: %d phone(s), %d char token(s)",
            phone_starts_t.numel(), phone_ids_t.numel(),
        )

    if windowed_attention:
        log.info(
            "Windowed attention: %d-frame context (±%d frames, ~±%.2f s @ %.0f fps)",
            context_frames, context_frames // 2,
            (context_frames // 2) / ENCODEC_FPS, ENCODEC_FPS,
        )
        z_e_t = inpaint_windowed_spans(
            model, z_e_t, labeled_shifted,
            ze_len=ze_len,
            phone2id=phone2id,
            device=device_t,
            context_frames=context_frames,
            infer_batch_size=infer_batch_size,
            full_phone=full_phone_conditioning,
            phone_ids=phone_ids_t,
            phone_starts=phone_starts_t,
            phone_ends=phone_ends_t,
            phone_nchars=phone_nchars_t,
        )
    elif full_phone_conditioning:
        # One forward for all spans — conditioning already encodes every target phone.
        mask_np = spans_to_frame_mask(spans, ze_len)
        log.info(
            "Full-phone inpaint: %d span(s) (%.1f%% of z_e frames)",
            len(spans), 100.0 * mask_np.mean(),
        )
        masked_position = torch.zeros(1, ze_len, dtype=torch.bool)
        masked_position[0, mask_np] = True
        text_pad, text_lengths, sp_seg, tx_seg = _full_phone_batch_tensors(
            phone_ids_t, phone_starts_t, phone_ends_t, phone_nchars_t, ze_len, device_t,
        )
        z_e_t = inpaint_ze(
            model, z_e_t, masked_position, device_t,
            text_pad=text_pad,
            text_lengths=text_lengths,
            speech_segment_pos=sp_seg,
            text_segment_pos=tx_seg,
        )
    else:
        by_phone: Dict[str, List[Tuple[float, float]]] = {}
        for start, end, p in labeled_shifted:
            by_phone.setdefault(p, []).append((start, end))

        for phone_label, phone_spans in by_phone.items():
            mask_np = spans_to_frame_mask(phone_spans, ze_len)
            log.info(
                "Inpainting %d span(s) for %r (%.1f%% of z_e frames)",
                len(phone_spans), phone_label, 100.0 * mask_np.mean(),
            )
            masked_position = torch.zeros(1, ze_len, dtype=torch.bool)
            masked_position[0, mask_np] = True
            cond_token_id = cond_token_id_for_phone(phone_label, phone2id)
            z_e_t = inpaint_ze(
                model, z_e_t, masked_position, device_t, cond_token_id=cond_token_id,
            )

    if full_decode:
        log.info("Full decode: entire z_e sequence in one pass…")
        out_wav = decode_ze_crop_t(ec_model, z_e_t[0]).detach().cpu().numpy()
        target_len = len(wav_padded)
        if len(out_wav) > target_len:
            out_wav = out_wav[:target_len]
        elif len(out_wav) < target_len:
            out_wav = np.pad(out_wav, (0, target_len - len(out_wav)))
        out_wav = out_wav.astype(np.float32)
    else:
        decode_ctx_frames = max(8, ola_ctx_frames)
        decode_regions = build_decode_regions(spans, ze_len, decode_ctx_frames)
        crossfade_samples = ola_ctx_frames * HOP_SAMP if ola else 0
        log.info(
            "Partial decode: %d region(s), ctx=%d frames%s",
            len(decode_regions),
            decode_ctx_frames,
            f", OLA crossfade={crossfade_samples} samples" if ola else "",
        )

        out_wav = wav_padded.astype(np.float64, copy=True)
        stitch_decoded_patches(
            out_wav,
            z_e_t,
            decode_regions,
            ec_model,
            crossfade_samples=crossfade_samples,
        )
        out_wav = out_wav.astype(np.float32)

    end_idx = len(out_wav) - pad_post_samp if pad_post_samp > 0 else len(out_wav)
    out_wav = out_wav[pad_pre_samp:end_idx]

    if normalize_vol is not None:
        out_wav = normalize_inpaint_volume(out_wav, spans, pad_pre_samp, normalize_vol)

    peak = float(np.max(np.abs(out_wav))) if len(out_wav) else 0.0
    if peak > 1.0:
        out_wav = out_wav / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), out_wav, FS)
    log.info("Wrote inpainted wav: %s", output_path)
    return output_path


def run_inference(
    *,
    phone: Optional[str],
    wav_path: Path,
    spans_path: Path,
    checkpoint: Path,
    output_path: Path,
    resampled_path: Optional[Path] = None,
    device: Optional[str] = None,
    max_utt_sec: float = float("inf"),
    ola: bool = False,
    ola_chunk_frames: int = 32,
    ola_hop_factor: int = 4,
    ola_ctx_frames: int = 8,
    normalize_vol: Optional[float] = None,
    ext_sec: float = 0.0,
    force_ext: bool = True,
    encodec_ckpt: Optional[Path] = None,
    windowed_attention: bool = False,
    context_frames: int = CONTEXT_FRAMES,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    full_phone_conditioning: bool = False,
    full_decode: bool = False,
) -> Path:
    """Single-job inpaint: load models, run one job. Behavior-compatible with
    the pre-refactor ``run_inference`` (same signature and return value)."""
    if full_decode and windowed_attention:
        raise ValueError("--full-decode and --windowed_attention cannot be combined")
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device: %s  encodec: VoiceCraft 16k (%d Hz, %.0f fps)",
             device_t, FS, ENCODEC_FPS)
    ec_model, model, phone2id, ckpt_full_phone = _load_cond_models(
        checkpoint, device_t, encodec_ckpt,
    )
    return _infer_one_cond(
        ec_model, model, phone2id, ckpt_full_phone,
        phone=phone,
        wav_path=wav_path,
        spans_path=spans_path,
        output_path=output_path,
        device_t=device_t,
        resampled_path=resampled_path,
        max_utt_sec=max_utt_sec,
        ola=ola,
        ola_chunk_frames=ola_chunk_frames,
        ola_hop_factor=ola_hop_factor,
        ola_ctx_frames=ola_ctx_frames,
        normalize_vol=normalize_vol,
        ext_sec=ext_sec,
        force_ext=force_ext,
        windowed_attention=windowed_attention,
        context_frames=context_frames,
        infer_batch_size=infer_batch_size,
        full_phone_conditioning=full_phone_conditioning,
        full_decode=full_decode,
    )


def run_batch_inference(
    *,
    jobs: Sequence[dict],
    checkpoint: Path,
    phone: Optional[str] = None,
    device: Optional[str] = None,
    max_utt_sec: float = float("inf"),
    ola: bool = False,
    ola_chunk_frames: int = 32,
    ola_hop_factor: int = 4,
    ola_ctx_frames: int = 8,
    normalize_vol: Optional[float] = None,
    ext_sec: float = 0.0,
    force_ext: bool = True,
    encodec_ckpt: Optional[Path] = None,
    windowed_attention: bool = False,
    context_frames: int = CONTEXT_FRAMES,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    full_phone_conditioning: bool = False,
    full_decode: bool = False,
) -> List[Path]:
    """Inpaint many (wav, spans, output) jobs, loading EnCodec + the phone-EC
    checkpoint exactly once and reusing them across every job.

    ``jobs`` is a sequence of dicts with keys ``wav``, ``spans``, ``output``
    (required) and optionally ``phone`` / ``save_resampled`` (per-job
    overrides of the like-named top-level argument). All other settings
    (checkpoint, device, ola*, normalize_vol, ext/force_ext, windowed
    attention, full_phone_conditioning, full_decode, ...) are shared across
    the whole batch — matching how run_inpaint_pipeline.py invokes this per
    source file today, just without reloading models each time.
    """
    if full_decode and windowed_attention:
        raise ValueError("--full-decode and --windowed_attention cannot be combined")
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device: %s  encodec: VoiceCraft 16k (%d Hz, %.0f fps)",
             device_t, FS, ENCODEC_FPS)
    ec_model, model, phone2id, ckpt_full_phone = _load_cond_models(
        checkpoint, device_t, encodec_ckpt,
    )
    log.info("Batch inpaint: %d job(s); models loaded once", len(jobs))

    outputs: List[Path] = []
    for i, job in enumerate(jobs):
        wav_path = Path(job["wav"])
        spans_path = Path(job["spans"])
        output_path = Path(job["output"])
        resampled = job.get("save_resampled")
        log.info(
            "[batch %d/%d] %s + %s -> %s",
            i + 1, len(jobs), wav_path.name, spans_path.name, output_path,
        )
        out = _infer_one_cond(
            ec_model, model, phone2id, ckpt_full_phone,
            phone=job.get("phone", phone),
            wav_path=wav_path,
            spans_path=spans_path,
            output_path=output_path,
            device_t=device_t,
            resampled_path=Path(resampled) if resampled else None,
            max_utt_sec=max_utt_sec,
            ola=ola,
            ola_chunk_frames=ola_chunk_frames,
            ola_hop_factor=ola_hop_factor,
            ola_ctx_frames=ola_ctx_frames,
            normalize_vol=normalize_vol,
            ext_sec=ext_sec,
            force_ext=force_ext,
            windowed_attention=windowed_attention,
            context_frames=context_frames,
            infer_batch_size=infer_batch_size,
            full_phone_conditioning=full_phone_conditioning,
            full_decode=full_decode,
        )
        outputs.append(out)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phone", default=None,
                   help="Default phone label when spans omit \"phone\" "
                        "(must be in checkpoint token_list, or <any> is used)")
    p.add_argument("--wav", type=Path, default=None,
                   help="Input wav file (ignored / not required with --batch_manifest)")
    p.add_argument("--spans", type=Path, default=None,
                   help='JSON with a top-level "spans" list; with '
                        '--full_phone_conditioning also needs "phone_intervals" '
                        '(ignored / not required with --batch_manifest)')
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Checkpoint (.pt) from train_phone_ec_cond.py")
    p.add_argument("--output", type=Path, default=None,
                   help="Output wav path (16 kHz) (ignored / not required with --batch_manifest)")
    p.add_argument(
        "--batch_manifest", type=Path, default=None,
        help='JSON with a top-level "jobs" list of {"wav", "spans", "output"'
             '[, "phone", "save_resampled"]} dicts. When set, EnCodec + the '
             "phone-EC checkpoint are loaded once and reused for every job "
             "(much faster than one process per file); --wav/--spans/--output "
             "are ignored.",
    )
    p.add_argument(
        "--encodec-ckpt", dest="encodec_ckpt", type=Path, default=None,
        help="Path to VoiceCraft EnCodec checkpoint (.th). "
             "Defaults to the canonical VoiceCraft location.",
    )
    p.add_argument(
        "--save-resampled", dest="save_resampled", type=Path, default=None,
        help="Optional: save the 16 kHz resampled input wav here",
    )
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--max_utt_sec", type=float, default=float("inf"),
                   help="Truncate input to this many seconds (default: no limit)")

    ola = p.add_argument_group("Patch crossfade (--ola)")
    ola.add_argument(
        "--ola", action="store_true",
        help="Crossfade decoded patches into the original wav at span boundaries "
             "(uses --ola_ctx_frames for fade length)",
    )
    ola.add_argument(
        "--ola_chunk_frames", type=int, default=32,
        help=argparse.SUPPRESS,
    )
    ola.add_argument(
        "--ola_hop_factor", type=int, default=4, choices=[2, 4],
        help=argparse.SUPPRESS,
    )
    ola.add_argument("--ola_ctx_frames", type=int, default=8,
                     help="Decoder context / crossfade length in EnCodec frames (default: 8)")

    p.add_argument("--normalize_vol", type=float, metavar="SCALE", default=None,
                   help="Match inpainted RMS to surrounding audio, then scale by SCALE")
    p.add_argument("--ext", type=float, default=0.0, metavar="SEC",
                   help="Extend each inpainted span by SEC on both sides (default: 0). "
                        "Spans marked follows_t (immediately after /t/) also get --ext "
                        "by default; pass --no-force_ext to skip those.")
    p.add_argument(
        "--force_ext", action=argparse.BooleanOptionalAction, default=True,
        help="Apply --ext even to spans marked follows_t (immediately after /t/) "
             "(default: on).",
    )
    p.add_argument(
        "--full_phone_conditioning", action="store_true",
        help="Condition on char-tokenized full utterance phone strings with SEGA "
             "alignment (matches train_phone_ec_cond.py --full_phone_conditioning). "
             "Requires \"phone_intervals\" in the spans JSON "
             "(remapped target labels for edited phones).",
    )

    win = p.add_argument_group(
        "Windowed attention / encode "
        "(matches train_phone_ec_cond --windowed_attention)"
    )
    win.add_argument(
        "--windowed_attention", action="store_true",
        help="Encode and run each forward pass on a context window around the "
             "masked span only (default: off; full-file encode + full-length MLM). "
             "Incompatible with --full-decode.",
    )
    win.add_argument(
        "--context_frames", type=int, default=CONTEXT_FRAMES,
        help=f"Context window size when --windowed_attention is set "
             f"(default: {CONTEXT_FRAMES} = ±{CONTEXT_FRAMES // 2} frames "
             f"≈ ±{CONTEXT_FRAMES / 2 / ENCODEC_FPS:.2f}s @ {ENCODEC_FPS:.0f} fps)",
    )
    win.add_argument(
        "--infer_batch_size", type=int, default=DEFAULT_INFER_BATCH_SIZE,
        help=f"Batch size for windowed span forwards (default: {DEFAULT_INFER_BATCH_SIZE}). "
             f"Spans in a batch are independent; use 1 to disable batching.",
    )
    p.add_argument(
        "--full-decode", "--full_decode", dest="full_decode", action="store_true",
        help="Decode the entire z_e sequence in one EnCodec decoder pass "
             "(default: off; partial decode of span regions only). "
             "Incompatible with --windowed_attention.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.full_decode and args.windowed_attention:
        raise SystemExit("--full-decode and --windowed_attention cannot be combined")
    if args.batch_manifest is not None:
        if args.wav is not None or args.spans is not None or args.output is not None:
            raise SystemExit(
                "--batch_manifest cannot be combined with --wav/--spans/--output"
            )
        manifest = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
        jobs = manifest["jobs"] if isinstance(manifest, dict) else manifest
        if not jobs:
            raise SystemExit(f"--batch_manifest {args.batch_manifest} has no jobs")
        run_batch_inference(
            jobs=jobs,
            checkpoint=args.checkpoint,
            phone=args.phone,
            device=args.device,
            max_utt_sec=args.max_utt_sec,
            ola=args.ola,
            ola_chunk_frames=args.ola_chunk_frames,
            ola_hop_factor=args.ola_hop_factor,
            ola_ctx_frames=args.ola_ctx_frames,
            normalize_vol=args.normalize_vol,
            ext_sec=args.ext,
            force_ext=args.force_ext,
            encodec_ckpt=args.encodec_ckpt,
            windowed_attention=args.windowed_attention,
            context_frames=args.context_frames,
            infer_batch_size=args.infer_batch_size,
            full_phone_conditioning=args.full_phone_conditioning,
            full_decode=args.full_decode,
        )
        return

    if args.wav is None or args.spans is None or args.output is None:
        raise SystemExit(
            "--wav, --spans and --output are required unless --batch_manifest is given"
        )
    run_inference(
        phone=args.phone,
        wav_path=args.wav,
        spans_path=args.spans,
        checkpoint=args.checkpoint,
        output_path=args.output,
        resampled_path=args.save_resampled,
        device=args.device,
        max_utt_sec=args.max_utt_sec,
        ola=args.ola,
        ola_chunk_frames=args.ola_chunk_frames,
        ola_hop_factor=args.ola_hop_factor,
        ola_ctx_frames=args.ola_ctx_frames,
        normalize_vol=args.normalize_vol,
        ext_sec=args.ext,
        force_ext=args.force_ext,
        encodec_ckpt=args.encodec_ckpt,
        windowed_attention=args.windowed_attention,
        context_frames=args.context_frames,
        infer_batch_size=args.infer_batch_size,
        full_phone_conditioning=args.full_phone_conditioning,
        full_decode=args.full_decode,
    )


if __name__ == "__main__":
    main()
