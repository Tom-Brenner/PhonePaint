#!/usr/bin/env python3
"""
Train a SINGLE phone-conditioned A3T masked acoustic model in EnCodec latent
space.  Successor to train_phone_ec.py (which is kept untouched).

What is different from train_phone_ec.py
-----------------------------------------
1. One model for many phones.  Instead of one checkpoint per phone, a target
   phone id is fed through the encoder's text/conditioning path (the same path
   train_phone_ec.py disables with the dummy ``-2`` token).  Training on the
   union of all phone instances multiplies the effective dataset and shares
   low-level acoustic representations -> far less overfitting.

2. Two-stage training:
     --stage pretrain : self-supervised RANDOM-span masking over ALL records
                        (phone labels not required); conditioned with the
                        generic <any> token.  Produces a regularised init.
     --stage finetune : phone-span masking; each utterance is conditioned on
                        one of its present target phones.  Usually started from
                        a pretrain checkpoint via --init.

3. Regularisation: AdamW (decoupled weight decay), configurable dropout, and
   latent-space SpecAugment (time + feature masking) applied to the *input*
   only -- the L1 loss is always computed against the CLEAN target latents, so
   augmentation never corrupts the regression target.

The script is codec-agnostic: it reads pre-computed EnCodec ``z_e`` latents and
``{phone}_mask`` arrays from the NPZ manifest produced by prep_encodec.py, so it
works with either the 24 kHz Meta EnCodec or the 16 kHz VoiceCraft EnCodec once
prep_encodec.py is switched over.

Examples
--------
    # Stage 1: self-supervised pretraining (default: all target phones)
    python train_phone_ec_cond.py --stage pretrain --outdir exp/cond_pretrain

    # Stage 2: phone-conditioned fine-tuning from pretrain checkpoint
    python train_phone_ec_cond.py --stage finetune \\
        --init exp/cond_pretrain/best.pt --outdir exp/cond_phone_ec

    # Finetune with extra phones: --init copies acoustic weights and matches
    # encoder.text_embed rows by token name; new phone rows are initialised from
    # the checkpoint <any> vector (no new output head is required).

Recommended epoch counts: ~50 for pretrain (watch val loss plateau), ~300 for
finetune.  Override with ``--epochs``.
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parent))

from espnet.nets.pytorch_backend.conformer.encoder import MLMDecoder, MLMEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
from espnet2.tts.sedit.sedit_model import ESPnetMLMEncAsDecoderModel
from phone_labels import (
    ALIGN_LABEL_ALIASES,
    DEFAULT_TARGET_PHONES,
    build_alignment_char_vocab,
    mask_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature config — must match prep_encodec.py
# ─────────────────────────────────────────────────────────────────────────────
ENCODEC_DIM = 128

# ─────────────────────────────────────────────────────────────────────────────
# Default model config (overridable on the CLI)
# ─────────────────────────────────────────────────────────────────────────────
ATTN_DIM       = 192
ATTN_HEADS     = 2
FFN_UNITS      = 768
ENC_BLOCKS     = 2
DEC_BLOCKS     = 2
ENC_CNN_KERNEL = 7
DEC_CNN_KERNEL = 31
DROPOUT        = 0.25
POSTNET_LAYERS = 3
POSTNET_CHANS  = 128
POSTNET_FILTS  = 5
CONTEXT_FRAMES = 101   # ±50 frames (~±0.67 s at 75 Hz) around the masked span; 0 = disabled

# ─────────────────────────────────────────────────────────────────────────────
# Default training config
# ─────────────────────────────────────────────────────────────────────────────
WARMUP_STEPS  = 4000
MAX_EPOCHS    = 300
PRETRAIN_EPOCHS = 50   # recommended SSL stage; finetune uses MAX_EPOCHS
BATCH_SIZE    = 8
NUM_WORKERS   = 4
BUCKET_WIDTH  = 64   # trim-length buckets when --windowed_attention; 0 = disable
GRAD_CLIP     = 1.0
LOG_EVERY     = 50
SAVE_EVERY    = 10
WEIGHT_DECAY  = 1e-2
GPU_CLEAR_EVERY_STEPS = 100

# Self-supervised (pretrain) masking defaults
PRETRAIN_MLM_PROB  = 0.15
PRETRAIN_MEAN_SPAN = 6

# Conditioning token reserved for "no specific phone" (SSL pretraining).
ANY_TOKEN = "<any>"


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(phones: List[str]):
    """token_list[0] = <any>; remaining entries are the target phones in order."""
    token_list = [ANY_TOKEN] + list(phones)
    phone2id = {ph: i for i, ph in enumerate(token_list)}
    return token_list, phone2id


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiPhoneNpzDataset(Dataset):
    """
    Loads NPZ latents in workers and applies per-utterance masking / SpecAugment.

    Returns (z_clean, z_in, masked_pos, cond_id) for a light main-process collate.
    """

    def __init__(
        self,
        records: List[dict],
        phones: List[str],
        stage: str,
        phone2id: Dict[str, int],
        *,
        mlm_prob: float = PRETRAIN_MLM_PROB,
        mean_span: int = PRETRAIN_MEAN_SPAN,
        sa_time_bands: int = 0,
        sa_time_width: int = 0,
        sa_freq_bands: int = 0,
        sa_freq_width: int = 0,
        context_frames: int = CONTEXT_FRAMES,
        full_phone: bool = False,
        base_seed: int = 0,
    ):
        self.records = records
        self.phones = phones
        self.mask_keys = {ph: mask_key(ph) for ph in phones}
        self.stage = stage
        self.phone2id = phone2id
        self.full_phone = full_phone
        self.mlm_prob = mlm_prob
        self.mean_span = mean_span
        self.sa_time_bands = sa_time_bands
        self.sa_time_width = sa_time_width
        self.sa_freq_bands = sa_freq_bands
        self.sa_freq_width = sa_freq_width
        self.context_frames = context_frames
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _worker_rng(self, idx: int) -> np.random.Generator:
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        seed = self.base_seed + self.epoch * 1_000_003 + worker_id * 10_007 + idx
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rng = self._worker_rng(idx)
        with np.load(rec["npz"]) as data:
            z_e = data["z_e"].astype(np.float32)
            t_len = z_e.shape[0]
            avail: Dict[str, np.ndarray] = {}
            for ph, key in self.mask_keys.items():
                if key in data:
                    m = data[key].astype(bool)
                    if m.any():
                        avail[ph] = m

            use_full_phone = (
                self.full_phone
                and self.stage == "finetune"
                and {"phone_ids", "phone_starts", "phone_ends", "phone_nchars"} <= set(data.files)
            )
            if use_full_phone:
                phone_ids = torch.from_numpy(data["phone_ids"].astype(np.int64))
                phone_starts = torch.from_numpy(data["phone_starts"].astype(np.int64))
                phone_ends = torch.from_numpy(data["phone_ends"].astype(np.int64))
                phone_nchars = torch.from_numpy(data["phone_nchars"].astype(np.int64))

        if self.stage == "pretrain":
            m = random_span_mask(t_len, self.mlm_prob, self.mean_span, rng)
            masked_pos = torch.from_numpy(m)
            cond_id = 0
            use_full_phone = False
        else:
            phones = list(avail.keys())
            if not phones:
                raise RuntimeError(f"no phone masks in {rec['npz']}")
            phone = phones[int(rng.integers(0, len(phones)))]
            m = avail[phone]
            m_len = min(t_len, len(m))
            masked_pos = torch.zeros(t_len, dtype=torch.bool)
            masked_pos[:m_len] = torch.from_numpy(m[:m_len])
            cond_id = self.phone2id.get(phone, 0)

        z_clean = torch.from_numpy(z_e)
        z_in = z_clean.clone()

        if self.context_frames > 0 and masked_pos.any():
            half = self.context_frames // 2
            nz = masked_pos.nonzero(as_tuple=False).squeeze(-1)
            span_start = int(nz[0].item())
            span_end   = int(nz[-1].item()) + 1          # exclusive
            ctx_start  = max(0, span_start - half)
            ctx_end    = min(t_len, span_end + half)
            z_clean    = z_clean[ctx_start:ctx_end]
            z_in       = z_in[ctx_start:ctx_end]
            masked_pos = masked_pos[ctx_start:ctx_end]
            if use_full_phone:
                phone_ids, phone_starts, phone_ends, phone_nchars = trim_phone_alignment(
                    phone_ids, phone_starts, phone_ends, phone_nchars,
                    ctx_start, ctx_end,
                )

        if self.sa_time_bands or self.sa_freq_bands:
            specaugment_(
                z_in, self.sa_time_bands, self.sa_time_width,
                self.sa_freq_bands, self.sa_freq_width, rng,
            )
        if use_full_phone:
            return z_clean, z_in, masked_pos, phone_ids, phone_starts, phone_ends, phone_nchars
        return z_clean, z_in, masked_pos, cond_id


# ─────────────────────────────────────────────────────────────────────────────
# Masking + augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_span_mask(length: int, mlm_prob: float, mean_span: int,
                     rng: np.random.Generator) -> np.ndarray:
    """SpanBERT-style random masking used for self-supervised pretraining."""
    mask = np.zeros(length, dtype=bool)
    if length == 0:
        return mask
    target = int(round(mlm_prob * length))
    if target <= 0:
        return mask
    guard = 0
    while mask.sum() < target and guard < 10 * length:
        guard += 1
        span = max(1, int(rng.poisson(mean_span)))
        start = int(rng.integers(0, length))
        end = min(length, start + span)
        mask[start:end] = True
    return mask


def specaugment_(z: torch.Tensor, time_bands: int, time_width: int,
                 freq_bands: int, freq_width: int,
                 rng: np.random.Generator) -> None:
    """In-place latent SpecAugment on a single [T, D] input tensor."""
    T, D = z.shape
    for _ in range(time_bands):
        if T <= 1 or time_width <= 0:
            break
        w = int(rng.integers(0, min(time_width, T) + 1))
        if w <= 0:
            continue
        t0 = int(rng.integers(0, T - w + 1))
        z[t0:t0 + w, :] = 0.0
    for _ in range(freq_bands):
        if D <= 1 or freq_width <= 0:
            break
        w = int(rng.integers(0, min(freq_width, D) + 1))
        if w <= 0:
            continue
        f0 = int(rng.integers(0, D - w + 1))
        z[:, f0:f0 + w] = 0.0


def trim_phone_alignment(
    phone_ids: torch.Tensor,
    phone_starts: torch.Tensor,
    phone_ends: torch.Tensor,
    phone_nchars: torch.Tensor,
    ctx_start: int,
    ctx_end: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip phone alignment arrays to a trimmed speech window."""
    ctx_len = ctx_end - ctx_start
    ids_out: List[int] = []
    starts_out: List[int] = []
    ends_out: List[int] = []
    nchars_out: List[int] = []
    char_off = 0
    for i in range(phone_starts.numel()):
        ns = int(phone_nchars[i].item())
        s = int(phone_starts[i].item())
        e = int(phone_ends[i].item())
        if e <= ctx_start or s >= ctx_end:
            char_off += ns
            continue
        ids_out.extend(phone_ids[char_off:char_off + ns].tolist())
        starts_out.append(max(0, s - ctx_start))
        ends_out.append(min(ctx_len, e - ctx_start))
        nchars_out.append(ns)
        char_off += ns
    if not ids_out:
        return (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([max(1, ctx_len)], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )
    return (
        torch.tensor(ids_out, dtype=torch.long),
        torch.tensor(starts_out, dtype=torch.long),
        torch.tensor(ends_out, dtype=torch.long),
        torch.tensor(nchars_out, dtype=torch.long),
    )


def build_segment_pos_from_alignment(
    speech_len: int,
    phone_starts: torch.Tensor,
    phone_ends: torch.Tensor,
    phone_nchars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SEGA segment ids for char-token phone strings (one segment per phoneme)."""
    text_len = int(phone_nchars.sum().item())
    speech_seg = torch.zeros(speech_len, dtype=torch.long)
    text_seg = torch.zeros(text_len, dtype=torch.long)
    char_off = 0
    for j in range(phone_starts.numel()):
        ns = int(phone_nchars[j].item())
        s = int(phone_starts[j].item())
        e = int(phone_ends[j].item())
        seg = j + 1
        if e > s:
            speech_seg[s:e] = seg
        if ns > 0:
            text_seg[char_off:char_off + ns] = seg
        char_off += ns
    return speech_seg, text_seg


# ─────────────────────────────────────────────────────────────────────────────
# Collate + length bucketing
# ─────────────────────────────────────────────────────────────────────────────

def collate_phone_batch(batch, full_phone: bool = False) -> dict:
    """Pad worker-prepared samples into the ESPnet model batch dict."""
    z_clean_list = [x[0] for x in batch]
    z_in_list = [x[1] for x in batch]
    masked_list = [x[2] for x in batch]

    lengths = torch.tensor([z.size(0) for z in z_clean_list], dtype=torch.long)
    z_clean = pad_list(z_clean_list, 0.0)
    z_in = pad_list(z_in_list, 0.0)
    masked_position = pad_list(masked_list, False)
    bsz, max_len, _ = z_clean.shape

    speech_mask = make_non_pad_mask(
        lengths.tolist(), z_clean[:, :, 0], length_dim=1,
    ).unsqueeze(-2)
    masked_position = masked_position & speech_mask.squeeze(1)

    if full_phone:
        text_list = [x[3] for x in batch]
        starts_list = [x[4] for x in batch]
        ends_list = [x[5] for x in batch]
        nchars_list = [x[6] for x in batch]
        text_lengths = torch.tensor([t.numel() for t in text_list], dtype=torch.long)
        text = pad_list(text_list, 0)
        max_tlen = int(text_lengths.max().item()) if text_lengths.numel() else 1
        text_mask = make_non_pad_mask(
            text_lengths.tolist(), text, length_dim=1,
        ).unsqueeze(-2)
        speech_segment_pos = torch.zeros(bsz, max_len, dtype=torch.long)
        text_segment_pos = torch.zeros(bsz, max_tlen, dtype=torch.long)
        for i, slen in enumerate(lengths.tolist()):
            sp_seg, tx_seg = build_segment_pos_from_alignment(
                slen, starts_list[i], ends_list[i], nchars_list[i],
            )
            speech_segment_pos[i, :slen] = sp_seg
            text_segment_pos[i, :text_lengths[i]] = tx_seg
    else:
        cond_ids = torch.tensor([x[3] for x in batch], dtype=torch.long)
        text = cond_ids.unsqueeze(1)
        text_lengths = torch.ones(bsz, dtype=torch.long)
        text_mask = torch.ones(bsz, 1, 1, dtype=torch.bool)
        speech_segment_pos = torch.zeros(bsz, max_len, dtype=torch.long)
        text_segment_pos = torch.ones(bsz, 1, dtype=torch.long)

    return dict(
        speech_in=z_in,
        speech_clean=z_clean,
        speech_lengths=lengths,
        text=text,
        text_lengths=text_lengths,
        masked_position=masked_position,
        speech_mask=speech_mask,
        text_mask=text_mask,
        speech_segment_pos=speech_segment_pos,
        text_segment_pos=text_segment_pos,
    )


class BucketBatchSampler(Sampler):
    """Group samples of similar trimmed length to reduce padding waste."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        bucket_width: int = BUCKET_WIDTH,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(self.lengths):
            bucket_id = length // self.bucket_width
            buckets.setdefault(bucket_id, []).append(idx)

        batches: List[List[int]] = []
        bucket_ids = sorted(buckets)
        if self.shuffle:
            rng.shuffle(bucket_ids)
        for bucket_id in bucket_ids:
            indices = buckets[bucket_id]
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def _split_records(records: List[dict], val_frac: float = 0.05, seed: int = 42
                   ) -> tuple[List[dict], List[dict]]:
    n_val = max(1, int(val_frac * len(records)))
    n_tr = len(records) - n_val
    perm = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val].tolist()
    tr_idx = perm[n_val:].tolist()
    return [records[i] for i in tr_idx], [records[i] for i in val_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(token_list: List[str], args, odim: int = ENCODEC_DIM
                ) -> ESPnetMLMEncAsDecoderModel:
    vocab_size = len(token_list)
    encoder = MLMEncoder(
        idim                          = odim,
        vocab_size                    = vocab_size,
        pre_speech_layer              = 0,
        attention_dim                 = args.attn_dim,
        attention_heads               = args.attn_heads,
        linear_units                  = args.ffn_units,
        num_blocks                    = args.enc_blocks,
        dropout_rate                  = args.dropout,
        positional_dropout_rate       = args.dropout,
        attention_dropout_rate        = args.dropout,
        input_layer                   = "sega_mlm",
        normalize_before              = True,
        macaron_style                 = True,
        pos_enc_layer_type            = "rel_pos",
        selfattention_layer_type      = "rel_selfattn",
        activation_type               = "swish",
        use_cnn_module                = True,
        cnn_module_kernel             = ENC_CNN_KERNEL,
        positionwise_layer_type       = "conv1d",
        positionwise_conv_kernel_size = 3,
    )
    decoder = MLMDecoder(
        idim                          = args.attn_dim,
        vocab_size                    = vocab_size,
        pre_speech_layer              = 0,
        attention_dim                 = args.attn_dim,
        attention_heads               = args.attn_heads,
        linear_units                  = args.ffn_units,
        num_blocks                    = args.dec_blocks,
        dropout_rate                  = args.dropout,
        positional_dropout_rate       = args.dropout,
        attention_dropout_rate        = args.dropout,
        input_layer                   = None,
        normalize_before              = True,
        macaron_style                 = True,
        pos_enc_layer_type            = "rel_pos",
        selfattention_layer_type      = "rel_selfattn",
        activation_type               = "swish",
        use_cnn_module                = True,
        cnn_module_kernel             = DEC_CNN_KERNEL,
        positionwise_layer_type       = "conv1d",
        positionwise_conv_kernel_size = 3,
    )
    model = ESPnetMLMEncAsDecoderModel(
        token_list     = token_list,
        odim           = odim,
        feats_extract  = None,
        normalize      = None,
        encoder        = encoder,
        decoder        = decoder,
        postnet_layers = args.postnet_layers,
        postnet_chans  = args.postnet_chans,
        postnet_filts  = args.postnet_filts,
        lsm_weight     = 0.1,
        masking_schema = "phn_span",
        mean_phn_span  = 0,
        mlm_prob       = 1.0,
        report_cer     = False,
        report_wer     = False,
    )
    if model.sfc is not None:
        nn.init.xavier_uniform_(model.sfc.weight)
    return model


TEXT_EMBED_KEY = "encoder.text_embed.0.weight"


def _token_list_from_checkpoint(ckpt: dict) -> Optional[List[str]]:
    token_list = ckpt.get("token_list")
    if token_list is not None:
        return list(token_list)
    phones = ckpt.get("phones")
    if phones is not None:
        tl, _ = build_vocab(list(phones))
        return tl
    return None


def _init_vector_for_new_embedding_rows(
    src_weight: torch.Tensor,
    src_token_list: List[str],
    dst_weight: torch.Tensor,
    copied_dst_indices: List[int],
) -> torch.Tensor:
    if ANY_TOKEN in src_token_list:
        return src_weight[src_token_list.index(ANY_TOKEN)].clone()
    if copied_dst_indices:
        return dst_weight[copied_dst_indices].mean(dim=0)
    return src_weight.mean(dim=0)


def copy_phone_embedding_by_token(
    dst_weight: torch.Tensor,
    src_weight: torch.Tensor,
    src_token_list: List[str],
    dst_token_list: List[str],
) -> tuple[torch.Tensor, Dict[str, List[str]]]:
    """Map encoder.text_embed rows by token string; init unseen tokens."""
    out = dst_weight.clone()
    copied: List[str] = []
    new_tokens: List[str] = []
    copied_indices: List[int] = []
    for i, tok in enumerate(dst_token_list):
        if tok in src_token_list:
            j = src_token_list.index(tok)
            out[i] = src_weight[j]
            copied.append(tok)
            copied_indices.append(i)
        else:
            new_tokens.append(tok)
    if new_tokens:
        init_vec = _init_vector_for_new_embedding_rows(
            src_weight, src_token_list, out, copied_indices,
        )
        for tok in new_tokens:
            out[dst_token_list.index(tok)] = init_vec
    return out, {"copied": copied, "new": new_tokens}


def load_pretrained_weights(
    model: nn.Module,
    init_path: Path,
    dst_token_list: List[str],
    map_location,
) -> None:
    """Load a checkpoint with phone-embedding expansion when the vocab grows."""
    ckpt = torch.load(init_path, map_location=map_location)
    src_state = ckpt.get("model", ckpt)
    src_token_list = _token_list_from_checkpoint(ckpt)
    dst_state = model.state_dict()

    merged: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[str] = []
    for key, dst_val in dst_state.items():
        if key not in src_state:
            continue
        src_val = src_state[key]
        if dst_val.shape == src_val.shape:
            merged[key] = src_val
        else:
            shape_mismatch.append(key)

    embed_stats: Optional[Dict[str, List[str]]] = None
    if TEXT_EMBED_KEY in dst_state and TEXT_EMBED_KEY in src_state:
        if src_token_list is None:
            log.warning(
                "Checkpoint %s has no token_list/phones metadata; "
                "encoder.text_embed cannot be matched by phone name",
                init_path,
            )
        else:
            merged[TEXT_EMBED_KEY], embed_stats = copy_phone_embedding_by_token(
                dst_state[TEXT_EMBED_KEY],
                src_state[TEXT_EMBED_KEY],
                src_token_list,
                dst_token_list,
            )
            if TEXT_EMBED_KEY in shape_mismatch:
                shape_mismatch.remove(TEXT_EMBED_KEY)

    load_state = dict(dst_state)
    load_state.update(merged)
    missing, unexpected = model.load_state_dict(load_state, strict=False)

    acoustic_keys = [
        k for k in merged if k.startswith("sfc.") or k.startswith("postnet.")
    ]
    log.info(
        "Initialised from %s: %d tensors transferred (%d acoustic-head), "
        "missing=%d unexpected=%d",
        init_path, len(merged), len(acoustic_keys), len(missing), len(unexpected),
    )
    if embed_stats is not None:
        log.info(
            "Phone embedding: %d shared tokens copied, %d new rows initialised "
            "(%s); latent output head unchanged",
            len(embed_stats["copied"]),
            len(embed_stats["new"]),
            embed_stats["new"] or "none",
        )
    if shape_mismatch:
        log.warning(
            "Skipped %d tensors due to shape mismatch (architecture change?): %s",
            len(shape_mismatch), ", ".join(sorted(shape_mismatch)[:8]),
        )
    ckpt_cfg = ckpt.get("model_config")
    if ckpt_cfg:
        log.info("Source checkpoint model_config: %s", ckpt_cfg)

    del ckpt, src_state, dst_state, load_state, merged


# ─────────────────────────────────────────────────────────────────────────────
# Loss (masked L1 against the clean target; mirrors _calc_mlm_loss)
# ─────────────────────────────────────────────────────────────────────────────

def masked_l1(before_outs, after_outs, target, masked_position):
    mp = masked_position.float()                            # [B, L]
    per_frame = (before_outs - target).abs().sum(dim=-1)    # [B, L]
    if after_outs is not None:
        per_frame = per_frame + (after_outs - target).abs().sum(dim=-1)
    return (per_frame * mp).sum() / (mp.sum() + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (Noam)
# ─────────────────────────────────────────────────────────────────────────────

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size: int, warmup_steps: int, last_epoch=-1):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.model_size ** -0.5 * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uses_cuda(device: Union[torch.device, str]) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def clear_gpu_cache(device: Union[torch.device, str], *, sync: bool = False) -> None:
    """Release free-listed CUDA blocks back to the driver.

    ``sync=False`` (default) is non-blocking and safe to call in the inner
    training loop.  ``sync=True`` adds a full GPU synchronize; only use this
    at the very end of training where stalling doesn't matter.
    """
    if not _uses_cuda(device) or not torch.cuda.is_available():
        return
    if sync:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _shutdown_dataloader(loader) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception:
            pass


def release_training_resources(
    device: Union[torch.device, str],
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    loaders=(),
) -> None:
    """Drop training objects and return cached GPU memory to the driver."""
    for loader in loaders:
        _shutdown_dataloader(loader)

    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass

    # Break reference cycles from the training step before collecting.
    del model, optimizer, scheduler
    gc.collect()
    clear_gpu_cache(device, sync=True)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, device,
              train: bool, epoch: int = 0,
              gpu_clear_every_steps: int = GPU_CLEAR_EVERY_STEPS,
              scaler: Optional["torch.cuda.amp.GradScaler"] = None,
              detach_postnet_grad: bool = False) -> float:
    model.train(train)
    # Accumulate loss on the GPU to avoid a synchronize on every step.
    # .item() is called only at log intervals and once at the epoch end.
    total_loss_gpu = torch.zeros(1, device=device)
    n_batches = 0
    t0 = time.time()
    phase = "train" if train else "val"
    amp_enabled = scaler is not None and scaler.is_enabled()

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model_batch = dict(
                speech_pad         = batch["speech_in"],
                text_pad           = batch["text"],
                masked_position    = batch["masked_position"],
                speech_mask        = batch["speech_mask"],
                text_mask          = batch["text_mask"],
                speech_segment_pos = batch["speech_segment_pos"],
                text_segment_pos   = batch["text_segment_pos"],
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp_enabled):
                before_outs, after_outs, _, _ = model._forward(
                    model_batch, batch["speech_segment_pos"]
                )
                if detach_postnet_grad and after_outs is not None:
                    # Stop the after_outs loss from propagating back through
                    # before_outs into the encoder/decoder.  Since the model
                    # computes after_outs = before_outs + postnet(before_outs),
                    # this expression equals after_outs numerically but the
                    # additive before_outs term is treated as a constant, so
                    # only the postnet weights receive gradient from this term.
                    after_outs = after_outs - before_outs + before_outs.detach()
                loss = masked_l1(
                    before_outs, after_outs, batch["speech_clean"],
                    batch["masked_position"],
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                scheduler.step()

            total_loss_gpu = total_loss_gpu + loss.detach().float()
            n_batches += 1

            del batch, model_batch, before_outs, after_outs, loss
            if gpu_clear_every_steps > 0 and n_batches % gpu_clear_every_steps == 0:
                clear_gpu_cache(device)  # non-blocking: no synchronize

            if n_batches % LOG_EVERY == 0:
                elapsed = time.time() - t0
                secs_per = elapsed / n_batches
                total_b = len(loader)
                eta = secs_per * (total_b - n_batches)
                avg_loss = (total_loss_gpu / n_batches).item()  # one sync per log interval
                log.info(
                    "Epoch %d [%s] step %d/%d  loss=%.4f  %.2fs/batch  ETA %dm%02ds",
                    epoch, phase, n_batches, total_b,
                    avg_loss, secs_per,
                    int(eta) // 60, int(eta) % 60,
                )

    gc.collect()
    clear_gpu_cache(device)
    return (total_loss_gpu / max(n_batches, 1)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Record loading / filtering
# ─────────────────────────────────────────────────────────────────────────────

def load_records(manifest_path: Path) -> List[dict]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rewrite_npz_root(records: List[dict], manifest_path: Path, npz_root: Path) -> List[dict]:
    """Rewrite NPZ paths in records to be relative to npz_root instead of the
    original root embedded in the manifest (detected as manifest_path.parent)."""
    original_root = manifest_path.parent
    out = []
    for rec in records:
        try:
            rel = Path(rec["npz"]).relative_to(original_root)
        except ValueError:
            rel = Path(rec["npz"]).name  # fallback: filename only
        out.append({**rec, "npz": str(npz_root / rel)})
    return out


def filter_records(
    records: List[dict],
    phones: List[str],
    stage: str,
    *,
    full_phone: bool = False,
) -> List[dict]:
    """pretrain: keep any readable NPZ. finetune: require >=1 present phone mask."""
    keys = {ph: mask_key(ph) for ph in phones}
    kept, missing_npz, no_phone, no_full_phone = [], 0, 0, 0
    for rec in records:
        npz_path = Path(rec["npz"])
        if not npz_path.exists():
            missing_npz += 1
            continue
        frames = rec.get("frames") or rec.get("_frames")
        try:
            if frames is not None and stage == "pretrain":
                kept.append({**rec, "_frames": int(frames)})
                continue
            with np.load(npz_path) as data:
                frames = int(data["z_e"].shape[0])
                if stage == "pretrain":
                    kept.append({**rec, "_frames": frames})
                elif any(k in data and data[k].any() for k in keys.values()):
                    if full_phone and not _npz_has_full_phone(data):
                        no_full_phone += 1
                        continue
                    kept.append({**rec, "_frames": frames})
                else:
                    no_phone += 1
        except Exception as exc:
            log.warning("Unreadable NPZ %s: %s", npz_path, exc)
            continue
    if missing_npz:
        log.warning("%d records have missing NPZ files — dropped", missing_npz)
    if no_phone:
        log.warning("%d records lack any of %s — dropped", no_phone, phones)
    if no_full_phone:
        log.warning(
            "%d records lack full-phone arrays — dropped "
            "(re-run prep_encodec_cond.py --full_phone_conditioning)",
            no_full_phone,
        )
    return kept


def _npz_has_full_phone(data) -> bool:
    needed = {"phone_ids", "phone_starts", "phone_ends", "phone_nchars"}
    if not needed <= set(data.files):
        return False
    n_phones = int(data["phone_starts"].shape[0])
    return (
        n_phones > 0
        and int(data["phone_ends"].shape[0]) == n_phones
        and int(data["phone_nchars"].shape[0]) == n_phones
        and int(data["phone_nchars"].sum()) == int(data["phone_ids"].shape[0])
    )


def upload_to_gcs(local_path: Path, bucket: str) -> None:
    """Copy a local file to gs://bucket/<same relative path>."""
    if not local_path.exists():
        return
    if not bucket.startswith("gs://"):
        bucket = "gs://" + bucket
    uri = f"{bucket.rstrip('/')}/{local_path.as_posix()}"
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), uri],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Uploaded %s -> %s", local_path, uri)
    except FileNotFoundError:
        log.warning("gcloud not found; skipped GCS upload for %s", local_path)
    except subprocess.CalledProcessError as exc:
        log.warning(
            "GCS upload failed for %s: %s",
            local_path,
            (exc.stderr or exc.stdout or "").strip(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    if args.epochs is None:
        args.epochs = PRETRAIN_EPOCHS if args.stage == "pretrain" else MAX_EPOCHS

    phones = args.phones
    if args.full_phone_conditioning:
        token_list, phone2id = build_alignment_char_vocab()
    else:
        token_list, phone2id = build_vocab(phones)
    full_phone_train = bool(args.full_phone_conditioning and args.stage == "finetune")
    outdir = Path(args.outdir) if args.outdir else Path(f"exp/cond_{args.stage}")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise SystemExit("--gpu was set but CUDA is not available.")
        n = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= n:
            raise SystemExit(f"--gpu {args.gpu} is out of range (have {n} CUDA devices).")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = optimizer = scheduler = None
    train_loader = val_loader = None

    try:
        log.info("Stage: %s  device: %s", args.stage, device)
        log.info(
            "Phones: %s  vocab_size: %d  full_phone: %s  outdir: %s",
            phones, len(token_list), args.full_phone_conditioning, outdir,
        )
        aliased = {p: ALIGN_LABEL_ALIASES[p] for p in phones if p in ALIGN_LABEL_ALIASES}
        if aliased:
            log.info("Alignment label aliases in play: %s", aliased)

        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            log.error(
                "Manifest not found: %s\nRun prep_encodec_cond.py --phone_masks %s first.",
                manifest_path, " ".join(phones),
            )
            sys.exit(1)

        npz_root = Path(args.npz_root) if args.npz_root else manifest_path.parent
        records = load_records(manifest_path)
        if npz_root != manifest_path.parent:
            records = rewrite_npz_root(records, manifest_path, npz_root)

        if args.dry_run:
            n_full = len(records)
            n_keep = max(1, int(n_full * 0.001))
            records = records[:n_keep]
            log.info(
                "Dry-run: using first %d / %d manifest records (0.1%%)",
                n_keep, n_full,
            )

        records = filter_records(
            records, phones, args.stage, full_phone=full_phone_train,
        )
        log.info("Loaded %d usable records from %s", len(records), manifest_path)
        if not records:
            log.error("No valid records found. Check manifest, NPZ paths, and --phones.")
            sys.exit(1)

        train_records, val_records = _split_records(records)
        log.info("Train: %d  Val: %d", len(train_records), len(val_records))

        _sa_kw = dict(sa_time_bands=0, sa_time_width=0, sa_freq_bands=0, sa_freq_width=0) \
            if args.no_augmentation else \
            dict(sa_time_bands=args.sa_time_bands, sa_time_width=args.sa_time_width,
                 sa_freq_bands=args.sa_freq_bands, sa_freq_width=args.sa_freq_width)
        if args.no_augmentation:
            log.info("Online SpecAugment disabled (--no_augmentation)")
        context_frames = args.context_frames if args.windowed_attention else 0
        if args.windowed_attention:
            log.info(
                "Windowed attention: trim to %d frames (±%d, ~±%.2f s at 75 Hz)",
                context_frames, context_frames // 2,
                context_frames / 2 / 75,
            )
        else:
            log.info("Windowed attention: disabled (full utterance)")
        train_ds = MultiPhoneNpzDataset(
            train_records, phones=phones, stage=args.stage, phone2id=phone2id,
            mlm_prob=args.mlm_prob, mean_span=args.mean_span,
            context_frames=context_frames,
            full_phone=full_phone_train,
            base_seed=args.seed, **_sa_kw,
        )
        val_ds = MultiPhoneNpzDataset(
            val_records, phones=phones, stage=args.stage, phone2id=phone2id,
            mlm_prob=args.mlm_prob, mean_span=args.mean_span,
            context_frames=context_frames,
            full_phone=full_phone_train,
            base_seed=args.seed + 1,
        )

        collate_fn = partial(collate_phone_batch, full_phone=full_phone_train)
        _persistent = args.num_workers > 0
        _prefetch = 4 if args.num_workers > 0 else None
        _pin = device.type == "cuda"
        _loader_kw = dict(
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=_pin,
        )
        if _persistent:
            _loader_kw["persistent_workers"] = True
            _loader_kw["prefetch_factor"] = _prefetch

        if args.windowed_attention and args.bucket_width > 0:
            # Upper bound on trimmed length after context windowing.
            bucket_lengths = lambda recs: [
                min(r["_frames"], context_frames) for r in recs
            ]
            train_batch_sampler = BucketBatchSampler(
                bucket_lengths(train_records), args.batch_size,
                bucket_width=args.bucket_width, shuffle=True, drop_last=True,
                seed=args.seed,
            )
            val_batch_sampler = BucketBatchSampler(
                bucket_lengths(val_records), args.batch_size,
                bucket_width=args.bucket_width, shuffle=False, drop_last=False,
                seed=args.seed + 1,
            )
            train_loader = DataLoader(
                train_ds, batch_sampler=train_batch_sampler, **_loader_kw,
            )
            val_loader = DataLoader(
                val_ds, batch_sampler=val_batch_sampler, **_loader_kw,
            )
            log.info(
                "Length bucketing enabled (width=%d frames, capped trim length)",
                args.bucket_width,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                drop_last=True, **_loader_kw,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, **_loader_kw,
            )

        model = build_model(token_list, args, odim=ENCODEC_DIM).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Model parameters: %s", f"{n_params:,}")

        if args.init and Path(args.init).exists():
            _init_ckpt = torch.load(args.init, map_location="cpu")
            _src_tl = _token_list_from_checkpoint(_init_ckpt)
            del _init_ckpt
            if _src_tl is not None:
                _src_phones = [t for t in _src_tl if t != ANY_TOKEN]
                _dst_phones = [t for t in token_list if t != ANY_TOKEN]
                _added   = [p for p in _dst_phones if p not in _src_phones]
                _removed = [p for p in _src_phones if p not in _dst_phones]
                log.info(
                    "Init checkpoint vocab: %d phones %s → training vocab: %d phones %s",
                    len(_src_phones), _src_phones,
                    len(_dst_phones), _dst_phones,
                )
                if _added:
                    log.info("  New phones (will init from <any>): %s", _added)
                if _removed:
                    log.info("  Dropped phones (not in this run): %s", _removed)
            load_pretrained_weights(
                model, Path(args.init), token_list, map_location=device,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0, weight_decay=args.weight_decay,
        )
        scheduler = NoamLR(optimizer, model_size=args.attn_dim, warmup_steps=args.warmup_steps)

        amp_enabled = args.amp and device.type == "cuda"
        if args.amp and not amp_enabled:
            log.warning("--amp requested but device is %s; running in fp32", device.type)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if amp_enabled:
            log.info("Mixed precision (AMP fp16) enabled")

        best_val = float("inf")
        best_path = outdir / "best.pt"
        history_path = outdir / "history.json"
        start_epoch = 1
        ckpt_val = float("nan")
        resumed = False
        history: List[dict] = []

        if history_path.exists():
            try:
                with open(history_path) as _hf:
                    history = json.load(_hf)
            except Exception as exc:
                log.warning("Could not load existing history.json: %s", exc)
                history = []

        if args.resume and Path(args.resume).exists():
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and amp_enabled:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            ckpt_val = ckpt.get("val_loss", float("nan"))
            resumed = True
            log.info(
                "Resumed from %s at epoch %d (checkpoint val_loss %.4f)",
                args.resume, start_epoch - 1, ckpt_val,
            )
            del ckpt

        if resumed:
            init_val = run_epoch(
                model, val_loader, optimizer, scheduler,
                device, train=False, epoch=0,
                gpu_clear_every_steps=args.gpu_clear_every_steps,
                scaler=scaler,
            )
            best_val = init_val
            log.info(
                "Resume baseline evaluation: val_loss=%.4f "
                "(checkpoint reported %.4f); %s kept unless beaten",
                best_val, ckpt_val, best_path,
            )

        def save(path: Path, epoch: int, val_loss: float):
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": val_loss,
                    "stage": args.stage,
                    "phones": phones,
                    "token_list": token_list,
                    "model_config": {
                        "attn_dim": args.attn_dim,
                        "attn_heads": args.attn_heads,
                        "ffn_units": args.ffn_units,
                        "enc_blocks": args.enc_blocks,
                        "dec_blocks": args.dec_blocks,
                        "dropout": args.dropout,
                        "postnet_layers": args.postnet_layers,
                        "postnet_chans": args.postnet_chans,
                        "postnet_filts": args.postnet_filts,
                        "detach_postnet_grad": args.detach_postnet_grad,
                        "full_phone_conditioning": args.full_phone_conditioning,
                    },
                },
                path,
            )

        for epoch in range(start_epoch, args.epochs + 1):
            _set_loader_epoch(train_loader, epoch)
            tr_loss = run_epoch(
                model, train_loader, optimizer, scheduler,
                device, train=True, epoch=epoch,
                gpu_clear_every_steps=args.gpu_clear_every_steps,
                scaler=scaler,
                detach_postnet_grad=args.detach_postnet_grad,
            )

            do_val = (epoch % args.val_every_epochs == 0) or (epoch == args.epochs)
            val_loss = None
            if do_val:
                _set_loader_epoch(val_loader, epoch)
                val_loss = run_epoch(
                    model, val_loader, optimizer, scheduler,
                    device, train=False, epoch=epoch,
                    gpu_clear_every_steps=args.gpu_clear_every_steps,
                    scaler=scaler,
                    detach_postnet_grad=args.detach_postnet_grad,
                )

            current_lr = scheduler.get_last_lr()[0]
            if do_val:
                log.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                    epoch, args.epochs, tr_loss, val_loss, current_lr,
                )
            else:
                log.info(
                    "Epoch %3d/%d  train=%.4f  val=skipped  lr=%.2e",
                    epoch, args.epochs, tr_loss, current_lr,
                )

            history.append({"epoch": epoch, "train_loss": tr_loss,
                             "val_loss": val_loss, "lr": current_lr})
            with open(history_path, "w") as _hf:
                json.dump(history, _hf, indent=2)
            if args.gcs_bucket:
                upload_to_gcs(history_path, args.gcs_bucket)

            if do_val and val_loss < best_val:
                best_val = val_loss
                save(best_path, epoch, best_val)
                log.info("  new best  val=%.4f -> %s", best_val, best_path)
                if args.gcs_bucket:
                    upload_to_gcs(best_path, args.gcs_bucket)

            if epoch % args.save_every == 0:
                save(outdir / f"epoch_{epoch:04d}.pt", epoch, val_loss)

            clear_gpu_cache(device)

        log.info("Done. Best val loss: %.4f", best_val)

    except KeyboardInterrupt:
        log.warning("Training interrupted — releasing GPU resources")
        raise
    except Exception:
        log.exception("Training failed — releasing GPU resources")
        raise
    finally:
        release_training_resources(
            device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=(train_loader, val_loader),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", required=True, choices=["pretrain", "finetune"],
                   help="pretrain = SSL random masking; finetune = phone-conditioned")
    p.add_argument("--phones", nargs="+", default=list(DEFAULT_TARGET_PHONES),
                   help=f"Target phones (default: {' '.join(DEFAULT_TARGET_PHONES)})")
    p.add_argument("--include_tts", action="store_true",
                   help="Use encodec_npzs_extended manifest (LibriTTS + VCTK)")
    p.add_argument("--manifest", type=str, default="manifest.jsonl",
                   help="Path to manifest.jsonl (default: ./manifest.jsonl)")
    p.add_argument("--npz_root", type=str, default=None,
                   help="Root directory containing the NPZ files. If omitted, "
                        "inferred as the directory containing --manifest (NPZ "
                        "paths are rewritten relative to that directory).")
    p.add_argument("--outdir", type=str, default=None,
                   help="Checkpoint dir (default: exp/cond_{stage})")
    p.add_argument(
        "--gcs_bucket", type=str, default=None,
        help="Upload best.pt and history.json to this bucket (e.g. gs://phone-paint-data)",
    )

    p.add_argument(
        "--gpu", type=int, default=None, metavar="IDX",
        help="Select CUDA GPU index (e.g. --gpu 0). Default: auto-select.",
    )
    p.add_argument("--epochs", type=int, default=None,
                   help=f"Max epochs (default: {PRETRAIN_EPOCHS} pretrain, {MAX_EPOCHS} finetune)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument(
        "--bucket_width", type=int, default=BUCKET_WIDTH,
        help=f"Bucket trimmed samples by length when --windowed_attention is set; "
             f"0=disable (default: {BUCKET_WIDTH}). Ignored without --windowed_attention.",
    )
    p.add_argument("--save_every", type=int, default=SAVE_EVERY)
    p.add_argument("--val_every_epochs", type=int, default=1, metavar="N",
                   help="Run validation every N epochs (and always on the final epoch). "
                        "Default: 1 (every epoch).")
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true",
                   help="Enable fp16 mixed precision (CUDA only). Default: off (fp32)")

    p.add_argument("--init", type=str, default=None,
                   help="Load weights only; phone embeddings matched by token name "
                        "when --phones grows (e.g. finetune from pretrain ckpt)")
    p.add_argument("--resume", type=str, default=None,
                   help="Full resume (model + optimizer + scheduler + epoch)")

    # Model capacity / regularisation
    p.add_argument("--attn_dim", type=int, default=ATTN_DIM)
    p.add_argument("--attn_heads", type=int, default=ATTN_HEADS)
    p.add_argument("--ffn_units", type=int, default=FFN_UNITS)
    p.add_argument("--enc_blocks", type=int, default=ENC_BLOCKS)
    p.add_argument("--dec_blocks", type=int, default=DEC_BLOCKS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--postnet_layers", type=int, default=POSTNET_LAYERS)
    p.add_argument("--postnet_chans", type=int, default=POSTNET_CHANS)
    p.add_argument("--postnet_filts", type=int, default=POSTNET_FILTS)
    p.add_argument(
        "--detach_postnet_grad", action="store_true",
        help="Detach before_outs from the postnet loss path so the postnet loss "
             "term updates only postnet weights, not the encoder/decoder.  "
             "No architecture change; affects gradient routing only.",
    )
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    # Latent SpecAugment (input-only; clean target preserved)
    p.add_argument("--no_augmentation", action="store_true",
                   help="Disable online SpecAugment on the training set "
                        "(overrides --sa_* flags).  Speeds up data loading.")
    p.add_argument("--sa_time_bands", type=int, default=2)
    p.add_argument("--sa_time_width", type=int, default=10,
                   help="Max width (frames) of each time mask band")
    p.add_argument("--sa_freq_bands", type=int, default=2)
    p.add_argument("--sa_freq_width", type=int, default=16,
                   help="Max width (channels) of each feature mask band")

    # Pretrain masking distribution
    p.add_argument("--mlm_prob", type=float, default=PRETRAIN_MLM_PROB,
                   help="Fraction of frames masked during SSL pretraining")
    p.add_argument("--mean_span", type=int, default=PRETRAIN_MEAN_SPAN,
                   help="Mean masked span length (frames) during SSL pretraining")
    p.add_argument(
        "--windowed_attention", action="store_true",
        help="Trim each sample to a local context window around the masked span and "
             "enable length bucketing. Without this flag, use the full utterance "
             "(original behaviour).",
    )
    p.add_argument(
        "--context_frames", type=int, default=CONTEXT_FRAMES,
        help=f"Context window size when --windowed_attention is set "
             f"(default: {CONTEXT_FRAMES} = ±{CONTEXT_FRAMES // 2} frames "
             f"≈ ±{CONTEXT_FRAMES / 2 / 75:.2f}s at 75 Hz). "
             f"When the span is within context_frames//2 of the utterance boundary the "
             f"window is clipped to the utterance edge; no padding is added.",
    )
    p.add_argument(
        "--full_phone_conditioning", action="store_true",
        help="Finetune with char-tokenized full utterance phone strings and SEGA "
             "alignment (requires prep_encodec_cond.py --full_phone_conditioning). "
             "Pretrain still uses a single <any> token but shares the char vocab.",
    )
    p.add_argument(
        "--gpu_clear_every_steps", type=int, default=GPU_CLEAR_EVERY_STEPS,
        help=f"Clear CUDA cache every N steps (batches). 0=disable. Default: {GPU_CLEAR_EVERY_STEPS}",
    )
    p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Load and train on only the first 0.1%% of manifest records "
             "(at least 1), before NPZ filtering. Quick end-to-end smoke test.",
    )
    return p.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        log.info("Stopped by user")
        sys.exit(130)
