"""PyTorch reconstruction of the MAPS Keras acoustic model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from maps_torch.phones import N_CLASSES, N_FEATURES, PHONES


class MapsAcousticModel(nn.Module):
    """Masking → 3×(LayerNorm + BiLSTM) → LayerNorm → Linear+softmax."""

    def __init__(self, hidden_size: int = 128, mask_value: float = -1.0):
        super().__init__()
        self.mask_value = mask_value
        self.input_ln = nn.LayerNorm(N_FEATURES, eps=1e-3)
        self.lstm1 = nn.LSTM(N_FEATURES, hidden_size, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_size * 2, eps=1e-3)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(hidden_size * 2, eps=1e-3)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.ln3 = nn.LayerNorm(hidden_size * 2, eps=1e-3)
        self.classifier = nn.Linear(hidden_size * 2, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ``[B, T, 39]``; return posterior probabilities ``[B, T, 61]``."""
        mask = (x == self.mask_value).all(dim=-1, keepdim=True)
        x = x.masked_fill(mask, 0.0)

        x = self.input_ln(x)
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x, _ = self.lstm2(x)
        x = self.ln2(x)
        x, _ = self.lstm3(x)
        x = self.ln3(x)
        logits = self.classifier(x)
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict_numpy(self, x: np.ndarray, device: torch.device) -> np.ndarray:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return self.forward(tensor).cpu().numpy()


def _keras_to_torch_lstm(
    state: dict[str, torch.Tensor],
    prefix: str,
    kernel: np.ndarray,
    recurrent: np.ndarray,
    bias: np.ndarray,
    *,
    reverse: bool = False,
) -> None:
    direction = "reverse" if reverse else ""
    suffix = f"_{direction}" if direction else ""
    state[f"{prefix}.weight_ih_l0{suffix}"] = torch.from_numpy(kernel.T.copy())
    state[f"{prefix}.weight_hh_l0{suffix}"] = torch.from_numpy(recurrent.T.copy())
    state[f"{prefix}.bias_ih_l0{suffix}"] = torch.from_numpy(bias.copy())
    state[f"{prefix}.bias_hh_l0{suffix}"] = torch.zeros_like(state[f"{prefix}.bias_ih_l0{suffix}"])


def _load_ln(state: dict[str, torch.Tensor], prefix: str, gamma: np.ndarray, beta: np.ndarray) -> None:
    state[f"{prefix}.weight"] = torch.from_numpy(gamma.copy())
    state[f"{prefix}.bias"] = torch.from_numpy(beta.copy())


def _find_weight(weights_by_keras_name: dict[str, np.ndarray], keras_name: str) -> np.ndarray:
    if keras_name not in weights_by_keras_name:
        raise KeyError(f"Missing exported weight {keras_name}")
    return weights_by_keras_name[keras_name]


def state_dict_from_layers(
    layers_info: list[dict[str, Any]],
    weights_by_keras_name: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Map exported Keras weights to Torch using authoritative layer metadata."""
    state: dict[str, torch.Tensor] = {}
    ln_targets = ["input_ln", "ln1", "ln2", "ln3"]
    lstm_targets = ["lstm1", "lstm2", "lstm3"]
    ln_i = 0
    lstm_i = 0

    for layer in layers_info:
        cls = layer["class_name"]
        name = layer["name"]
        if cls == "Masking":
            continue
        if cls == "LayerNormalization":
            prefix = ln_targets[ln_i]
            ln_i += 1
            gamma = _find_weight(weights_by_keras_name, f"{name}/gamma:0")
            beta = _find_weight(weights_by_keras_name, f"{name}/beta:0")
            _load_ln(state, prefix, gamma, beta)
        elif cls == "Bidirectional":
            prefix = lstm_targets[lstm_i]
            lstm_i += 1
            weight_names = [w["name"] for w in layer["weights"]]
            fwd_prefix = re.sub(
                r"/(kernel|recurrent_kernel|bias):0$",
                "",
                next(n for n in weight_names if "/forward_" in n),
            )
            bwd_prefix = re.sub(
                r"/(kernel|recurrent_kernel|bias):0$",
                "",
                next(n for n in weight_names if "/backward_" in n),
            )
            for is_rev, cell_prefix in ((False, fwd_prefix), (True, bwd_prefix)):
                kernel = _find_weight(weights_by_keras_name, f"{cell_prefix}/kernel:0")
                recurrent = _find_weight(weights_by_keras_name, f"{cell_prefix}/recurrent_kernel:0")
                bias = _find_weight(weights_by_keras_name, f"{cell_prefix}/bias:0")
                _keras_to_torch_lstm(state, prefix, kernel, recurrent, bias, reverse=is_rev)
        elif cls == "TimeDistributed":
            kernel = _find_weight(weights_by_keras_name, f"{name}/kernel:0")
            bias = _find_weight(weights_by_keras_name, f"{name}/bias:0")
            state["classifier.weight"] = torch.from_numpy(kernel.T.copy())
            state["classifier.bias"] = torch.from_numpy(bias.copy())
        else:
            raise ValueError(f"Unexpected layer class {cls}")

    if ln_i != 4 or lstm_i != 3:
        raise ValueError(f"Expected 4 LayerNorm and 3 BiLSTM layers, got {ln_i} and {lstm_i}")
    return state


def state_dict_from_npz(weights: dict[str, np.ndarray], layers_path: Path) -> dict[str, torch.Tensor]:
    layers_info = json.loads(layers_path.read_text())
    keras_map: dict[str, np.ndarray] = {}
    for layer in layers_info:
        for w in layer["weights"]:
            npz_key = w["name"].replace("/", "__").replace(":", "_")
            keras_map[w["name"]] = weights[npz_key]
    return state_dict_from_layers(layers_info, keras_map)


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> MapsAcousticModel:
    """Load a ``.pt`` checkpoint produced by ``tools/build_torch_checkpoints.py``."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta: dict[str, Any] = payload["metadata"]
    if meta["phones"] != PHONES:
        raise ValueError("Checkpoint phone inventory does not match MAPS reference order")
    if meta["n_features"] != N_FEATURES or meta["n_classes"] != N_CLASSES:
        raise ValueError("Checkpoint feature/class dimensions mismatch")

    model = MapsAcousticModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(device)
