"""Frozen EEGPT with the original SSVEP crop and summary-slot mean."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from mgpa.data.prepared import pinned_path

CHECKPOINT_SHA256 = "fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621"
CHECKPOINT_REVISION = "e41cb3ae2ce4fd9eb736862292c91f8128d15618"
CHANNELS = ("POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2")
N_TIMES, SFREQ, N_PATCHES, EMBED_NUM, EMBED_DIM = 500, 250, 15, 4, 512


def preprocess_eegpt(trials):
    """Crop the fixed response interval and apply per-channel population scaling."""
    values = np.asarray(trials, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 710):
        raise ValueError(f"Expected [trial,8,710], got {values.shape}")
    # Copy prevents preprocessing from modifying caller-owned raw signals.
    # Values and arithmetic order match the original crop/z-score exactly.
    values = values[:, :, 160:660].copy()
    values -= values.mean(axis=-1, keepdims=True)
    values /= np.maximum(values.std(axis=-1, keepdims=True), 1e-6)
    if not np.isfinite(values).all():
        raise ValueError("Invalid EEGPT input")
    return values.astype(np.float32)


class FrozenEEGPTEncoder:
    """Frozen published EEGPT with the SSVEP channel and output interface."""
    def __init__(self, checkpoint: Path, *, device="cpu"):
        """Authenticate checkpoint tensors and create a frozen encoder on the requested device."""
        from .runtime import ensure_runtime
        ensure_runtime()
        import inspect
        import importlib.metadata
        import mne
        import torch
        from braindecode.models import EEGPT
        from safetensors.torch import load_file
        from mgpa.data.prepared import sha256

        checkpoint = pinned_path(checkpoint, CHECKPOINT_SHA256)
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; no silent backend switch")
        info = mne.create_info(list(CHANNELS), SFREQ, ch_types="eeg")
        info.set_montage("standard_1005")
        model = EEGPT(n_chans=8, chs_info=info["chs"], n_times=N_TIMES,
                      sfreq=float(SFREQ), return_encoder_output=True, chan_proj_type="none")
        checkpoint_state = load_file(str(checkpoint), device="cpu")
        state = model.state_dict()
        compatible = {key: value for key, value in checkpoint_state.items()
                      if key in state and tuple(value.shape) == tuple(state[key].shape)}
        skipped = tuple(sorted(set(checkpoint_state) - set(compatible)))
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if unexpected or skipped != ("chans_id",) or tuple(missing) != ("chans_id",):
            raise ValueError("Unexpected frozen EEGPT conversion")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.manifest = {"model_id": "braindecode/eegpt-pretrained", "revision": CHECKPOINT_REVISION,
                         "checkpoint_sha256": CHECKPOINT_SHA256, "device": device,
                         "encoder_implementation_sha256": sha256(inspect.getfile(EEGPT)),
                         "versions": {name: importlib.metadata.version(name) for name in ("torch", "braindecode", "mne", "numpy", "safetensors")},
                         "output_shape": [N_PATCHES, EMBED_NUM, EMBED_DIM],
                         "preprocessing": "crop160:660 at250Hz; per-channel population z-score floor1e-6",
                         "channels": list(CHANNELS), "weights_frozen": True}

    def encode_patch_mean_grid(self, trials, *, batch_size=48):
        """Encode trials and average summary slots while retaining all time tokens."""
        import torch
        values = preprocess_eegpt(trials)
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                encoded = self.model(torch.from_numpy(values[start:start+batch_size]).to(self.device))
                if tuple(encoded.shape[1:]) != (N_PATCHES, EMBED_NUM, EMBED_DIM):
                    raise ValueError("Unexpected encoder output shape")
                pieces.append(encoded.mean(dim=2).flatten(1).float().cpu().numpy())
        result = np.concatenate(pieces).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite encoder features")
        return result
