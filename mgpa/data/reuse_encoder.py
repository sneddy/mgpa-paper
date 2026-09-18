"""Frozen EEGPT for one-second ERP epochs; explicit checkpoint required."""
from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np

from scipy.signal import resample_poly

ALLOWED = set("FP1 FPZ FP2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 O1 OZ O2".split())

def digest(path):
    """Compute a streaming SHA256 digest for checkpoint and encoder provenance."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def select_channels(channels):
    """Select unique EEGPT-supported scalp channels in their original order."""
    names = [str(x) for x in channels]
    if len(set(x.upper() for x in names)) != len(names):
        raise ValueError("duplicate channel names")
    keep = [i for i, x in enumerate(names) if x.upper() in ALLOWED]
    if len(keep) < 8:
        raise ValueError("too few supported scalp channels")
    return keep, [names[i] for i in keep], [x for x in names if x.upper() not in ALLOWED]

def preprocess(x, sfreq, normalization):
    """Resample one-second ERP epochs to 256 Hz and apply fixed 20 μV scaling."""
    values = np.array(x, dtype=np.float32, copy=True)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("expected finite [epoch,channel,time] in volts")
    if float(sfreq) not in (128., 256.):
        raise ValueError("only audited 128/256Hz ERP caches supported")
    if sfreq == 128:
        values = resample_poly(values, 2, 1, axis=-1, padtype="line").astype(np.float32)
    if values.shape[-1] != 256:
        raise ValueError("expect exactly one-second ERP epoch; no silent crop/padding")
    if normalization == "fixed_20uv":
        values /= 20e-6
    else:
        raise ValueError(normalization)
    return values


class Encoder:
    """Frozen EEGPT encoder retaining all seven windows and four summary tokens."""
    def __init__(self, channels, *, checkpoint, device="cpu"):
        """Load the explicit checkpoint and verify channel-aware tensor compatibility."""
        import mne
        import torch
        from braindecode.models import EEGPT
        from safetensors.torch import load_file
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; do not silently change backend")
        torch.manual_seed(20260909)
        info = mne.create_info(channels, 256., ch_types="eeg")
        info.set_montage("standard_1005", match_case=False)
        model = EEGPT(n_chans=len(channels), chs_info=info["chs"], n_times=256,
                      sfreq=256., return_encoder_output=True, chan_proj_type="none")
        state = load_file(str(checkpoint), device="cpu")
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k != "chans_id" and k in model_state and v.shape == model_state[k].shape}
        skipped = sorted(set(state) - set(compatible))
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if skipped != ["chans_id"] or list(missing) != ["chans_id"] or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: {skipped=}, {missing=}, {unexpected=}")
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device).eval()
        self.device = device
        self.manifest = {
            "model": "braindecode/eegpt-pretrained", "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint), "loaded_tensors": len(compatible),
            "channel_ids": model.chans_id.detach().cpu().tolist(),
            "device": device, "n_times": 256, "sfreq": 256., "patches": model.n_patches,
            "torch_version": torch.__version__, "mne_version": mne.__version__,
            "encoder_implementation_sha256": digest(__import__("inspect").getfile(EEGPT)),
        }

    def encode(self, values, batch_size=24):
        """Encode batches without gradients and return finite full-token arrays."""
        import torch
        parts = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size], device=self.device)
                parts.append(self.model(batch).float().cpu().numpy())
        tokens = np.concatenate(parts)
        if not np.isfinite(tokens).all():
            raise FloatingPointError("nonfinite encoder output")
        return tokens
