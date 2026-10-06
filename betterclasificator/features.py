"""Shared feature helpers (same recipes as notebooks fase_b/f/g).

IQR normalization and patch-bank scoring. Thresholds are always calibrated
on GOOD-train scores only; see notebooks for the CV loops.
"""

import numpy as np


def iqr_params(scores_good):
    """Median/IQR fitted on GOOD scores. Returns (median, iqr)."""
    scores_good = np.asarray(scores_good, dtype=np.float64)
    med = float(np.median(scores_good))
    iqr = float(np.subtract(*np.percentile(scores_good, [75, 25])))
    return med, max(iqr, 1e-9)


def normalize(scores, median, iqr):
    """Robust z-score with pre-fitted (median, iqr)."""
    return (np.asarray(scores, dtype=np.float64) - median) / max(iqr, 1e-9)


def quantile_cut(scores_good, far_target=0.05):
    """Cut leaving `far_target` fraction of GOOD above it (train only)."""
    return float(np.quantile(np.asarray(scores_good, dtype=np.float64), 1 - far_target))


def patch_scores_1nn(patches, bank, chunk=4096):
    """Max patch anomaly score (1-NN distance) with torch GPU, chunked.

    patches: (P, D) tensor for one image. bank: (N, D) tensor.
    Same exact 1-NN as sklearn, 10-50x faster on CUDA.
    """
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(bank), chunk):
            d = torch.cdist(patches, bank[i : i + chunk])
            out.append(d.min(dim=1).values)
    return torch.stack(out, dim=1).min(dim=1).values.max().item()
