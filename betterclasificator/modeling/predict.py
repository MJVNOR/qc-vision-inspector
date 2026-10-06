"""Pilot inference: DINOv2-LogReg + patch-local rescue (no registry needed).

Artifacts (see train.py, all under models/):
  logreg_dinov2.joblib, thresholds_dinov2_prod.json,
  dino_patch_bank.pt, thresholds_local.json

Usage: uv run python -m betterclasificator.modeling.predict IMG [IMG ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from betterclasificator import config


def load_artifacts():
    import joblib

    mdir = config.MODELS_DIR
    missing = [
        p.name
        for p in [
            mdir / "logreg_dinov2.joblib",
            mdir / "thresholds_dinov2_prod.json",
            mdir / "dino_patch_bank.pt",
            mdir / "thresholds_local.json",
        ]
        if not p.exists()
    ]
    if missing:
        sys.exit(f"faltan artefactos en models/: {missing} (corre train.py --all)")
    clf = joblib.load(mdir / "logreg_dinov2.joblib")
    thr = json.loads((mdir / "thresholds_dinov2_prod.json").read_text())
    bank = torch.load(mdir / "dino_patch_bank.pt", map_location="cpu", weights_only=False)["bank"]
    loc = json.loads((mdir / "thresholds_local.json").read_text())
    proc = AutoImageProcessor.from_pretrained(config.DINO_BACKBONE)
    bm = AutoModel.from_pretrained(config.DINO_BACKBONE).eval()
    for p in bm.parameters():
        p.requires_grad_(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return clf, thr["cut_produccion"], bank.to(dev), loc["cut_rescue_q99"], proc, bm.to(dev), dev


def predict(paths):
    from betterclasificator.features import patch_scores_1nn

    clf, cut, bank, cut99, proc, bm, dev = load_artifacts()
    rows = []
    with torch.inference_mode():
        for path in paths:
            im = Image.open(path).convert("RGB")
            inp = proc(images=im, return_tensors="pt")
            h = bm(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state
            dino = float(clf.predict_proba(h[:, 0, :].cpu().numpy())[:, 1][0])
            patches = h[:, 1:, :].squeeze(0)
            local = patch_scores_1nn(patches, bank)
            bad = (dino >= cut) or (local >= cut99)
            rows.append({"file": str(path), "dino": round(dino, 4), "local": round(local, 2), "bad": bad})
            print(f"{path}: dino={dino:.4f} local={local:.2f} -> {'BAD' if bad else 'good'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    predict(a.images)


if __name__ == "__main__":
    main()
