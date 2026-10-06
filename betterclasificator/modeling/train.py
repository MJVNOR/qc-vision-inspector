"""Materialize production artifacts into models/ (replaces registry dependency).

Recipes mirror the notebooks (same hyperparams, full data, seed 42):
  --logreg : DINOv2 CLS + LogisticRegression(C=1.0, balanced) on all 313,
             saves models/logreg_dinov2.joblib + models/thresholds_dinov2_prod.json
             (cut = quantile-95 of GOOD train scores, same rule as fase A).
  --bank   : DINOv2-224 patch bank over 273 GOOD + rescue q99 cut,
             saves models/dino_patch_bank.pt + models/thresholds_local.json
             (same recipe as fase F; torch GPU 1-NN).
  --em     : EfficientAd-medium retrain on all GOOD (5000 steps, batch 1),
             saves models/efficientad_em.ckpt (needs ~40 min GPU + ImageNette
             cache at data/external/imagenette; same recipe as fase C).

Usage: uv run python -m betterclasificator.modeling.train --logreg --bank
"""

import argparse
import json
from pathlib import Path

from betterclasificator import config


def cmd_logreg():
    import joblib
    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression

    labels = pd.read_csv(config.LABELS_CSV)
    y = (labels["label"] == "bad").to_numpy(int)
    emb = config.MODELS_DIR / "fase_a_emb_dinov2.pt"
    if not emb.exists():
        emb = Path("fase_a_emb_dinov2.pt")  # ubicación pre-migración
    saved = torch.load(emb, map_location="cpu", weights_only=False)
    order = [str(f) for f in saved["fnames"]]
    lab = labels.set_index("filename").loc[order]
    y = (lab["label"] == "bad").to_numpy(int)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)
    clf.fit(saved["X"], y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.MODELS_DIR / "logreg_dinov2.joblib")
    cut = float(__import__("numpy").quantile(clf.predict_proba(saved["X"][y == 0])[:, 1], 0.95))
    (config.MODELS_DIR / "thresholds_dinov2_prod.json").write_text(
        json.dumps(
            {
                "cut_produccion": cut,
                "regla": "cuantil 95 scores GOOD full-train (273 good / 40 bad)",
                "backbone": config.DINO_BACKBONE,
                "C": 1.0,
            },
            indent=1,
        )
    )
    print(f"logreg ok cut={cut:.4f}")


def cmd_bank():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from betterclasificator.features import quantile_cut

    import pandas as pd

    labels = pd.read_csv(config.LABELS_CSV)
    proc = AutoImageProcessor.from_pretrained(config.DINO_BACKBONE)
    bm = AutoModel.from_pretrained(config.DINO_BACKBONE).eval()
    for p in bm.parameters():
        p.requires_grad_(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm.to(dev)
    parts = []
    with torch.inference_mode():
        for _, r in labels[labels.label == "good"].iterrows():
            im = Image.open(config.DATA_PROCESSED / "good" / r.filename).convert("RGB")
            inp = proc(images=im, size={"height": 224, "width": 224}, return_tensors="pt")
            h = bm(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state[:, 1:, :]
            parts.append(h.squeeze(0).cpu())
    bank = torch.cat(parts)
    gen = torch.Generator().manual_seed(config.SEED)
    bank = bank[torch.randperm(bank.shape[0], generator=gen)[: max(1, bank.shape[0] // 10)]]
    scores = []
    with torch.inference_mode():
        for i in range(0, len(parts), 8):
            d = torch.cdist(torch.cat(parts[i : i + 8]).view(-1, 768).to(dev), bank.to(dev))
            s = d.view(len(parts[i : i + 8]), -1, bank.shape[0]).min(dim=2).values.max(dim=1).values
            scores.extend(s.cpu().tolist())
    scores = np.array(scores)
    cut = quantile_cut(scores, 0.01)  # rescue q99 over GOOD
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"bank": bank, "backbone": config.DINO_BACKBONE, "layers": "patch-tokens-224", "seed": config.SEED},
        config.MODELS_DIR / "dino_patch_bank.pt",
    )
    (config.MODELS_DIR / "thresholds_local.json").write_text(
        json.dumps({"cut_rescue_q99": cut, "regla": "cuantil 99 scores GOOD full-train (fase F)"}, indent=1)
    )
    print(f"bank ok {tuple(bank.shape)} cut99={cut:.2f}")


def cmd_em():
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from lightning.pytorch import LightningDataModule
    from anomalib.data.dataclasses.torch import ImageBatch
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd
    import pandas as pd

    labels = pd.read_csv(config.LABELS_CSV)

    class _DM(LightningDataModule):
        def __init__(self, tensors):
            super().__init__()
            self.train_batch_size = 1
            self.name = "transistor"
            self.category = "transistor"
            self._t = tensors

        def train_dataloader(self):
            return DataLoader(
                self._t, batch_size=1, shuffle=True, collate_fn=lambda b: ImageBatch(image=torch.stack(b))
            )

    tensors = []
    for _, r in labels[labels.label == "good"].iterrows():
        im = Image.open(config.DATA_PROCESSED / "good" / r.filename).convert("RGB").resize((256, 256))
        tensors.append(torch.from_numpy(__import__("numpy").asarray(im, dtype="float32") / 255.0).permute(2, 0, 1))
    model = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
    engine = Engine(logger=False, default_root_dir="results_prod_em", max_steps=5000)
    engine.fit(model=model, datamodule=_DM(tensors))
    print("em ok -> results_prod_em/ (copy best .ckpt to models/efficientad_em.ckpt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logreg", action="store_true")
    ap.add_argument("--bank", action="store_true")
    ap.add_argument("--em", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all or a.logreg:
        cmd_logreg()
    if a.all or a.bank:
        cmd_bank()
    if a.all or a.em:
        cmd_em()


if __name__ == "__main__":
    main()
