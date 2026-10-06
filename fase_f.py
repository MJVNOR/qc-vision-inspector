import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Fase F — small defect detection (DINOv2 patch tokens + top-k)
    D-alpha congelado como baseline (0.95 / 0.055). Score local patch-level,
    recall por tamano de defecto, fusion sin tocar bases ni sobreajustar bad_022.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_f ready")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from PIL import Image as _PIf
    FLAB = pd.read_csv(Path("transistor_binary") / "labels.csv")
    FY = (FLAB["label"] == "bad").to_numpy(int)
    FFOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(FLAB, FY))
    FREZ = [224, 512]
    FAREA, FBUCKET = {}, {}
    for _r in FLAB[FLAB.label == "bad"].itertuples():
        _mp = Path("mvtec_anomaly_detection/transistor/ground_truth") / _r.defect / (_r.original_path.split("/")[-1].replace(".png", "_mask.png"))
        _a = 100 * (np.asarray(_PIf.open(_mp).convert("L")) > 127).mean()
        FAREA[_r.filename] = round(float(_a), 2)
        FBUCKET[_r.filename] = "tiny<1%" if _a < 1 else ("mid1-3%" if _a < 3 else "big>3%")
    pd.Series(FAREA).describe().round(2).to_string()
    return FFOLDS, FLAB, FREZ, FY, Path, np


@app.cell
def _(FFOLDS, FLAB, FREZ, FY, Path, np):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from sklearn.neighbors import NearestNeighbors
    from PIL import Image
    _proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bm = AutoModel.from_pretrained("facebook/dinov2-base")
    _bm.eval()
    for _p in _bm.parameters():
        _p.requires_grad_(False)
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bm.to(_dev)
    def _patches(_idx, _sz):
        _r = FLAB.iloc[int(_idx)]
        _im = Image.open(Path("transistor_binary") / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB")
        _inp = _proc(images=_im, size={"height": _sz, "width": _sz}, return_tensors="pt")
        with torch.inference_mode():
            _h = _bm(**{k: v.to(_dev) for k, v in _inp.items()}).last_hidden_state[:, 1:, :].squeeze(0)
        return _h.cpu()
    fl_scores = {}
    for _sz in FREZ:
        for _fold, (_tr, _te) in enumerate(FFOLDS):
            _good_tr = [int(_i) for _i in _tr if FY[_i] == 0]
            _tel = [int(_x) for _x in _te]
            _bank = torch.cat([_patches(_i, _sz) for _i in _good_tr])
            if _sz > 224:
                _g = torch.Generator().manual_seed(42)
                _bank = _bank[torch.randperm(_bank.shape[0], generator=_g)[:max(1, _bank.shape[0] // 10)]]
            _nn = NearestNeighbors(n_neighbors=1).fit(_bank.numpy())
            _SM, _DD = {}, {}
            for _ii in _tel:
                _pa = _patches(_ii, _sz).numpy()
                _d = _nn.kneighbors(_pa)[0].flatten()
                _ds = np.sort(_d)
                _SM[_ii] = {"max": float(_ds[-1]), "top10": float(_ds[-10:].mean()), "p95": float(np.quantile(_d, 0.95)), "p99": float(np.quantile(_d, 0.99))}
            for _agg in ["max", "top10", "p95", "p99"]:
                fl_scores[(_sz, _agg, _fold)] = (_tel, np.array([_SM[_ii][_agg] for _ii in _tel]))
            print(f"res {_sz} fold {_fold} done bank={tuple(_bank.shape)}", flush=True)
            del _bank
    print("local scores ok")
    return


if __name__ == "__main__":
    app.run()
