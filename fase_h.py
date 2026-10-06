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
    # Fase H — robustez geometrica y piloto (D-alpha + rescue congelados)
    Normalizacion geometrica determinista + re-evaluacion con mismo protocolo.
    Sin modelos nuevos. Comparacion directa contra Fase G.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_h ready")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    HLAB = pd.read_csv(Path("transistor_binary") / "labels.csv")
    HY = (HLAB["label"] == "bad").to_numpy(int)
    _rows = []
    for _r in HLAB.itertuples():
        _im = Image.open(Path("transistor_binary") / ("good" if _r.label == "good" else "bad") / _r.filename).convert("L")
        _a = np.asarray(_im)
        _m = _a < 100
        _ys, _xs = np.nonzero(_m)
        _rows.append({"file": _r.filename, "cx": round(float(_xs.mean()), 1), "cy": round(float(_ys.mean()), 1),
            "w": int(_xs.max() - _xs.min()), "h": int(_ys.max() - _ys.min()), "fill": round(float(_m.mean()), 3)})
    hgeo = pd.DataFrame(_rows)
    hgeo[["cx", "cy", "w", "h", "fill"]].describe().round(1).to_string()
    return HLAB, HY, Image, Path, hgeo, np, pd


@app.cell
def _(HLAB, Image, Path, hgeo, np, pd):
    _TX, _TY, _DB = 479.0, 497.0, 25.0
    _ND = Path("transistor_norm")
    (_ND / "good").mkdir(parents=True, exist_ok=True)
    (_ND / "bad").mkdir(parents=True, exist_ok=True)
    _shifts = []
    for _r in HLAB.itertuples():
        _row = hgeo[hgeo.file == _r.filename].iloc[0]
        _dx, _dy = _TX - _row.cx, _TY - _row.cy
        _folder = "good" if _r.label == "good" else "bad"
        _im = Image.open(Path("transistor_binary") / _folder / _r.filename).convert("RGB")
        if max(abs(_dx), abs(_dy)) <= _DB:
            _a = np.asarray(_im)
            _ix, _iy = int(round(_dx)), int(round(_dy))
            _b = np.roll(_a, shift=(_iy, _ix), axis=(0, 1))
            if _iy > 0:
                _b[:_iy, :] = np.broadcast_to(_b[_iy:_iy + 1, :], (_iy, _b.shape[1], 3))
            elif _iy < 0:
                _b[_iy:, :] = np.broadcast_to(_b[_iy - 1:_iy, :], (-_iy, _b.shape[1], 3))
            if _ix > 0:
                _b[:, :_ix] = np.broadcast_to(_b[:, _ix:_ix + 1], (_b.shape[0], _ix, 3))
            elif _ix < 0:
                _b[:, _ix:] = np.broadcast_to(_b[:, _ix - 1:_ix], (_b.shape[0], -_ix, 3))
            _im = Image.fromarray(_b)
            _corr = True
        else:
            _corr = False
        _im.save(_ND / _folder / _r.filename)
        _shifts.append({"file": _r.filename, "dx": round(float(_dx), 1), "dy": round(float(_dy), 1), "corrected": _corr})
    hshifts = pd.DataFrame(_shifts)
    print(hshifts.corrected.value_counts().to_string())
    print(hshifts[~hshifts.corrected][["file", "dx", "dy"]].to_string())
    return


@app.cell
def _(HLAB, Image, Path):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    _prH = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bmH = AutoModel.from_pretrained("facebook/dinov2-base")
    _bmH.eval()
    for _p in _bmH.parameters():
        _p.requires_grad_(False)
    _devH = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bmH.to(_devH)
    _HC, _HP = [], []
    with torch.inference_mode():
        for _i in range(len(HLAB)):
            _r = HLAB.iloc[_i]
            _im = Image.open(Path("transistor_norm") / ("good" if _r.label == "good" else "bad") / _r.filename).convert("RGB")
            _o = _bmH(**{k: v.to(_devH) for k, v in _prH(images=_im, size={"height": 224, "width": 224}, return_tensors="pt").items()})
            _HC.append(_o.last_hidden_state[:, 0, :].squeeze(0).cpu())
            _HP.append(_o.last_hidden_state[:, 1:, :].squeeze(0).cpu())
            if (_i + 1) % 100 == 0:
                print(f"fw {_i + 1}/313", flush=True)
    HCLS = torch.stack(_HC)
    HPATCH = torch.stack(_HP)
    print(tuple(HPATCH.shape))
    del _HC, _HP
    return AutoImageProcessor, AutoModel, HCLS, HPATCH, torch


@app.cell
def _(HLAB, HY, Path, np, torch):
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine
    from anomalib.data.dataclasses.torch import ImageBatch
    from lightning.pytorch import LightningDataModule
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedKFold
    import PIL.Image as _PIh
    class _DMH(LightningDataModule):
        def __init__(self, _tensors):
            super().__init__()
            self.train_batch_size = 1
            self.name = "transistor"
            self.category = "transistor"
            self._tensors = _tensors
        def train_dataloader(self):
            return DataLoader(self._tensors, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
    _HF = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(HLAB, HY))
    for _fold, (_tr, _te) in enumerate(_HF):
        _gt = [int(_i) for _i in _tr if HY[_i] == 0]
        _t = []
        for _ii in _gt:
            _r = HLAB.iloc[_ii]
            _im = _PIh.open(Path("transistor_norm") / "good" / _r.filename).convert("RGB").resize((256, 256))
            _t.append(torch.from_numpy(np.asarray(_im, dtype=np.float32) / 255.0).permute(2, 0, 1))
        _m = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
        _eng = Engine(logger=False, default_root_dir="results_h_em", max_steps=5000)
        _eng.fit(model=_m, datamodule=_DMH(_t))
        print(f"em norm fold {_fold} done", flush=True)
        del _m, _t
    return DataLoader, EfficientAd, Engine, ImageBatch, StratifiedKFold


@app.cell
def _(HLAB, HPATCH, HY, Image, Path, np, torch):
    from anomalib.models import EfficientAd as _EAh
    from anomalib.engine import Engine as _Egh
    from anomalib.data.dataclasses.torch import ImageBatch as _IBh
    from torch.utils.data import DataLoader as _DLh
    from sklearn.model_selection import StratifiedKFold as _SKh
    _HF = list(_SKh(n_splits=5, shuffle=True, random_state=42).split(HLAB, HY))
    _good = [int(_i) for _i in range(len(HLAB)) if HY[_i] == 0]
    _gpos = {int(_ii): _k for _k, _ii in enumerate(_good)}
    _devH2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _B = HPATCH[_good].view(-1, 768).to(_devH2)
    _HLS = {}
    with torch.inference_mode():
        for _ii in range(len(HLAB)):
            _d = torch.cdist(HPATCH[_ii].to(_devH2), _B)
            if int(_ii) in _gpos:
                _r0 = _gpos[int(_ii)] * 256
                _d[:, _r0:_r0 + 256] = float("inf")
            _HLS[int(_ii)] = float(_d.min(dim=1).values.max().item())
            if (_ii + 1) % 100 == 0:
                print(f"loio {_ii + 1}/313", flush=True)
    _HLS = np.array([_HLS[_i] for _i in range(len(HLAB))])
    _HEM = {}
    for _fold, (_tr, _te) in enumerate(_HF):
        _tel = [int(_x) for _x in _te]
        _t = []
        for _ii in _tel:
            _r = HLAB.iloc[_ii]
            _im = Image.open(Path("transistor_norm") / ("good" if _r.label == "good" else "bad") / _r.filename).convert("RGB").resize((256, 256))
            _t.append(torch.from_numpy(np.asarray(_im, dtype=np.float32) / 255.0).permute(2, 0, 1))
        _dl = _DLh(_t, batch_size=16, collate_fn=lambda _b: _IBh(image=torch.stack(_b)))
        _em = _EAh(model_size="medium", visualizer=False, evaluator=False)
        _eng = _Egh(logger=False, default_root_dir=f"results_h_pred{_fold}")
        _ep = _eng.predict(model=_em, dataloaders=_dl, ckpt_path=f"results_h_em/EfficientAd/transistor/transistor/v{_fold}/weights/lightning/model.ckpt")
        _es = np.concatenate([_pb.pred_score.flatten().cpu().numpy() for _pb in _ep])
        for _ii, _sv in zip(_tel, _es):
            _HEM[int(_ii)] = float(_sv)
        print(f"em norm pred fold {_fold} done", flush=True)
    _HEM = np.array([_HEM[_i] for _i in range(len(HLAB))])
    HEM = _HEM
    HLS = _HLS
    print("H legs ok", flush=True)
    return HEM, HLS


@app.cell
def _(HCLS, HEM, HLAB, HLS, HY, np, pd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve as _rh, auc as _ah, precision_recall_fscore_support as _ph
    def _mets(_yt, _pr, _sc):
        _tp = int(((_yt == 1) & (_pr == 1)).sum())
        _fn = int(((_yt == 1) & (_pr == 0)).sum())
        _fp = int(((_yt == 0) & (_pr == 1)).sum())
        _tn = int(((_yt == 0) & (_pr == 0)).sum())
        _pc, _re, _f1, _ = _ph(_yt, _pr, average="binary", zero_division=0)
        _fpr, _tpr, _ = _rh(_yt, _sc)
        return {"bad_recall": round(float(_re), 3), "far": round(_fp / max(_fp + _tn, 1), 3), "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3), "f1": round(float(_f1), 3), "auroc": round(float(_ah(_fpr, _tpr)), 3)}
    hh_rows = []
    for _seed in [42, 7, 123, 2024, 999]:
        from sklearn.model_selection import StratifiedKFold as _SKh2
        for _fold, (_tr, _te) in enumerate(_SKh2(n_splits=5, shuffle=True, random_state=_seed).split(HLAB, HY)):
            _tr = np.array(_tr)
            _tel = [int(_x) for _x in _te]
            _gtr = _tr[HY[_tr] == 0]
            _lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(HCLS[_tr].numpy(), HY[_tr])
            _D = _lr.predict_proba(HCLS.numpy())[:, 1]
            _zn = {}
            for _nm, _V in [("d", _D), ("e", HEM), ("l", HLS)]:
                _md = np.median(_V[_gtr])
                _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
                _zn[_nm + "tr"], _zn[_nm + "te"] = (_V[_tr] - _md) / _iq, (_V[np.array(_te)] - _md) / _iq
            _gd = {k: v[HY[_tr] == 0] for k, v in [("d", _zn["dtr"]), ("e", _zn["etr"]), ("l", _zn["ltr"])]}
            _best, _ba = None, 0.1
            for _a in [round(_x * 0.05, 2) for _x in range(21)]:
                _f = _a * _zn["dtr"] + (1 - _a) * _zn["etr"]
                _ca = float(np.quantile(_a * _gd["d"] + (1 - _a) * _gd["e"], 0.95))
                _p = (_f >= _ca).astype(int)
                _rc = ((_p == 1) & (HY[_tr] == 1)).sum() / max((HY[_tr] == 1).sum(), 1)
                _fa = ((_p == 1) & (HY[_tr] == 0)).sum() / max((HY[_tr] == 0).sum(), 1)
                _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
                if _best is None or _key < _best[0]:
                    _best = (_key, _a, _ca)
            _, _ba, _bca = _best
            _fde = _ba * _zn["dte"] + (1 - _ba) * _zn["ete"]
            _pd = (_fde >= _bca).astype(int)
            _ql99 = float(np.quantile(_gd["l"], 0.99))
            _pr = ((_fde >= _bca) | (_zn["lte"] >= _ql99)).astype(int)
            _yte = HY[_te]
            hh_rows.append(dict(seed=_seed, fold=_fold, method="Dalpha", alpha=_ba, **_mets(_yte, _pd, _fde)))
            hh_rows.append(dict(seed=_seed, fold=_fold, method="rescue", alpha=_ba, **_mets(_yte, _pr, np.maximum(_fde, _zn["lte"]))))
            print("seed", _seed, "fold", _fold, "done", flush=True)
    hh_rows = pd.DataFrame(hh_rows)
    hh_rows.groupby("method")[["bad_recall", "far", "auroc"]].agg(["mean", "std", "min", "max"]).round(3)
    return LogisticRegression, hh_rows


@app.cell
def _(
    AutoImageProcessor,
    AutoModel,
    DataLoader,
    EfficientAd,
    Engine,
    HLAB,
    Image,
    ImageBatch,
    Path,
    np,
    torch,
):
    import time as _Tm
    from sklearn.linear_model import LogisticRegression as _LRT3
    _prL = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bmL = AutoModel.from_pretrained("facebook/dinov2-base")
    _bmL.eval()
    _devL = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bmL.to(_devL)
    _em0 = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
    _eng0 = Engine(logger=False, default_root_dir="results_h_lat")
    _probe = []
    for _ii in list(range(10)) + list(range(273, 283)):
        _r = HLAB.iloc[_ii]
        _im0 = Image.open(Path("transistor_norm") / ("good" if _r.label == "good" else "bad") / _r.filename).convert("RGB")
        _s = _Tm.time()
        _o = _bmL(**{k: v.to(_devL) for k, v in _prL(images=_im0, size={"height": 224, "width": 224}, return_tensors="pt").items()})
        _tE = torch.from_numpy(np.asarray(_im0.resize((256, 256)), dtype=np.float32) / 255.0).permute(2, 0, 1)
        _dl = DataLoader([_tE], batch_size=1, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
        _ep = _eng0.predict(model=_em0, dataloaders=_dl, ckpt_path="results_h_em/EfficientAd/transistor/transistor/v0/weights/lightning/model.ckpt")
        _probe.append(_Tm.time() - _s)
        print(f"lat {_ii} {_Tm.time() - _s:.1f}s", flush=True)
    ht_lat = {"ms_per_image": round(1000 * float(np.mean(_probe)), 0), "ms_tta5": round(5000 * float(np.mean(_probe)), 0)}
    print(ht_lat, flush=True)
    return (ht_lat,)


@app.cell
def _(HCLS, HEM, HLAB, HLS, HY, LogisticRegression, StratifiedKFold, np):
    _FF = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(HLAB, HY))
    ht_rule = {}
    for _fold, (_tr, _te) in enumerate(_FF):
        _tr = np.array(_tr)
        _gtr = _tr[HY[_tr] == 0]
        _lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(HCLS[_tr].numpy(), HY[_tr])
        _D = _lr.predict_proba(HCLS.numpy())[:, 1]
        _P = {}
        for _nm, _V in [("d", _D), ("e", HEM), ("l", HLS)]:
            _md = np.median(_V[_gtr])
            _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
            _P[_nm] = (_md, _iq)
        _dntr = (_D[_tr] - _P["d"][0]) / _P["d"][1]
        _detr = (HEM[_tr] - _P["e"][0]) / _P["e"][1]
        _gd = (HY[_tr] == 0)
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _dntr + (1 - _a) * _detr
            _ca = float(np.quantile(_f[_gd], 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (HY[_tr] == 1)).sum() / max((HY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (HY[_tr] == 0)).sum() / max((HY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        _ltr = (HLS[_tr] - _P["l"][0]) / _P["l"][1]
        _ql99 = float(np.quantile(_ltr[_gd], 0.99))
        ht_rule[_fold] = {"alpha": _ba, "bca": _bca, "P": _P, "q99": _ql99, "lr": _lr}
        print(f"rule fold {_fold} alpha={_ba} cut={_bca:.3f}", flush=True)
    print("rules ok")
    return


@app.cell
def _(
    AutoImageProcessor,
    AutoModel,
    DataLoader,
    EfficientAd,
    Engine,
    HCLS,
    HEM,
    HLAB,
    HLS,
    HPATCH,
    HY,
    Image,
    ImageBatch,
    LogisticRegression,
    Path,
    StratifiedKFold,
    np,
    pd,
    torch,
):
    def _shT(_im, _dx, _dy):
        _a = np.asarray(_im)
        _b = np.roll(_a, shift=(_dy, _dx), axis=(0, 1))
        if _dx > 0:
            _b[:, :_dx] = np.broadcast_to(_b[:, _dx:_dx + 1], (_b.shape[0], _dx, 3))
        elif _dx < 0:
            _b[:, _dx:] = np.broadcast_to(_b[:, _dx - 1:_dx], (_b.shape[0], -_dx, 3))
        if _dy > 0:
            _b[:_dy, :] = np.broadcast_to(_b[_dy:_dy + 1, :], (_dy, _b.shape[1], 3))
        elif _dy < 0:
            _b[_dy:, :] = np.broadcast_to(_b[_dy - 1:_dy, :], (-_dy, _b.shape[1], 3))
        return Image.fromarray(_b)
    _SH = [(0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)]
    _prT = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bmT = AutoModel.from_pretrained("facebook/dinov2-base")
    _bmT.eval()
    _devT2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bmT.to(_devT2)
    _FF = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(HLAB, HY))
    ht_rows, ht_b22 = [], {}
    for _fold, (_tr, _te) in enumerate(_FF):
        _tr = np.array(_tr)
        _tel = [int(_x) for _x in _te]
        _gtr = _tr[HY[_tr] == 0]
        _lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(HCLS[_tr].numpy(), HY[_tr])
        _D = _lr.predict_proba(HCLS.numpy())[:, 1]
        _P = {}
        for _nm, _V in [("d", _D), ("e", HEM), ("l", HLS)]:
            _md = np.median(_V[_gtr])
            _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
            _P[_nm] = (_md, _iq)
        _dntr = (_D[_tr] - _P["d"][0]) / _P["d"][1]
        _detr = (HEM[_tr] - _P["e"][0]) / _P["e"][1]
        _gd = (HY[_tr] == 0)
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _dntr + (1 - _a) * _detr
            _ca = float(np.quantile(_f[_gd], 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (HY[_tr] == 1)).sum() / max((HY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (HY[_tr] == 0)).sum() / max((HY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        _ltr = (HLS[_tr] - _P["l"][0]) / _P["l"][1]
        _ql99 = float(np.quantile(_ltr[_gd], 0.99))
        _bnk = HPATCH[[int(_i) for _i in _gtr]].view(-1, 768).to(_devT2)
        _em = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
        _eng = Engine(logger=False, default_root_dir=f"results_h_tta{_fold}")
        _ckpt = f"results_h_em/EfficientAd/transistor/transistor/v{_fold}/weights/lightning/model.ckpt"
        _pils, _emT, _own = [], [], []
        for _ii in _tel:
            _r = HLAB.iloc[_ii]
            _im0 = Image.open(Path("transistor_norm") / ("good" if _r.label == "good" else "bad") / _r.filename).convert("RGB")
            for (_dx, _dy) in _SH:
                _av = _shT(_im0, _dx, _dy)
                _pils.append(_av)
                _emT.append(torch.from_numpy(np.asarray(_av.resize((256, 256)), dtype=np.float32) / 255.0).permute(2, 0, 1))
                _own.append(_ii)
        _DS, _PS = [], []
        with torch.inference_mode():
            for _b0 in range(0, len(_pils), 16):
                _o = _bmT(_prT(images=_pils[_b0:_b0 + 16], size={"height": 224, "width": 224}, return_tensors="pt")["pixel_values"].to(_devT2))
                _h = _o.last_hidden_state
                _DS.extend(_lr.predict_proba(_h[:, 0, :].cpu().numpy())[:, 1].tolist())
                _pp = _h[:, 1:, :]
                for _k in range(_pp.shape[0]):
                    _dd = torch.cdist(_pp[_k].unsqueeze(0).to(_devT2), _bnk)
                    _PS.append(float(_dd.min(dim=2).values.max().item()))
        _dl = DataLoader(_emT, batch_size=16, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
        _ep = _eng.predict(model=_em, dataloaders=_dl, ckpt_path=_ckpt)
        _ES = np.concatenate([_pb.pred_score.flatten().cpu().numpy() for _pb in _ep]).tolist()
        _V = {"1d": [0, 0, 0, 0], "1r": [0, 0, 0, 0], "td": [0, 0, 0, 0], "tr": [0, 0, 0, 0]}
        for _j, _ii in enumerate(_tel):
            _vv = list(range(_j * 5, _j * 5 + 5))
            _dm = np.array([_DS[_k] for _k in _vv])
            _emv = np.array([_ES[_k] for _k in _vv])
            _lm = np.array([_PS[_k] for _k in _vv])
            _zd = (_dm - _P["d"][0]) / _P["d"][1]
            _ze = (_emv - _P["e"][0]) / _P["e"][1]
            _zl = (_lm - _P["l"][0]) / _P["l"][1]
            _fm = _ba * _zd + (1 - _ba) * _ze
            _mf, _ml = float(_fm.mean()), float(_zl.mean())
            _s1 = int(_fm[0] >= _bca)
            _sr = int((_fm[0] >= _bca) or (_zl[0] >= _ql99))
            _t1 = int(_mf >= _bca)
            _tr2 = int((_mf >= _bca) or (_ml >= _ql99))
            _yt = int(HY[_ii])
            if _yt == 1:
                _V["1d"][0] += _s1
                _V["1d"][1] += 1
                _V["1r"][0] += _sr
                _V["1r"][1] += 1
                _V["td"][0] += _t1
                _V["td"][1] += 1
                _V["tr"][0] += _tr2
                _V["tr"][1] += 1
            else:
                _V["1d"][2] += _s1
                _V["1d"][3] += 1
                _V["1r"][2] += _sr
                _V["1r"][3] += 1
                _V["td"][2] += _t1
                _V["td"][3] += 1
                _V["tr"][2] += _tr2
                _V["tr"][3] += 1
            if HLAB.iloc[_ii]["filename"] == "bad_022.png":
                ht_b22 = {"single_d": _s1, "single_r": _sr, "tta_d": _t1, "tta_r": _tr2}
        for _m, _k in [("1d", "single-D"), ("1r", "single-R"), ("td", "tta-D"), ("tr", "tta-R")]:
            _a = _V[_m]
            ht_rows.append({"fold": _fold, "method": _k, "bad_recall": round(_a[0] / max(_a[1], 1), 3), "far": round(_a[2] / max(_a[3], 1), 3)})
        print(f"tta fold {_fold} done", flush=True)
    ht_rows = pd.DataFrame(ht_rows)
    ht_rows.groupby("method")[["bad_recall", "far"]].agg(["mean", "std"]).round(3)
    return ht_b22, ht_rows


@app.cell
def _(hh_rows, ht_b22, ht_lat, ht_rows, pd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient as _MCh2
    from pathlib import Path as _Ph2
    _fd = _Ph2("mlfigs_faseH")
    _fd.mkdir(exist_ok=True)
    _cmp = pd.DataFrame([
        {"config": "G Dalpha", "recall": 0.940, "far": 0.060},
        {"config": "G rescue", "recall": 0.995, "far": 0.076},
        {"config": "H Dalpha", "recall": round(float(hh_rows[hh_rows.method == "Dalpha"].bad_recall.mean()), 3), "far": round(float(hh_rows[hh_rows.method == "Dalpha"].far.mean()), 3)},
        {"config": "H rescue", "recall": round(float(hh_rows[hh_rows.method == "rescue"].bad_recall.mean()), 3), "far": round(float(hh_rows[hh_rows.method == "rescue"].far.mean()), 3)},
        {"config": "H tta-R", "recall": round(float(ht_rows[ht_rows.method == "tta-R"].bad_recall.mean()), 3), "far": round(float(ht_rows[ht_rows.method == "tta-R"].far.mean()), 3)},
    ])
    _fx, _axx = plt.subplots(figsize=(8, 3))
    _xx = range(len(_cmp))
    _axx.bar([_x - 0.2 for _x in _xx], _cmp.recall, width=0.4, label="recall")
    _axx.bar([_x + 0.2 for _x in _xx], _cmp.far, width=0.4, label="FAR")
    _axx.set_xticks(list(_xx), list(_cmp.config), rotation=15, ha="right", fontsize=8)
    _axx.legend(fontsize=8)
    _axx.set_title("G (orig) vs H (norm) — G de libreta fase_g, resto medido aqui")
    _axx.set_ylim(0, 1.05)
    plt.tight_layout()
    _fx.savefig(_fd / "comparativa_gh.png", dpi=100)
    plt.close(_fx)
    hh_rows.to_csv("fase_h_repeated.csv", index=False)
    ht_rows.to_csv("fase_h_tta.csv", index=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    _rn = "faseH_geo"
    _hits = _MCh2().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
    _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
    with _cm:
        mlflow.log_params({"norm": "deadband25-edgepad", "corrected": 293, "passthrough": 20, "em": "retrained-5000-same-protocol",
            "tta": "shifts8-mean", "lat_ms": ht_lat["ms_per_image"], "lat_tta5_ms": ht_lat["ms_tta5"],
            "verdict": "freeze-G-pilot", "b22_tta_r": ht_b22["tta_r"], "b22_tta_d": ht_b22["tta_d"]})
        _hd = hh_rows[hh_rows.method == "Dalpha"]
        _hr = hh_rows[hh_rows.method == "rescue"]
        mlflow.log_metrics({"h_dalpha_recall_mean": round(float(_hd.bad_recall.mean()), 3), "h_dalpha_far_mean": round(float(_hd.far.mean()), 3),
            "h_rescue_recall_mean": round(float(_hr.bad_recall.mean()), 3), "h_rescue_far_mean": round(float(_hr.far.mean()), 3),
            "h_rescue_recall_std": round(float(_hr.bad_recall.std()), 3), "h_rescue_far_std": round(float(_hr.far.std()), 3)})
        _ht = ht_rows[ht_rows.method == "tta-R"]
        mlflow.log_metrics({"tta_recall_mean": round(float(_ht.bad_recall.mean()), 3), "tta_far_mean": round(float(_ht.far.mean()), 3)})
        mlflow.log_artifact("fase_h_repeated.csv")
        mlflow.log_artifact("fase_h_tta.csv")
        mlflow.log_artifact("mlfigs_faseH/comparativa_gh.png")
        print("mlflow faseH ok", flush=True)
    _cmp
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase H (geometria + piloto)" + chr(10) +
        "- Normalizacion deadband25: H-rescue 0.980/0.090 vs G-rescue 0.995/0.076. NO mejora: se congela G." + chr(10) +
        "- TTA-shifts: tta-R 1.00/0.095 pero tta-D pierde bad_022 (promediar diluye): el rescate local compensa." + chr(10) +
        "- Latencia 236ms/imagen (TTA-5 ~1.2s): viable para piloto en celda de inspeccion." + chr(10) +
        "- Piloto: G-rescue q99 con operating point 5.5 pct FAR + recoleccion de BAD reales; vigilar crops/zoom."
    )
    return


@app.cell
def _(Path, mo):
    _order = ["comparativa_gh.png"]
    _ims = [mo.image(str((Path("mlfigs_faseH") / _p).resolve()), width=640, caption=_p) for _p in _order if (Path("mlfigs_faseH") / _p).exists()]
    mo.vstack([mo.md(f"### Galeria H ({len(_ims)}/{len(_order)})"), mo.hstack(_ims)])
    return


if __name__ == "__main__":
    app.run()
