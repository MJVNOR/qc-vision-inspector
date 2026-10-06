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
    # Fase G — validacion de robustez (D-alpha + rescue congelados)
    Repeated CV 5x5 sobre la fusion + stress test solo en inferencia.
    Reporte honesto con n=40: "100% en el conjunto evaluado".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_g ready")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    GLAB = pd.read_csv(Path("data/processed/transistor_binary") / "labels.csv")
    GY = (GLAB["label"] == "bad").to_numpy(int)
    GSEEDS = [42, 7, 123, 2024, 999]
    _d2 = np.load("models/fase_a_oof_dinov2.npz")
    _d2m = dict(zip([str(_f) for _f in _d2["fnames"]], _d2["scores"]))
    GD2 = np.array([_d2m[_f] for _f in GLAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emm = {}
    for _f in range(5):
        for _i, _ss in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emm[int(_i)] = float(_ss)
    GEM = np.array([_emm[_i] for _i in range(len(GLAB))])
    _proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bm = AutoModel.from_pretrained("facebook/dinov2-base")
    _bm.eval()
    for _p in _bm.parameters():
        _p.requires_grad_(False)
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bm.to(_dev)
    _PL = []
    with torch.inference_mode():
        for _i in range(len(GLAB)):
            _r = GLAB.iloc[_i]
            _im = Image.open(Path("data/processed/transistor_binary") / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB")
            _inp = _proc(images=_im, size={"height": 224, "width": 224}, return_tensors="pt")
            _PL.append(_bm(**{k: v.to(_dev) for k, v in _inp.items()}).last_hidden_state[:, 1:, :].squeeze(0).cpu())
            if (_i + 1) % 100 == 0:
                print(f"fw {_i + 1}/313", flush=True)
    GPATCH = torch.stack(_PL)
    print(f"patches {tuple(GPATCH.shape)}")
    del _PL
    return (
        AutoImageProcessor,
        AutoModel,
        GD2,
        GEM,
        GLAB,
        GPATCH,
        GSEEDS,
        GY,
        Image,
        Path,
        StratifiedKFold,
        np,
        pd,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        "### Nota: sklearn 1-NN sustituido por torch cdist en GPU" + chr(10) +
        "- El kNN de sklearn solo usa CPU; torch.cdist por chunks da el mismo 1-NN exacto 10-50x mas rapido." + chr(10) +
        "- Ademas el scoring ahora es leave-one-image-out: cada imagen enmascara sus propias filas del bank." + chr(10) +
        "- La celda wzGT queda deshabilitada (calibraba con scores in-sample y colapsaba FAR a 1.0)."
    )
    return


@app.cell
def _(GD2, GEM, GLAB, GPATCH, GSEEDS, GY, StratifiedKFold, np, pd, torch):
    def _mets(_yt, _pr, _sc):
        from sklearn.metrics import roc_curve as _roc, auc as _au, precision_recall_fscore_support as _prf
        _tp = int(((_yt == 1) & (_pr == 1)).sum())
        _fn = int(((_yt == 1) & (_pr == 0)).sum())
        _fp = int(((_yt == 0) & (_pr == 1)).sum())
        _tn = int(((_yt == 0) & (_pr == 0)).sum())
        _pc, _re, _f1, _ = _prf(_yt, _pr, average="binary", zero_division=0)
        _fpr, _tpr, _ = _roc(_yt, _sc)
        return {"bad_recall": round(float(_re), 3), "far": round(_fp / max(_fp + _tn, 1), 3), "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3), "f1": round(float(_f1), 3), "auroc": round(float(_au(_fpr, _tpr)), 3)}
    _good = [int(_i) for _i in range(len(GLAB)) if GY[_i] == 0]
    _gpos = {int(_ii): _k for _k, _ii in enumerate(_good)}
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _B = GPATCH[_good].view(-1, 768).to(_dev)
    _LS = {}
    with torch.inference_mode():
        for _ii in range(len(GLAB)):
            _d = torch.cdist(GPATCH[_ii].to(_dev), _B)
            if int(_ii) in _gpos:
                _r0 = _gpos[int(_ii)] * 256
                _d[:, _r0:_r0 + 256] = float("inf")
            _LS[int(_ii)] = float(_d.min(dim=1).values.max().item())
            if (_ii + 1) % 100 == 0:
                print(f"loio {_ii + 1}/313", flush=True)
    _LSA = np.array([_LS[_i] for _i in range(len(GLAB))])
    gg_rows = []
    for _seed in GSEEDS:
        _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=_seed)
        for _fold, (_tr, _te) in enumerate(_skf.split(GLAB, GY)):
            _tr = np.array(_tr)
            _tel = [int(_x) for _x in _te]
            _gtr = _tr[GY[_tr] == 0]
            _zn = {}
            for _nm, _V in [("d", GD2), ("e", GEM), ("l", _LSA)]:
                _md = np.median(_V[_gtr])
                _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
                _zn[_nm + "tr"], _zn[_nm + "te"] = (_V[_tr] - _md) / _iq, (_V[np.array(_te)] - _md) / _iq
            _gd = {k: v[GY[_tr] == 0] for k, v in [("d", _zn["dtr"]), ("e", _zn["etr"]), ("l", _zn["ltr"])]}
            _best, _ba = None, 0.1
            for _a in [round(_x * 0.05, 2) for _x in range(21)]:
                _f = _a * _zn["dtr"] + (1 - _a) * _zn["etr"]
                _ca = float(np.quantile(_a * _gd["d"] + (1 - _a) * _gd["e"], 0.95))
                _p = (_f >= _ca).astype(int)
                _re2 = ((_p == 1) & (GY[_tr] == 1)).sum() / max((GY[_tr] == 1).sum(), 1)
                _fa = ((_p == 1) & (GY[_tr] == 0)).sum() / max((GY[_tr] == 0).sum(), 1)
                _key = (0 if _fa <= 0.05 else 1, -_re2, _fa)
                if _best is None or _key < _best[0]:
                    _best = (_key, _a, _ca)
            _, _ba, _bca = _best
            _fde = _ba * _zn["dte"] + (1 - _ba) * _zn["ete"]
            _pd = (_fde >= _bca).astype(int)
            _ql99 = float(np.quantile(_gd["l"], 0.99))
            _pr = ((_fde >= _bca) | (_zn["lte"] >= _ql99)).astype(int)
            _yte = GY[_te]
            gg_rows.append(dict(seed=_seed, fold=_fold, method="Dalpha", alpha=_ba, **_mets(_yte, _pd, _fde)))
            gg_rows.append(dict(seed=_seed, fold=_fold, method="rescue", alpha=_ba, **_mets(_yte, _pr, np.maximum(_fde, _zn["lte"]))))
            print("seed", _seed, "fold", _fold, "done", flush=True)
    gg_rows = pd.DataFrame(gg_rows)
    gg_rows.groupby("method")[["bad_recall", "far", "auroc"]].agg(["mean", "std", "min", "max"]).round(3)
    return (gg_rows,)


@app.cell
def _(GLAB, GY, np):
    from PIL import Image as _PIs, ImageEnhance as _IEs
    from anomalib.models import EfficientAd as _EAd
    from anomalib.engine import Engine as _Eng
    from anomalib.data.dataclasses.torch import ImageBatch as _IB
    from torch.utils.data import DataLoader as _DL
    from sklearn.linear_model import LogisticRegression as _LRg2
    import torch.nn.functional as _F2
    _s2 = __import__("torch").load("models/fase_a_emb_dinov2.pt", weights_only=False)
    _f2 = [str(_x) for _x in _s2["fnames"]]
    _yF = (GLAB.set_index("filename").loc[_f2]["label"] == "bad").to_numpy(int)
    _lrG = _LRg2(class_weight="balanced", C=1.0, max_iter=2000).fit(_s2["X"], _yF)
    _rng = np.random.default_rng(7)
    _SB = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 1], 40, replace=False).tolist())
    _SG = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 0], 40, replace=False).tolist())
    _SUB = _SB + _SG
    print("subset ok", len(_SUB))
    return


@app.cell
def _(AutoImageProcessor, AutoModel, GLAB, torch):
    from PIL import Image as _PIs2
    from PIL import ImageEnhance as _IEs2
    from torch.utils.data import DataLoader
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine
    from anomalib.data.dataclasses.torch import ImageBatch
    from sklearn.linear_model import LogisticRegression as _LRs
    def _aug(_im, _v):
        _w, _h = _im.size
        if _v == "clean":
            return _im
        if _v.startswith("bright"):
            return _IEs2.Brightness(_im).enhance(float(_v[6:]) / 100)
        if _v.startswith("shift"):
            _d = int(_v[5:])
            return _im.transform((_w, _h), _PIs2.AFFINE, (1, 0, _d, 0, 1, _d // 2))
        if _v.startswith("zoomout"):
            _s = float(_v[7:]) / 100
            _sm = _im.resize((int(_w * _s), int(_h * _s)))
            _cv = _PIs2.new("RGB", (_w, _h), (0, 0, 0))
            _cv.paste(_sm, ((_w - _sm.size[0]) // 2, (_h - _sm.size[1]) // 2))
            return _cv
        _s = float(_v[6:]) / 100
        _l, _t = int(_w * (1 - _s) / 2), int(_h * (1 - _s) / 2)
        return _im.crop((_l, _t, _l + int(_w * _s), _t + int(_h * _s))).resize((_w, _h))
    _VARS = ["clean", "bright085", "bright070", "shift10", "shift20", "zoomout090", "zoomout080", "cropin090", "cropin080"]
    _s2 = torch.load("models/fase_a_emb_dinov2.pt", weights_only=False)
    _lrS = _LRs(class_weight="balanced", C=1.0, max_iter=2000).fit(_s2["X"], (GLAB.set_index("filename").loc[[str(_f) for _f in _s2["fnames"]]]["label"] == "bad").to_numpy(int))
    _prS = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bmS = AutoModel.from_pretrained("facebook/dinov2-base")
    _bmS.eval()
    for _p in _bmS.parameters():
        _p.requires_grad_(False)
    _devS = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bmS.to(_devS)
    print("stress setup ok", flush=True)
    return DataLoader, EfficientAd, Engine, ImageBatch


@app.cell
def _(
    AutoImageProcessor,
    AutoModel,
    DataLoader,
    EfficientAd,
    Engine,
    GLAB,
    GPATCH,
    GY,
    Image,
    ImageBatch,
    Path,
    StratifiedKFold,
    np,
    pd,
    torch,
):
    from PIL import ImageEnhance as _IEg
    from sklearn.linear_model import LogisticRegression as _LRT2
    _VARS = ["clean", "bright085", "bright070", "shift10", "shift20", "zoomout090", "zoomout080", "cropin090", "cropin080"]
    def _aug2(_im, _v):
        _w, _h = _im.size
        if _v == "clean":
            return _im
        if _v.startswith("bright"):
            return _IEg.Brightness(_im).enhance(float(_v[6:]) / 100)
        if _v.startswith("shift"):
            _d = int(_v[5:])
            return _im.transform((_w, _h), Image.AFFINE, (1, 0, _d, 0, 1, _d // 2))
        if _v.startswith("zoomout"):
            _s = float(_v[7:]) / 100
            _sm = _im.resize((int(_w * _s), int(_h * _s)))
            _cv = Image.new("RGB", (_w, _h), (0, 0, 0))
            _cv.paste(_sm, ((_w - _sm.size[0]) // 2, (_h - _sm.size[1]) // 2))
            return _cv
        _s = float(_v[6:]) / 100
        _l, _t = int(_w * (1 - _s) / 2), int(_h * (1 - _s) / 2)
        return _im.crop((_l, _t, _l + int(_w * _s), _t + int(_h * _s))).resize((_w, _h))
    _sT = torch.load("models/fase_a_emb_dinov2.pt", weights_only=False)
    _lrT = _LRT2(class_weight="balanced", C=1.0, max_iter=2000).fit(_sT["X"], (GLAB.set_index("filename").loc[[str(_f) for _f in _sT["fnames"]]]["label"] == "bad").to_numpy(int))
    _d2 = np.load("models/fase_a_oof_dinov2.npz")
    _d2m = dict(zip([str(_f) for _f in _d2["fnames"]], _d2["scores"]))
    _D2 = np.array([_d2m[_f] for _f in GLAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emm = {}
    for _f in range(5):
        for _i, _ss in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emm[int(_i)] = float(_ss)
    _EM = np.array([_emm[_i] for _i in range(len(GLAB))])
    _good = [int(_i) for _i in range(len(GLAB)) if GY[_i] == 0]
    _gpos = {int(_ii): _k for _k, _ii in enumerate(_good)}
    _devT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _B = GPATCH[_good].view(-1, 768).to(_devT)
    _LSA = {}
    with torch.inference_mode():
        for _ii in range(len(GLAB)):
            _d = torch.cdist(GPATCH[_ii].to(_devT), _B)
            if int(_ii) in _gpos:
                _r0 = _gpos[int(_ii)] * 256
                _d[:, _r0:_r0 + 256] = float("inf")
            _LSA[int(_ii)] = float(_d.min(dim=1).values.max().item())
    _LSA = np.array([_LSA[_i] for _i in range(len(GLAB))])
    print("loio ok", flush=True)
    _FF = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(GLAB, GY))
    _rng = np.random.default_rng(7)
    _SB = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 1], 40, replace=False).tolist())
    _SG = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 0], 40, replace=False).tolist())
    _prC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _bmC = AutoModel.from_pretrained("facebook/dinov2-base")
    _bmC.eval()
    for _p in _bmC.parameters():
        _p.requires_grad_(False)
    _bmC.to(_devT)
    gs_drift, gs_b22 = [], {}
    for _fold, (_tr, _te) in enumerate(_FF):
        _tr = np.array(_tr)
        _gtr = _tr[GY[_tr] == 0]
        _P = {}
        for _nm, _V in [("d", _D2), ("e", _EM), ("l", _LSA)]:
            _md = np.median(_V[_gtr])
            _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
            _P[_nm] = (_md, _iq)
        _dntr = (_D2[_tr] - _P["d"][0]) / _P["d"][1]
        _detr = (_EM[_tr] - _P["e"][0]) / _P["e"][1]
        _gd = (GY[_tr] == 0)
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _dntr + (1 - _a) * _detr
            _ca = float(np.quantile(_f[_gd], 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (GY[_tr] == 1)).sum() / max((GY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (GY[_tr] == 0)).sum() / max((GY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        _ltr = (_LSA[_tr] - _P["l"][0]) / _P["l"][1]
        _ql99 = float(np.quantile(_ltr[_gd], 0.99))
        _bnk = _B
        _em = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
        _eng = Engine(logger=False, default_root_dir=f"results_g_em{_fold}")
        _ckpt = f"results_c_em/EfficientAd/transistor/transistor/v{_fold + 5}/weights/lightning/model.ckpt"
        _sub = [int(_i) for _i in _SB + _SG if int(_i) in set(int(_x) for _x in _te)]
        _pils, _emT, _keys = [], [], []
        for _ii in _sub:
            _r = GLAB.iloc[_ii]
            _im0 = Image.open(Path("data/processed/transistor_binary") / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB")
            for _v in _VARS:
                _av = _aug2(_im0, _v)
                _pils.append(_av)
                _emT.append(torch.from_numpy(np.asarray(_av.resize((256, 256)), dtype=np.float32) / 255.0).permute(2, 0, 1))
                _keys.append((_ii, _v))
        _ds, _ps = [], []
        with torch.inference_mode():
            for _b0 in range(0, len(_pils), 16):
                _o = _bmC(_prC(images=_pils[_b0:_b0 + 16], size={"height": 224, "width": 224}, return_tensors="pt")["pixel_values"].to(_devT))
                _h = _o.last_hidden_state
                _ds.extend(_lrT.predict_proba(_h[:, 0, :].cpu().numpy())[:, 1].tolist())
                _pp = _h[:, 1:, :]
                for _k in range(_pp.shape[0]):
                    _gi = _keys[_b0 + _k][0]
                    _dd = torch.cdist(_pp[_k].unsqueeze(0).to(_devT), _bnk)
                    _ddm = _dd.clone()
                    if _gi in _gpos:
                        _r0 = _gpos[_gi] * 256
                        _ddm[:, _r0:_r0 + 256] = float("inf")
                    _ps.append(float(_ddm.min(dim=2).values.max().item()))
        for _kk, (_ii, _vv) in enumerate(_keys):
            if _vv == "clean":
                _ds[_kk] = float(_D2[_ii])
        _dl = DataLoader(_emT, batch_size=16, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
        _ep = _eng.predict(model=_em, dataloaders=_dl, ckpt_path=_ckpt)
        _es = np.concatenate([_pb.pred_score.flatten().cpu().numpy() for _pb in _ep])
        _A = {v: {"dtp": 0, "dfn": 0, "dfp": 0, "dtn": 0, "rtp": 0, "rfn": 0, "rfp": 0, "rtn": 0} for v in _VARS}
        for (_ii, _v), _sd, _se, _sl in zip(_keys, _ds, _es.tolist(), _ps):
            _zd = (_sd - _P["d"][0]) / _P["d"][1]
            _ze = (_se - _P["e"][0]) / _P["e"][1]
            _zl = (_sl - _P["l"][0]) / _P["l"][1]
            _f = _ba * _zd + (1 - _ba) * _ze
            _pdv = int(_f >= _bca)
            _prv = int((_f >= _bca) or (_zl >= _ql99))
            _yt = int(GY[_ii])
            _C = _A[_v]
            if _yt == 1:
                _C["dtp"] += _pdv
                _C["dfn"] += 1 - _pdv
                _C["rtp"] += _prv
                _C["rfn"] += 1 - _prv
            else:
                _C["dfp"] += _pdv
                _C["dtn"] += 1 - _pdv
                _C["rfp"] += _prv
                _C["rtn"] += 1 - _prv
            if GLAB.iloc[_ii]["filename"] == "bad_022.png":
                gs_b22[_v] = {"d": _pdv, "r": _prv}
        for _v in _VARS:
            _C = _A[_v]
            gs_drift.append({"fold": _fold, "variant": _v, "d_recall": round(_C["dtp"] / max(_C["dtp"] + _C["dfn"], 1), 3), "r_recall": round(_C["rtp"] / max(_C["rtp"] + _C["rfn"], 1), 3), "d_far": round(_C["dfp"] / max(_C["dfp"] + _C["dtn"], 1), 3), "r_far": round(_C["rfp"] / max(_C["rfp"] + _C["rtn"], 1), 3)})
        print(f"stress fold {_fold} done", flush=True)
    gs_drift = pd.DataFrame(gs_drift)
    gs_drift.groupby("variant")[["d_recall", "r_recall", "d_far", "r_far"]].mean().round(3)
    return gs_b22, gs_drift


@app.cell
def _(gg_rows, gs_b22, gs_drift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient as _MCh
    from pathlib import Path as _Ph
    from scipy.stats import beta as _beta
    _cp_low = round(float(_beta.ppf(0.025, 40, 1)), 3)
    _fd = _Ph("reports/figures/mlfigs_faseG")
    _fd.mkdir(exist_ok=True)
    _gr = gg_rows[gg_rows.method == "rescue"]
    _fg, _ax = plt.subplots(1, 2, figsize=(10, 4))
    _ax[0].hist(_gr.bad_recall, bins=5, alpha=0.7)
    _ax[0].axvline(1.0, color="red", lw=2)
    _ax[0].set_title("rescue recall 25 splits")
    _ax[1].hist(_gr.far, bins=10, alpha=0.7)
    _ax[1].axvline(0.05, color="red", lw=1, ls="--")
    _ax[1].set_title("rescue FAR 25 splits")
    _fg.savefig(_fd / "distrib.png", dpi=100)
    plt.close(_fg)
    _fs, _as = plt.subplots()
    _o = ["clean", "bright085", "bright070", "shift10", "shift20", "zoomout090", "zoomout080", "cropin090", "cropin080"]
    _m = gs_drift.groupby("variant")[["d_recall", "r_recall", "d_far", "r_far"]].mean().reindex(_o)
    _as.plot(_o, _m.r_recall, "o-", label="rescue recall")
    _as.plot(_o, _m.r_far, "s-", label="rescue FAR")
    _as.plot(_o, _m.d_far, "^-", label="dalpha FAR")
    _as.set_xticks(range(len(_o)), _o, rotation=30, ha="right", fontsize=7)
    _as.legend(fontsize=8)
    _as.set_title("stress: variantes (media 5 folds, subset 80)")
    _fs.savefig(_fd / "stress.png", dpi=100)
    plt.close(_fs)
    gg_rows.to_csv("reports/fase_g_repeated.csv", index=False)
    gs_drift.to_csv("reports/fase_g_stress.csv", index=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    _rn = "faseG_robustez"
    _hits = _MCh().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
    _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
    with _cm:
        mlflow.log_params({"design": "5seedsx5folds-fusion-only", "bases": "frozen", "local": "torch-cdist-LOIO-224max",
            "stress": "subset40+40-8vars-mild+med", "em_ckpt": "v5-9-5000steps", "dino_leg": "fulldata-LR-caveat",
            "honest": "100pct-en-conjunto-evaluado", "clopper40_40_low95": _cp_low})
        mlflow.log_metrics({"rescue_recall_mean": round(float(_gr.bad_recall.mean()), 3), "rescue_recall_std": round(float(_gr.bad_recall.std()), 3),
            "rescue_recall_min": round(float(_gr.bad_recall.min()), 3), "rescue_n100": int((_gr.bad_recall == 1.0).sum()),
            "rescue_far_mean": round(float(_gr.far.mean()), 3), "rescue_far_std": round(float(_gr.far.std()), 3),
            "dalpha_recall_mean": round(float(gg_rows[gg_rows.method == "Dalpha"].bad_recall.mean()), 3),
            "dalpha_far_mean": round(float(gg_rows[gg_rows.method == "Dalpha"].far.mean()), 3)})
        mlflow.log_metrics({f"b22_{_k}_{_kk}": float(_vv) for _k, _dd in gs_b22.items() for _kk, _vv in _dd.items()})
        mlflow.log_artifact("reports/fase_g_repeated.csv")
        mlflow.log_artifact("reports/fase_g_stress.csv")
        mlflow.log_artifact("reports/figures/mlfigs_faseG/distrib.png")
        mlflow.log_artifact("reports/figures/mlfigs_faseG/stress.png")
        print(f"mlflow faseG ok cp_low={_cp_low}", flush=True)
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase G (robustez, bases congeladas)" + chr(10) +
        "- Repeated 5x5: rescue recall 0.995+-0.025 (24/25 en 1.00), FAR 0.076+-0.040. Estable." + chr(10) +
        "- Stress: bad_022 capturada en las 9 variantes. Mild (bright085/shift10) tolerable; crop/zoom rompen ambas ramas (FAR->1.0, artefacto de bordes negros incluido)." + chr(10) +
        "- Lenguaje honesto: 100 pct recall en el conjunto evaluado; IC95 inferior Clopper-Pearson con 40/40 = 0.912." + chr(10) +
        "- Veredicto: candidato a piloto con operating point actual; vigilar invariancia geometrica en produccion."
    )
    return


@app.cell
def _(Path, mo):
    _order = ["distrib.png", "stress.png"]
    _ims = [mo.image(str((Path("reports/figures/mlfigs_faseG") / _p).resolve()), width=560, caption=_p) for _p in _order if (Path("reports/figures/mlfigs_faseG") / _p).exists()]
    mo.vstack([mo.md(f"### Galeria G ({len(_ims)}/{len(_order)})"), mo.hstack(_ims)])
    return


if __name__ == "__main__":
    app.run()
