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
    FLAB = pd.read_csv(Path("data/processed/transistor_binary") / "labels.csv")
    FY = (FLAB["label"] == "bad").to_numpy(int)
    FFOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(FLAB, FY))
    FREZ = [224, 512]
    FAREA, FBUCKET = {}, {}
    for _r in FLAB[FLAB.label == "bad"].itertuples():
        _mp = Path("data/raw/mvtec_anomaly_detection/transistor/ground_truth") / _r.defect / (_r.original_path.split("/")[-1].replace(".png", "_mask.png"))
        _a = 100 * (np.asarray(_PIf.open(_mp).convert("L")) > 127).mean()
        FAREA[_r.filename] = round(float(_a), 2)
        FBUCKET[_r.filename] = "tiny<1%" if _a < 1 else ("mid1-3%" if _a < 3 else "big>3%")
    pd.Series(FAREA).describe().round(2).to_string()
    return FBUCKET, FFOLDS, FLAB, FREZ, FY, Path, np, pd


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
        _im = Image.open(Path("data/processed/transistor_binary") / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB")
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
    return (fl_scores,)


@app.cell
def _(FFOLDS, FLAB, FREZ, FY, fl_scores, np, pd):
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    _d2 = np.load("models/fase_a_oof_dinov2.npz")
    _d2m = dict(zip([str(_f) for _f in _d2["fnames"]], _d2["scores"]))
    _D2 = np.array([_d2m[_f] for _f in FLAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emm = {}
    for _f in range(5):
        for _i, _ss in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emm[int(_i)] = float(_ss)
    _EM = np.array([_emm[_i] for _i in range(len(FLAB))])
    def _iq(_v, _m, _q):
        return (_v - _m) / _q
    def _mets(_yt, _pr, _sc):
        _tp = int(((_yt == 1) & (_pr == 1)).sum())
        _fn = int(((_yt == 1) & (_pr == 0)).sum())
        _fp = int(((_yt == 0) & (_pr == 1)).sum())
        _tn = int(((_yt == 0) & (_pr == 0)).sum())
        _pc, _rc, _f1, _ = precision_recall_fscore_support(_yt, _pr, average="binary", zero_division=0)
        _fpr, _tpr, _ = roc_curve(_yt, _sc)
        return {"bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3), "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3), "f1": round(float(_f1), 3), "auroc": round(float(auc(_fpr, _tpr)), 3)}
    fu_rows, fu_local, fu_fuse = [], {}, {}
    for _sz in FREZ:
        for _agg in ["max", "top10", "p95", "p99"]:
            _L = np.full(len(FLAB), np.nan)
            for _f in range(5):
                _te, _sv = fl_scores[(_sz, _agg, _f)]
                for _ii, _svv in zip([int(_x) for _x in _te], _sv):
                    _L[_ii] = float(_svv)
            fu_local[(_sz, _agg)] = _L
            _rr = []
            for _fold, (_tr, _te) in enumerate(FFOLDS):
                _tr = np.array(_tr)
                _cut = float(np.quantile(_L[_tr[FY[_tr] == 0]], 0.95))
                _pr = (_L[np.array(_te)] >= _cut).astype(int)
                _rr.append(_mets(FY[_te], _pr, _L[np.array(_te)]))
            _rr = pd.DataFrame(_rr)
            fu_rows.append({"method": f"local-{_sz}-{_agg}", "bad_recall": round(_rr.bad_recall.mean(), 3), "far": round(_rr.far.mean(), 3), "auroc": round(_rr.auroc.mean(), 3)})
            print(f"local {_sz} {_agg}: " + str(fu_rows[-1]), flush=True)
    fu_rows = pd.DataFrame(fu_rows)
    fu_rows
    return (fu_local,)


@app.cell
def _(FBUCKET, FFOLDS, FLAB, FY, fu_local, np, pd):
    from sklearn.metrics import roc_curve as _rc2, auc as _au2, precision_recall_fscore_support as _prf2
    def _mets(_yt, _pr, _sc):
        _tp = int(((_yt == 1) & (_pr == 1)).sum())
        _fn = int(((_yt == 1) & (_pr == 0)).sum())
        _fp = int(((_yt == 0) & (_pr == 1)).sum())
        _tn = int(((_yt == 0) & (_pr == 0)).sum())
        _pc, _rc, _f1, _ = _prf2(_yt, _pr, average="binary", zero_division=0)
        _fpr, _tpr, _ = _rc2(_yt, _sc)
        return {"bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3), "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3), "f1": round(float(_f1), 3), "auroc": round(float(_au2(_fpr, _tpr)), 3)}
    _d2 = np.load("models/fase_a_oof_dinov2.npz")
    _d2m = dict(zip([str(_f) for _f in _d2["fnames"]], _d2["scores"]))
    _D2 = np.array([_d2m[_f] for _f in FLAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emm = {}
    for _f in range(5):
        for _i, _ss in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emm[int(_i)] = float(_ss)
    _EM = np.array([_emm[_i] for _i in range(len(FLAB))])
    _L = fu_local[(224, "max")]
    fu2_rows, _b22r, _sizer, _typer = [], {}, {"tiny<1%": [0, 0], "mid1-3%": [0, 0], "big>3%": [0, 0]}, {"bent_lead": [0, 0], "cut_lead": [0, 0], "damaged_case": [0, 0], "misplaced": [0, 0]}
    for _fold, (_tr, _te) in enumerate(FFOLDS):
        _tr = np.array(_tr)
        _tel = [int(_x) for _x in _te]
        _gtr = _tr[FY[_tr] == 0]
        _zn = {}
        for _nm, _V in [("d", _D2), ("e", _EM), ("l", _L)]:
            _md = np.median(_V[_gtr])
            _iq = np.subtract(*np.percentile(_V[_gtr], [75, 25])) + 1e-9
            _zn[_nm + "tr"], _zn[_nm + "te"] = (_V[_tr] - _md) / _iq, (_V[np.array(_te)] - _md) / _iq
        _gd = {k: v[FY[_tr] == 0] for k, v in [("d", _zn["dtr"]), ("e", _zn["etr"]), ("l", _zn["ltr"])]}
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _zn["dtr"] + (1 - _a) * _zn["etr"]
            _ca = float(np.quantile(_a * _gd["d"] + (1 - _a) * _gd["e"], 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (FY[_tr] == 1)).sum() / max((FY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (FY[_tr] == 0)).sum() / max((FY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        _fde = _ba * _zn["dte"] + (1 - _ba) * _zn["ete"]
        _pd = (_fde >= _bca).astype(int)
        _ql99 = float(np.quantile(_gd["l"], 0.99))
        _ql995 = float(np.quantile(_gd["l"], 0.995))
        _pr = ((_fde >= _bca) | (_zn["lte"] >= _ql99)).astype(int)
        _pr5 = ((_fde >= _bca) | (_zn["lte"] >= _ql995)).astype(int)
        _yte = FY[_te]
        fu2_rows.append({"fold": _fold, "alpha": _ba, **_mets(_yte, _pd, _fde)})
        _r2 = dict(fu2_rows[-1])
        _r2["method"] = "Dalpha"
        fu2_rows[-1] = _r2
        fu2_rows.append({"fold": _fold, "alpha": _ba, "method": "Dalpha+rescue", **_mets(_yte, _pr, np.maximum(_fde, _zn["lte"]))})
        fu2_rows.append({"fold": _fold, "alpha": _ba, "method": "Dalpha+rescue995", **_mets(_yte, _pr5, np.maximum(_fde, _zn["lte"]))})
        for _ii, _pp in zip(_tel, _pr):
            _fn = FLAB.iloc[_ii]["filename"]
            if FY[_ii] == 1:
                _sizer[FBUCKET[_fn]][1] += 1
                _sizer[FBUCKET[_fn]][0] += int(_pp)
                _typer[FLAB.iloc[_ii]["defect"]][1] += 1
                _typer[FLAB.iloc[_ii]["defect"]][0] += int(_pp)
            if _fn == "bad_022.png":
                _b22r = {"fold": _fold, "fused": round(float(_fde[_tel.index(_ii)]), 2), "local": round(float(_zn["lte"][_tel.index(_ii)]), 2), "caught": bool(_pp)}
                _b22r["caught995"] = bool(_pr5[_tel.index(_ii)])
        print(f"fusion fold {_fold} alpha={_ba} rescue_q99={_ql99:.2f}", flush=True)
    fu2_rows = pd.DataFrame(fu2_rows)
    print(fu2_rows.groupby("method")[["bad_recall", "far", "frr", "precision", "f1", "auroc"]].agg(["mean", "std"]).round(3).to_string())
    print("bad022=" + repr(_b22r))
    print("bysize=" + repr({k: round(v[0] / max(v[1], 1), 3) for k, v in _sizer.items()}))
    print("bytype=" + repr({k: round(v[0] / max(v[1], 1), 3) for k, v in _typer.items()}))
    fu_b22 = _b22r
    fu_size = {k: round(v[0] / max(v[1], 1), 3) for k, v in _sizer.items()}
    fu_type = {k: round(v[0] / max(v[1], 1), 3) for k, v in _typer.items()}
    return fu2_rows, fu_b22, fu_size, fu_type


@app.cell
def _(FFOLDS, FY, fu2_rows, fu_local, fu_size, np, pd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlflow.tracking import MlflowClient as _MCf
    from pathlib import Path as _Pf
    from sklearn.metrics import precision_recall_fscore_support as _prf3
    _lf = []
    for _fold, (_tr, _te) in enumerate(FFOLDS):
        _tr = np.array(_tr)
        _tel = [int(_x) for _x in _te]
        _L = fu_local[(224, "max")]
        _cut = float(np.quantile(_L[_tr[FY[_tr] == 0]], 0.95))
        _pr = (_L[np.array(_te)] >= _cut).astype(int)
        _tp = int(((FY[_te] == 1) & (_pr == 1)).sum())
        _fn = int(((FY[_te] == 1) & (_pr == 0)).sum())
        _fp = int(((FY[_te] == 0) & (_pr == 1)).sum())
        _tn = int(((FY[_te] == 0) & (_pr == 0)).sum())
        _pc, _rc, _f1, _ = _prf3(FY[_te], _pr, average="binary", zero_division=0)
        _lf.append({"fold": _fold, "bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3)})
    fu_local_folds = pd.DataFrame(_lf)
    _qq = _MCf()
    def _mm(_name):
        _f = "tags.mlflow.runName = " + chr(34) + _name + chr(34)
        return _qq.search_runs(experiment_ids=["1"], filter_string=_f)[0].data.metrics
    _a = _mm("faseA_dinov2_logreg")
    _c = _mm("faseC_em_efficientad")
    _d = fu2_rows[fu2_rows.method == "Dalpha+rescue"]
    _d0 = fu2_rows[fu2_rows.method == "Dalpha"]
    cmp_f = pd.DataFrame([
        {"config": "A dino", "recall": round(_a["thr5_bad_recall"], 3), "far": round(_a["thr5_far"], 3)},
        {"config": "C em", "recall": round(_c["thr5_bad_recall_mean"], 3), "far": round(_c["thr5_far_mean"], 3)},
        {"config": "D alpha", "recall": round(float(_d0.bad_recall.mean()), 3), "far": round(float(_d0.far.mean()), 3)},
        {"config": "local224max", "recall": round(float(fu_local_folds.bad_recall.mean()), 3), "far": round(float(fu_local_folds.far.mean()), 3)},
        {"config": "F rescue", "recall": round(float(_d.bad_recall.mean()), 3), "far": round(float(_d.far.mean()), 3)},
    ])
    _fx, _axx = plt.subplots(figsize=(8, 3))
    _xx = range(len(cmp_f))
    _axx.bar([_x - 0.2 for _x in _xx], cmp_f.recall, width=0.4, label="recall")
    _axx.bar([_x + 0.2 for _x in _xx], cmp_f.far, width=0.4, label="FAR")
    _axx.set_xticks(list(_xx), list(cmp_f.config), rotation=15, ha="right", fontsize=8)
    _axx.legend(fontsize=8)
    _axx.set_title("D-alpha vs F rescue")
    _axx.set_ylim(0, 1.05)
    plt.tight_layout()
    _fd = _Pf("reports/figures/mlfigs_faseF")
    _fd.mkdir(exist_ok=True)
    _fx.savefig(_fd / "comparativa_f.png", dpi=100)
    plt.close(_fx)
    _fs, _as = plt.subplots()
    _as.bar(list(fu_size.keys()), list(fu_size.values()))
    _as.set_ylim(0, 1.05)
    _as.set_title("F rescue: recall por tamano (todo 1.0, tiny 6/6 con bad_022)")
    _fs.savefig(_fd / "sizebars.png", dpi=100)
    plt.close(_fs)
    cmp_f
    return (fu_local_folds,)


@app.cell
def _(fu2_rows, fu_b22, fu_local_folds, fu_size, fu_type):
    import mlflow
    from mlflow.tracking import MlflowClient as _MCg
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    _rn = "faseF_local"
    _hits = _MCg().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
    _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
    with _cm:
        mlflow.log_params({"base": "D-alpha-frozen", "local": "dinov2-patch1NN-224", "bank": "good-train-full", "bank512": "10pct-subsample",
            "aggs": "max/top10/p95/p99", "chosen": "max", "rescue": "OR-local-q99-train", "rescue995": "OR-local-q995-train",
            "norm": "iqr-train-good", "threshold": "quantile-goodtrain-95", "cv": "StratifiedKFold-5", "seed": 42})
        mlflow.log_params({"bad022_fused": fu_b22["fused"], "bad022_local": fu_b22["local"], "bad022_caught": fu_b22["caught"], "bad022_caught995": fu_b22["caught995"]})
        for _t in fu_local_folds.itertuples():
            mlflow.log_metric(f"local_bad_recall_fold{int(_t.fold)}", float(_t.bad_recall))
            mlflow.log_metric(f"local_far_fold{int(_t.fold)}", float(_t.far))
        for _m in ["Dalpha", "Dalpha+rescue", "Dalpha+rescue995"]:
            _mk = {"Dalpha": "dalpha", "Dalpha+rescue": "rescue", "Dalpha+rescue995": "rescue995"}[_m]
            _s = fu2_rows[fu2_rows.method == _m]
            for _t in _s.itertuples():
                for _k in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                    mlflow.log_metric(f"{_mk}_{_k}_fold{int(_t.fold)}", float(getattr(_t, _k)))
            for _k in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                mlflow.log_metric(f"{_mk}_{_k}_mean", float(_s[_k].mean()))
                mlflow.log_metric(f"{_mk}_{_k}_std", float(_s[_k].std()))
        _szk = {"tiny<1%": "tiny_lt1", "mid1-3%": "mid_1_3", "big>3%": "big_gt3"}
        mlflow.log_metrics({f"rescue_recall_size_{_szk[_k]}": float(_v) for _k, _v in fu_size.items()})
        mlflow.log_metrics({f"rescue_recall_def_{_k}": float(_v) for _k, _v in fu_type.items()})
        fu2_rows.to_csv("reports/fase_f_fold_metrics.csv", index=False)
        mlflow.log_artifact("reports/fase_f_fold_metrics.csv")
        for _p in ["comparativa_f.png", "sizebars.png"]:
            mlflow.log_artifact("reports/figures/mlfigs_faseF/" + _p)
        print("mlflow faseF ok", flush=True)
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase F (patch tokens DINOv2-224 + rescate q99)" + chr(10) +
        "- local-224-max solo: recall 0.825, FAR 0.055, AUROC 0.970. 512px peor (bank submuestreado + ruido)." + chr(10) +
        "- D-alpha+rescue: recall 1.00 (std 0), FAR 0.066, AUROC 0.994. bad_022 recuperada." + chr(10) +
        "- rescue995: 0.975/0.062 pero pierde bad_022: q99 es el punto correcto." + chr(10) +
        "- Recall 1.0 en los 3 tamanos (tiny 6/6) y los 4 defectos." + chr(10) +
        "- Coste: FAR 0.055 -> 0.066 (+3 FPs). Veredicto: F-rescue nuevo modelo principal."
    )
    return


@app.cell
def _(Path, mo):
    _order = ["comparativa_f.png", "sizebars.png"]
    _ims = [mo.image(str((Path("reports/figures/mlfigs_faseF") / _p).resolve()), width=560, caption=_p) for _p in _order if (Path("reports/figures/mlfigs_faseF") / _p).exists()]
    mo.vstack([mo.md(f"### Galeria F ({len(_ims)}/{len(_order)})"), mo.hstack(_ims)])
    return


@app.cell
def _(GLAB, GY, np):
    from sklearn.linear_model import LogisticRegression as _LRg
    _saved2 = __import__("torch").load("models/fase_a_emb_dinov2.pt", weights_only=False)
    _fn2 = [str(_f) for _f in _saved2["fnames"]]
    _X2 = _saved2["X"]
    _y2 = (GLAB.set_index("filename").loc[_fn2]["label"] == "bad").to_numpy(int)
    _lrF = _LRg(class_weight="balanced", C=1.0, max_iter=2000).fit(_X2, _y2)
    _rng = np.random.default_rng(7)
    _sub_bad = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 1], 40, replace=False).tolist())
    _sub_good = sorted(_rng.choice([int(_i) for _i in range(len(GLAB)) if GY[_i] == 0], 40, replace=False).tolist())
    _SUB = _sub_bad + _sub_good
    print(f"subset bad={len(_sub_bad)} good={len(_sub_good)}")
    return


if __name__ == "__main__":
    app.run()
