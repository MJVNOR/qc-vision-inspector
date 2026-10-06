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
    # Fase B — PatchCore no supervisado (anomalib)
    Solo GOOD para entrenar; BAD para threshold/test.
    Mismas metricas y esquema MLflow que fase A.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_b ready")
    return


@app.cell
def _():
    ptag = "wrn50"
    return (ptag,)


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    BDIR = Path("transistor_binary")
    LAB = pd.read_csv(BDIR / "labels.csv")
    PBACK = {"wrn50": "wide_resnet50_2", "r18": "resnet18"}
    Y = (LAB["label"] == "bad").to_numpy(int)
    FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(LAB, Y))
    IMGSZ = 256
    LAB.shape
    return BDIR, FOLDS, IMGSZ, LAB, PBACK, Path, Y, pd


@app.cell
def _(BDIR, FOLDS, IMGSZ, LAB, PBACK, Y, pd, ptag):
    import torch
    import numpy as np
    from PIL import Image
    import torch.nn.functional as _F
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    from anomalib.models.components.feature_extractors import TimmFeatureExtractor
    _IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def _pb_pre(_im, _sz=IMGSZ, _dev="cuda"):
        _t = _im.convert("RGB").resize((_sz, _sz))
        _a = np.asarray(_t, dtype=np.float32) / 255.0
        _x = torch.from_numpy(_a).permute(2, 0, 1).unsqueeze(0)
        _x = (_x - _IMN_MEAN) / _IMN_STD
        return _x.to(_dev)
    def _pb_patches(_feat):
        _l2, _l3 = _feat["layer2"], _feat["layer3"]
        _l2r = _F.interpolate(_l2, size=_l3.shape[-2:], mode="bilinear", align_corners=False)
        _cat = torch.cat([_l2r, _l3], dim=1).squeeze(0)
        _c, _h, _w = _cat.shape
        return _cat.view(_c, -1).T, (_h, _w)
    _pdev = "cuda" if torch.cuda.is_available() else "cpu"
    _ext = TimmFeatureExtractor(backbone=PBACK[ptag], layers=["layer2", "layer3"], pre_trained=True)
    _ext.eval().to(_pdev)
    _pb_rows, pb_maps, pb_testscores, _pb_testy = [], {}, [], []
    with torch.inference_mode():
        for _fold, (_tr, _te) in enumerate(FOLDS):
            _good_tr = [int(_i) for _i in _tr if Y[_i] == 0]
            _bank = []
            for _i in _good_tr:
                _r = LAB.iloc[int(_i)]
                _im = Image.open(BDIR / ("good" if _r["label"] == "good" else "bad") / _r["filename"])
                _pa, _ = _pb_patches(_ext(_pb_pre(_im, _dev=_pdev)))
                _bank.append(_pa.cpu())
            _bank = torch.cat(_bank)
            _g = torch.Generator().manual_seed(42)
            _sel = torch.randperm(_bank.shape[0], generator=_g)[:max(1, int(0.1 * _bank.shape[0]))]
            _bank = _bank[_sel].numpy()
            _nn = NearestNeighbors(n_neighbors=9).fit(_bank)
            def _score(_idx):
                _r = LAB.iloc[int(_idx)]
                _im = Image.open(BDIR / ("good" if _r["label"] == "good" else "bad") / _r["filename"])
                _pa, _hw = _pb_patches(_ext(_pb_pre(_im, _dev=_pdev)))
                _d, _ = _nn.kneighbors(_pa.cpu().numpy())
                _pm = _d.mean(axis=1)
                return float(_pm.max()), _pm.reshape(_hw)
            _tr_scores = [_score(_i)[0] for _i in _good_tr]
            _te_s, _te_m = [], {}
            for _i in [int(_x) for _x in _te]:
                _ss, _mm = _score(_i)
                _te_s.append(_ss)
                _te_m[int(_i)] = _mm
            _te_s = np.array(_te_s)
            _yte = Y[_te]
            for _ft in [0.05, 0.01]:
                _cut = float(np.quantile(np.array(_tr_scores), 1 - _ft))
                _pr = (_te_s >= _cut).astype(int)
                _tp = int(((_yte == 1) & (_pr == 1)).sum())
                _fn = int(((_yte == 1) & (_pr == 0)).sum())
                _fp = int(((_yte == 0) & (_pr == 1)).sum())
                _tn = int(((_yte == 0) & (_pr == 0)).sum())
                _pc, _rc, _f1, _ = precision_recall_fscore_support(_yte, _pr, average="binary", zero_division=0)
                _fpr, _tpr, _ = roc_curve(_yte, _te_s)
                _pb_rows.append({"fold": _fold, "far_obj": _ft, "cut": round(_cut, 4),
                    "bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3),
                    "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3),
                    "f1": round(float(_f1), 3), "auroc": round(float(auc(_fpr, _tpr)), 3)})
            pb_testscores.append((_te, _te_s.copy()))
            pb_maps.update(_te_m)
            print(f"fold {_fold} done", flush=True)
    pb_rows = pd.DataFrame(_pb_rows)
    pb_rows.groupby("far_obj")[["bad_recall", "far", "frr", "precision", "f1", "auroc"]].agg(["mean", "std"]).round(3)
    return (
        Image,
        NearestNeighbors,
        TimmFeatureExtractor,
        np,
        pb_maps,
        pb_rows,
        pb_testscores,
        torch,
    )


@app.cell
def _(BDIR, Image, LAB, Path, Y, np, pb_maps, pb_rows, pb_testscores, ptag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    _ooi, _oos = [], []
    for _te, _ss in pb_testscores:
        _ooi.extend([int(_x) for _x in _te])
        _oos.extend([float(_x) for _x in _ss])
    _oos = np.array(_oos)
    _ooy = Y[np.array(_ooi)]
    pb_cut5 = float(pb_rows[pb_rows.far_obj == 0.05]["cut"].mean())
    _pb_pred = (_oos >= pb_cut5).astype(int)
    _dmap = dict(zip(LAB.filename, LAB.defect))
    _fnames_oof = [LAB.iloc[int(_i)]["filename"] for _i in _ooi]
    pb_perdef = {}
    for _d in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
        _ix = [_k for _k, _f in enumerate(_fnames_oof) if _dmap[_f] == _d]
        pb_perdef[_d] = round(float(_pb_pred[_ix].mean()), 3)
    _figdir = Path(f"mlfigs_faseB_{ptag}")
    _figdir.mkdir(exist_ok=True)
    _fig, _ax = plt.subplots()
    RocCurveDisplay.from_predictions(_ooy, _oos, ax=_ax, name="pooled OOF")
    _ax.plot([0, 1], [0, 1], "k--", lw=1)
    _ax.set_title(f"ROC PatchCore {ptag} (pooled)")
    _fig.savefig(_figdir / "roc.png", dpi=100)
    plt.close(_fig)
    _fig2, _ax2 = plt.subplots()
    _ax2.hist([_oos[_ooy == 0], _oos[_ooy == 1]], bins=30, label=["good", "bad"], alpha=0.6)
    _ax2.axvline(pb_cut5, color="red", lw=2, label="corte FAR5%")
    _ax2.legend()
    _ax2.set_title(f"Scores OOF {ptag} good vs bad")
    _fig2.savefig(_figdir / "scores_hist.png", dpi=100)
    plt.close(_fig2)
    _fig3, _ax3 = plt.subplots(figsize=(8, 4))
    _order = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]
    for _k, _d in enumerate(_order):
        _vv = _oos[[_k2 for _k2, _f in enumerate(_fnames_oof) if _dmap[_f] == _d]]
        _rg = np.random.default_rng(_k)
        _ax3.scatter(_rg.normal(_k, 0.08, len(_vv)), _vv, s=8, alpha=0.6)
    _ax3.set_xticks(range(len(_order)), _order)
    _ax3.axhline(pb_cut5, color="red", lw=2, label="corte FAR5%")
    _ax3.legend()
    _ax3.set_title(f"Scores por defecto {ptag}")
    _fig3.savefig(_figdir / "por_defecto.png", dpi=100)
    plt.close(_fig3)
    from matplotlib import cm as _cm
    from PIL import Image as _PI
    from PIL import ImageDraw as _DRm
    _fn_idx = [_ooi[_k] for _k in range(len(_ooi)) if _ooy[_k] == 1 and _pb_pred[_k] == 0]
    _fp_idx = [_ooi[_k] for _k in range(len(_ooi)) if _ooy[_k] == 0 and _pb_pred[_k] == 1][:8]
    _thumbs = []
    for _ii in _fn_idx + _fp_idx:
        _r = LAB.iloc[int(_ii)]
        _im0 = Image.open(BDIR / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB").resize((160, 160))
        _m = pb_maps[int(_ii)]
        _m = (_m - _m.min()) / max(_m.max() - _m.min(), 1e-8)
        _heat = (_cm.jet(_m)[:, :, :3] * 255).astype("uint8")
        _hov = _PI.fromarray(_heat).resize((160, 160))
        _thumbs.append((_r["filename"], _PI.blend(_im0, _hov, 0.45)))
    if _thumbs:
        _sheet = _PI.new("RGB", (160 * len(_thumbs), 180), "white")
        _drm = _DRm.Draw(_sheet)
        for _k, (_ff, _t) in enumerate(_thumbs):
            _sheet.paste(_t, (_k * 160, 20))
            _drm.text((_k * 160 + 4, 2), _ff, fill="black")
        _sheet.save(_figdir / "mapas.png")
    print(f"figs {ptag} ok cut={pb_cut5:.2f} perdef={pb_perdef} nmaps={len(_thumbs)}")
    return ConfusionMatrixDisplay, RocCurveDisplay, pb_cut5, pb_perdef, plt


@app.cell
def _(IMGSZ, PBACK, pb_cut5, pb_perdef, pb_rows, ptag):
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    from mlflow.tracking import MlflowClient as _MCf
    _MCf_hits = _MCf().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + f"faseB_{ptag}_patchcore" + chr(34))
    _MCf_cm = mlflow.start_run(run_id=sorted(_MCf_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _MCf_hits else mlflow.start_run(run_name=f"faseB_{ptag}_patchcore")
    with _MCf_cm:
        mlflow.log_params({"backbone": PBACK[ptag], "layers": "layer2+layer3", "img": IMGSZ,
            "coreset": 0.1, "k": 9, "train": "good-only", "cv": "StratifiedKFold-5", "seed": 42,
            "threshold": "quantile-goodtrain", "engine": "anomalib-TimmFeatureExtractor+skNN",
            "anomalib": "2.6.2", "n_good": 273, "n_bad": 40})
        for _t in pb_rows.itertuples():
            for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                mlflow.log_metric(f"{_m}_fold{int(_t.fold)}_far{int(_t.far_obj * 100)}", float(getattr(_t, _m)))
        for _fobj, _sfx in [(0.05, "thr5"), (0.01, "thr1")]:
            _sub = pb_rows[pb_rows.far_obj == _fobj]
            for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                mlflow.log_metric(f"{_sfx}_{_m}_mean", float(_sub[_m].mean()))
                mlflow.log_metric(f"{_sfx}_{_m}_std", float(_sub[_m].std()))
        mlflow.log_metrics({f"recall_def_{_k}": float(_v) for _k, _v in pb_perdef.items()})
        mlflow.log_metric("cut_far5", float(pb_cut5))
        pb_rows.to_csv(f"fase_b_fold_metrics_{ptag}.csv", index=False)
        mlflow.log_artifact(f"fase_b_fold_metrics_{ptag}.csv")
        for _p in ["roc.png", "scores_hist.png", "por_defecto.png", "mapas.png"]:
            mlflow.log_artifact(f"mlfigs_faseB_{ptag}/" + _p)
        print(f"mlflow faseB_{ptag} ok")
    return (mlflow,)


@app.cell
def _(Y, pb_cut5, pb_testscores):
    pb_ooi, _pb_oos_l = [], []
    for _te, _ss in pb_testscores:
        pb_ooi.extend([int(_x) for _x in _te])
        _pb_oos_l.extend([float(_x) for _x in _ss])
    import numpy as _npoof
    pb_oos = _npoof.array(_pb_oos_l)
    pb_ooy = Y[_npoof.array(pb_ooi)]
    pb_pred = (pb_oos >= pb_cut5).astype(int)
    del _npoof
    f"oof {len(pb_ooi)} imgs, cut={pb_cut5:.2f}, recall={(pb_pred[pb_ooy==1]).mean():.3f}"
    return pb_ooi, pb_oos, pb_ooy, pb_pred


@app.cell
def _(Path, pd, plt, ptag):
    from mlflow.tracking import MlflowClient as _MCab
    _qab = _MCab()
    def _mab(_name):
        _f = "tags.mlflow.runName = " + chr(34) + _name + chr(34)
        return _qab.search_runs(experiment_ids=["1"], filter_string=_f)[0].data.metrics
    _a2 = _mab("faseA_dinov2_logreg")
    _a3 = _mab("faseA_dinov3b_logreg")
    _bW = _mab("faseB_wrn50_patchcore")
    _bR = _mab("faseB_r18_patchcore")
    cmp_ab = pd.DataFrame([
        {"config": "A v2-Base LogReg", "recall_far5": round(_a2["thr5_bad_recall"], 3), "auroc": round(_a2["auroc_mean"], 3), "far": round(_a2["thr5_far"], 3)},
        {"config": "A v3-Base LogReg", "recall_far5": round(_a3["thr5_bad_recall"], 3), "auroc": round(_a3["auroc_mean"], 3), "far": round(_a3["thr5_far"], 3)},
        {"config": "B wrn50 PatchCore", "recall_far5": round(_bW["thr5_bad_recall_mean"], 3), "auroc": round(_bW["thr5_auroc_mean"], 3), "far": round(_bW["thr5_far_mean"], 3)},
        {"config": "B r18 PatchCore", "recall_far5": round(_bR["thr5_bad_recall_mean"], 3), "auroc": round(_bR["thr5_auroc_mean"], 3), "far": round(_bR["thr5_far_mean"], 3)},
    ])
    _figc, _axc = plt.subplots(figsize=(7, 3))
    _xx = range(len(cmp_ab))
    _axc.bar([_x - 0.2 for _x in _xx], cmp_ab.recall_far5, width=0.4, label="recall@FAR5%")
    _axc.bar([_x + 0.2 for _x in _xx], cmp_ab.auroc, width=0.4, label="AUROC")
    _axc.set_xticks(list(_xx), list(cmp_ab.config), rotation=15, ha="right", fontsize=8)
    _axc.legend(fontsize=8)
    _axc.set_title("A (supervisado) vs B (PatchCore) — desde MLflow")
    _axc.set_ylim(0, 1.05)
    plt.tight_layout()
    _figc.savefig(Path(f"mlfigs_faseB_{ptag}") / "comparativa_ab.png", dpi=100)
    plt.close(_figc)
    cmp_ab
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase B (PatchCore no supervisado, 2 backbones)" + chr(10) +
        "- wrn50 (L2+L3): recall 0.625@FAR5% (FAR real 0.055), AUROC 0.909." + chr(10) +
        "- r18 (L2+L3): recall 0.550@FAR5% (FAR real 0.055), AUROC 0.797." + chr(10) +
        "- Brecha vs Fase A supervisada (0.95 / 0.982): esperable con solo GOOD en train." + chr(10) +
        "- damaged_case el peor defecto tambien aqui (recall 0.2 en ambos)." + chr(10) +
        "- Siguiente: fase C (EfficientAD) y eleccion de produccion A/B/C."
    )
    return


@app.cell
def _(Path, RocCurveDisplay, Y, np, pb_oos, pb_ooy, pb_testscores, plt, ptag):
    from sklearn.metrics import PrecisionRecallDisplay
    _figdir2 = Path(f"mlfigs_faseB_{ptag}")
    _figp, _axp = plt.subplots()
    PrecisionRecallDisplay.from_predictions(pb_ooy, pb_oos, ax=_axp, name="pooled OOF")
    _axp.set_title(f"PR PatchCore {ptag} (pooled)")
    _figp.savefig(_figdir2 / "pr.png", dpi=100)
    plt.close(_figp)
    _figr, _axr = plt.subplots()
    for _f, (_te, _ss) in enumerate(pb_testscores):
        RocCurveDisplay.from_predictions(Y[np.array([int(_x) for _x in _te])], np.array([float(_x) for _x in _ss]), ax=_axr, name=f"fold {_f}")
    _axr.plot([0, 1], [0, 1], "k--", lw=1)
    _axr.set_title(f"ROC por fold PatchCore {ptag}")
    _figr.savefig(_figdir2 / "roc_folds.png", dpi=100)
    plt.close(_figr)
    print(f"pr+roc_folds {ptag} ok")
    return


@app.cell
def _(Path, np, pb_oos, pb_ooy, pb_rows, pd, plt, ptag):
    _sw = []
    for _q in [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]:
        _cut = float(np.quantile(pb_oos[pb_ooy == 0], _q))
        _pp = (pb_oos >= _cut).astype(int)
        _tp = int(((pb_ooy == 1) & (_pp == 1)).sum())
        _fn = int(((pb_ooy == 1) & (_pp == 0)).sum())
        _fp = int(((pb_ooy == 0) & (_pp == 1)).sum())
        _tn = int(((pb_ooy == 0) & (_pp == 0)).sum())
        _sw.append({"q": _q, "cut": round(_cut, 2), "recall": round(_tp / max(_tp + _fn, 1), 3),
            "far": round(_fp / max(_fp + _tn, 1), 3), "prec": round(_tp / max(_tp + _fp, 1), 3)})
    pb_sweep = pd.DataFrame(_sw)
    _figs, _axs = plt.subplots()
    _axs.plot(pb_sweep.q, pb_sweep.recall, "o-", label="recall")
    _axs.plot(pb_sweep.q, pb_sweep.far, "s-", label="far")
    _axs.plot(pb_sweep.q, pb_sweep.prec, "^-", label="precision")
    _axs.legend(fontsize=8)
    _axs.set_title(f"Barrido del corte pooled OOF {ptag}")
    _figs.savefig(Path(f"mlfigs_faseB_{ptag}") / "sweep.png", dpi=100)
    plt.close(_figs)
    _figb, _axb = plt.subplots()
    for _f in sorted(pb_rows.fold.unique()):
        _sub = pb_rows[pb_rows.fold == _f]
        _axb.scatter(_sub.far_obj, _sub.far, s=30, alpha=0.7, label=f"fold {_f}")
    _axb.plot([0, 0.06], [0, 0.06], "k--", lw=1, label="ideal")
    _axb.legend(fontsize=7)
    _axb.set_xlabel("FAR nominal (cuantil train-GOOD)")
    _axb.set_ylabel("FAR realizado (test)")
    _axb.set_title(f"Calibracion FAR {ptag}: nominal vs realizado")
    _figb.savefig(Path(f"mlfigs_faseB_{ptag}") / "calib.png", dpi=100)
    plt.close(_figb)
    pb_sweep
    return


@app.cell
def _(
    BDIR,
    ConfusionMatrixDisplay,
    Image,
    LAB,
    Path,
    pb_ooi,
    pb_ooy,
    pb_pred,
    plt,
    ptag,
):
    _figm, _axm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(pb_ooy, pb_pred, display_labels=["good", "bad"], ax=_axm)
    _axm.set_title(f"Confusion corte FAR5% {ptag}")
    _figm.savefig(Path(f"mlfigs_faseB_{ptag}") / "confusion.png", dpi=100)
    plt.close(_figm)
    from PIL import Image as _PI2
    from PIL import ImageDraw as _DR2
    _fns = [LAB.iloc[int(_i)]["filename"] for _i in pb_ooi]
    _fn_plain = [_fns[_k] for _k in range(len(_fns)) if pb_ooy[_k] == 1 and pb_pred[_k] == 0]
    _fp_plain = [_fns[_k] for _k in range(len(_fns)) if pb_ooy[_k] == 0 and pb_pred[_k] == 1][:8]
    _th2 = []
    for _ff in _fn_plain + _fp_plain:
        _im = Image.open(BDIR / ("bad" if _ff.startswith("bad") else "good") / _ff).convert("RGB").resize((160, 160))
        _th2.append((_ff, _im))
    if _th2:
        _sh2 = _PI2.new("RGB", (160 * len(_th2), 180), "white")
        _dr2 = _DR2.Draw(_sh2)
        for _k, (_ff, _im) in enumerate(_th2):
            _sh2.paste(_im, (_k * 160, 20))
            _dr2.text((_k * 160 + 4, 2), _ff, fill="black")
        _sh2.save(Path(f"mlfigs_faseB_{ptag}") / "errores_plain.png")
    print(f"confusion+galeria {ptag} ok fn={len(_fn_plain)} fp8={len(_fp_plain)}")
    return


@app.cell
def _(LAB, Path, np, pb_maps, plt, ptag):
    from sklearn.manifold import TSNE as _TSNE
    _dm = dict(zip(LAB.filename, LAB.defect))
    _ids = sorted(pb_maps.keys())
    _M = np.stack([np.asarray(pb_maps[_i], dtype=np.float32).ravel() for _i in _ids])
    _Z = _TSNE(n_components=2, perplexity=30, random_state=42, init="random").fit_transform(_M)
    _fgt, _axt = plt.subplots()
    for _d in ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]:
        _ii = [_k for _k, _i in enumerate(_ids) if _dm[LAB.iloc[int(_i)]["filename"]] == _d]
        _axt.scatter(_Z[_ii, 0], _Z[_ii, 1], s=10, alpha=0.7, label=_d)
    _axt.legend(markerscale=2, fontsize=8)
    _axt.set_title(f"t-SNE mapas PatchCore {ptag} (exploratorio)")
    _fgt.savefig(Path(f"mlfigs_faseB_{ptag}") / "tsne.png", dpi=100)
    plt.close(_fgt)
    print(f"tsne {ptag} ok n={len(_ids)}")
    return


@app.cell
def _(LAB, mo, pb_cut5, pb_ooi, pb_oos, pb_ooy, pd):
    import altair as alt
    _df = pd.DataFrame({"score": list(pb_oos), "label": ["bad" if _v else "good" for _v in pb_ooy]})
    _h = alt.Chart(_df).mark_bar(opacity=0.6).encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=30)), y="count():Q", color="label:N").properties(width=400)
    _r = alt.Chart(pd.DataFrame({"cut": [pb_cut5]})).mark_rule(color="red").encode(x="cut:Q")
    _dm2 = dict(zip(LAB.filename, LAB.defect))
    _fns2 = [LAB.iloc[int(_i)]["filename"] for _i in pb_ooi]
    _por = pd.DataFrame({"score": list(pb_oos), "defect": [_dm2[_f] for _f in _fns2]})
    _s = alt.Chart(_por).mark_tick(size=12).encode(x="defect:N", y="score:Q").properties(width=400)
    _rc = alt.Chart(pd.DataFrame({"cut": [pb_cut5]})).mark_rule(color="red").encode(y="cut:Q")
    mo.vstack([mo.md("### Scores interactivos"), _h + _r, _s + _rc])
    return


@app.cell
def _(Path, Y, np, pb_cut5, pb_testscores, plt, ptag):
    _figf, _axf = plt.subplots(figsize=(8, 4))
    for _f, (_te, _ss) in enumerate(pb_testscores):
        _tei = np.array([int(_x) for _x in _te])
        _ssv = np.array([float(_x) for _x in _ss])
        _rg0 = np.random.default_rng(100 + _f)
        _rg1 = np.random.default_rng(200 + _f)
        _axf.scatter(_rg0.normal(_f, 0.08, int((Y[_tei] == 0).sum())), _ssv[Y[_tei] == 0], s=8, alpha=0.5, color="blue", label="good" if _f == 0 else None)
        _axf.scatter(_rg1.normal(_f, 0.08, int((Y[_tei] == 1).sum())), _ssv[Y[_tei] == 1], s=20, alpha=0.8, color="orange", label="bad" if _f == 0 else None)
    _axf.set_xticks(range(5), [f"fold {_f}" for _f in range(5)])
    _axf.axhline(pb_cut5, color="red", lw=2, label="corte FAR5%")
    _axf.legend(fontsize=8)
    _axf.set_title(f"Scores test por fold {ptag}")
    _figf.savefig(Path(f"mlfigs_faseB_{ptag}") / "scores_folds.png", dpi=100)
    plt.close(_figf)
    print(f"scores_folds {ptag} ok")
    return


@app.cell
def _(Path, mlflow, ptag):
    from mlflow.tracking import MlflowClient as _MC3
    _c3 = _MC3()
    _hits3 = _c3.search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + f"faseB_{ptag}_patchcore" + chr(34))
    _rid3 = sorted(_hits3, key=lambda _r: _r.info.start_time)[-1].info.run_id
    _figdir3 = Path(f"mlfigs_faseB_{ptag}")
    with mlflow.start_run(run_id=_rid3):
        for _p in ["comparativa_ab.png", "pr.png", "roc_folds.png", "sweep.png", "calib.png", "confusion.png", "errores_plain.png", "tsne.png", "scores_folds.png"]:
            mlflow.log_artifact(str(_figdir3 / _p))
        print(f"extra figs log ok en {_rid3[:8]} ({ptag})")
    return


@app.cell
def _(Path, mo, ptag):
    _d = Path(f"mlfigs_faseB_{ptag}")
    _order = ["comparativa_ab.png", "roc.png", "roc_folds.png", "pr.png", "scores_hist.png", "scores_folds.png", "por_defecto.png", "sweep.png", "calib.png", "confusion.png", "errores_plain.png", "mapas.png", "tsne.png"]
    _ims = [mo.image(str((_d / _p).resolve()), width=520, caption=_p) for _p in _order if (_d / _p).exists()]
    _rows = [mo.hstack(_ims[_k:_k + 2]) for _k in range(0, len(_ims), 2)]
    mo.vstack([mo.md(f"### Galeria {ptag} ({len(_ims)}/{len(_order)})"), *_rows])
    return


@app.cell
def _(
    BDIR,
    IMGSZ,
    Image,
    LAB,
    NearestNeighbors,
    PBACK,
    Path,
    TimmFeatureExtractor,
    Y,
    mlflow,
    np,
    torch,
):
    import torch.nn.functional as _Fp
    import json as _Jp
    from mlflow.tracking import MlflowClient as _MCp
    _pcp = _MCp()
    _pdev = "cuda" if torch.cuda.is_available() else "cpu"
    _IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for _pt in ["wrn50", "r18"]:
        _ex = TimmFeatureExtractor(backbone=PBACK[_pt], layers=["layer2", "layer3"], pre_trained=True)
        _ex.eval().to(_pdev)
        _good = [int(_i) for _i in range(len(LAB)) if Y[_i] == 0]
        _parts = []
        with torch.inference_mode():
            for _i in _good:
                _r = LAB.iloc[_i]
                _im = Image.open(BDIR / "good" / _r["filename"]).convert("RGB").resize((IMGSZ, IMGSZ))
                _a = np.asarray(_im, dtype=np.float32) / 255.0
                _x = (torch.from_numpy(_a).permute(2, 0, 1).unsqueeze(0) - _IMEAN) / _ISTD
                _f = _ex(_x.to(_pdev))
                _l2r = _Fp.interpolate(_f["layer2"], size=_f["layer3"].shape[-2:], mode="bilinear", align_corners=False)
                _cat = torch.cat([_l2r, _f["layer3"]], dim=1).squeeze(0)
                _parts.append(_cat.view(_cat.shape[0], -1).T.cpu())
        _B = torch.cat(_parts)
        _g = torch.Generator().manual_seed(42)
        _B = _B[torch.randperm(_B.shape[0], generator=_g)[:max(1, int(0.1 * _B.shape[0]))]].numpy().astype("float32")
        _nn = NearestNeighbors(n_neighbors=9).fit(_B)
        _gs = np.array([float(_nn.kneighbors(_pm.numpy())[0].mean(axis=1).max()) for _pm in _parts])
        _c95, _c99 = float(np.quantile(_gs, 0.95)), float(np.quantile(_gs, 0.99))
        torch.save({"bank": _B, "backbone": PBACK[_pt], "layers": ["layer2", "layer3"], "img": IMGSZ, "k": 9, "coreset": 0.1, "seed": 42}, f"fase_b_bank_{_pt}.pt")
        Path(f"thresholds_patchcore_{_pt}.json").write_text(_Jp.dumps({"cut_produccion": _c95, "cut_far1": _c99, "regla": "cuantil 95 de scores GOOD full-train (273)", "backbone": PBACK[_pt], "layers": "layer2+layer3", "k": 9}, indent=1))
        _h = _pcp.search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + f"faseB_{_pt}_patchcore" + chr(34))
        _rid = sorted(_h, key=lambda _r: _r.info.start_time)[-1].info.run_id
        with mlflow.start_run(run_id=_rid):
            mlflow.log_artifact(f"fase_b_bank_{_pt}.pt")
            mlflow.log_artifact(f"thresholds_patchcore_{_pt}.json")
        print(f"prod {_pt} ok bank={_B.shape} cut95={_c95:.2f} -> {_rid[:8]}", flush=True)
        del _ex, _parts, _B
    return


if __name__ == "__main__":
    app.run()
