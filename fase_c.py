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
    # Fase C — EfficientAD no supervisado (anomalib)
    Solo GOOD para entrenar; BAD para threshold/test.
    Mismas metricas y esquema MLflow que fases A/B.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_c ready")
    return


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    BDIR = Path("transistor_binary")
    LAB = pd.read_csv(BDIR / "labels.csv")
    CSIZES = {"es": "small", "em": "medium"}
    Y = (LAB["label"] == "bad").to_numpy(int)
    FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(LAB, Y))
    IMGSZ = 256
    NSTEPS = 5000
    LAB.shape
    return BDIR, CSIZES, FOLDS, IMGSZ, LAB, NSTEPS, Path, Y, pd


@app.cell
def _(BDIR, CSIZES, FOLDS, IMGSZ, LAB, NSTEPS, Y, pd):
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    from anomalib.data.dataclasses.torch import ImageBatch
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine
    from lightning.pytorch import LightningDataModule
    from lightning.pytorch import seed_everything as _seed
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    class _DM(LightningDataModule):
        def __init__(self, _tensors):
            super().__init__()
            self.train_batch_size = 1
            self.name = "transistor"
            self.category = "transistor"
            self._tensors = _tensors
        def train_dataloader(self):
            return DataLoader(self._tensors, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
    def _tload(_idx):
        _r = LAB.iloc[int(_idx)]
        _im = Image.open(BDIR / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB").resize((IMGSZ, IMGSZ))
        return torch.from_numpy(np.asarray(_im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    def _pred_all(_eng, _model, _idxs, _bs=16):
        _dl = DataLoader([_tload(_i) for _i in _idxs], batch_size=_bs, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))
        _ss, _mm = [], []
        for _pb in _eng.predict(model=_model, dataloaders=_dl):
            _ss.extend([float(_x) for _x in _pb.pred_score.flatten().tolist()])
            _mm.extend([_m for _m in _pb.anomaly_map.cpu().numpy()])
        return np.array(_ss), _mm
    _seed(42)
    _all = [_tload(_i) for _i in range(len(LAB))]
    ec_rows, ec_testscores, ec_maps = [], {}, {}
    for _tag, _size in CSIZES.items():
        for _fold, (_tr, _te) in enumerate(FOLDS):
            _good_tr = [int(_i) for _i in _tr if Y[_i] == 0]
            _tel = [int(_x) for _x in _te]
            _m = EfficientAd(model_size=_size, visualizer=False, evaluator=False)
            _eng = Engine(logger=False, default_root_dir=f"results_c_{_tag}", max_steps=NSTEPS)
            _eng.fit(model=_m, datamodule=_DM([_all[_i] for _i in _good_tr]))
            _tr_scores, _ = _pred_all(_eng, _m, _good_tr)
            _te_s, _te_m = _pred_all(_eng, _m, _tel)
            for _ii, _mm in zip(_tel, _te_m):
                ec_maps[(_tag, int(_ii))] = _mm
            ec_testscores[(_tag, _fold)] = (_tel, _te_s.copy())
            _yte = Y[_te]
            for _ft in [0.05, 0.01]:
                _cut = float(np.quantile(_tr_scores, 1 - _ft))
                _pr = (_te_s >= _cut).astype(int)
                _tp = int((( _yte == 1) & (_pr == 1)).sum())
                _fn = int((( _yte == 1) & (_pr == 0)).sum())
                _fp = int((( _yte == 0) & (_pr == 1)).sum())
                _tn = int((( _yte == 0) & (_pr == 0)).sum())
                _pc, _rc, _f1, _ = precision_recall_fscore_support(_yte, _pr, average="binary", zero_division=0)
                _fpr, _tpr, _ = roc_curve(_yte, _te_s)
                ec_rows.append({"tag": _tag, "fold": _fold, "far_obj": _ft, "cut": round(_cut, 4),
                    "bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3),
                    "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3),
                    "f1": round(float(_f1), 3), "auroc": round(float(auc(_fpr, _tpr)), 3)})
            print(f"{_tag} fold {_fold} done", flush=True)
    ec_rows = pd.DataFrame(ec_rows)
    ec_rows.groupby(["tag", "far_obj"])[["bad_recall", "far", "frr", "precision", "f1", "auroc"]].agg(["mean", "std"]).round(3)
    return Image, ec_maps, ec_rows, ec_testscores, np


@app.cell
def _(
    BDIR,
    CSIZES,
    Image,
    LAB,
    NSTEPS,
    Path,
    Y,
    ec_maps,
    ec_rows,
    ec_testscores,
    np,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from matplotlib import cm as _cm
    from PIL import Image as _PI
    from PIL import ImageDraw as _DR
    import json as _J
    ec_cut5, ec_perdef = {}, {}
    for _tag in CSIZES:
        _oi, _os = [], []
        for _f in range(5):
            _te, _ss = ec_testscores[(_tag, _f)]
            _oi.extend([int(_x) for _x in _te])
            _os.extend([float(_x) for _x in _ss])
        _os = np.array(_os)
        _oy = Y[np.array(_oi)]
        _cut = float(ec_rows[(ec_rows.tag == _tag) & (ec_rows.far_obj == 0.05)]["cut"].mean())
        ec_cut5[_tag] = _cut
        _pr = (_os >= _cut).astype(int)
        _dm = dict(zip(LAB.filename, LAB.defect))
        _fns = [LAB.iloc[int(_i)]["filename"] for _i in _oi]
        _pd = {}
        for _d in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
            _ix = [_k for _k, _f in enumerate(_fns) if _dm[_f] == _d]
            _pd[_d] = round(float(_pr[_ix].mean()), 3)
        ec_perdef[_tag] = _pd
        _fd = Path(f"mlfigs_faseC_{_tag}")
        _fd.mkdir(exist_ok=True)
        _fg, _ax = plt.subplots()
        RocCurveDisplay.from_predictions(_oy, _os, ax=_ax, name="pooled OOF")
        _ax.plot([0, 1], [0, 1], "k--", lw=1)
        _ax.set_title(f"ROC EfficientAD-{_tag} (pooled)")
        _fg.savefig(_fd / "roc.png", dpi=100)
        plt.close(_fg)
        _fh, _ah = plt.subplots()
        _ah.hist([_os[_oy == 0], _os[_oy == 1]], bins=30, label=["good", "bad"], alpha=0.6)
        _ah.axvline(_cut, color="red", lw=2, label="corte FAR5%")
        _ah.legend()
        _ah.set_title(f"Scores OOF {_tag} good vs bad")
        _fh.savefig(_fd / "scores_hist.png", dpi=100)
        plt.close(_fh)
        _fp2, _ap = plt.subplots(figsize=(8, 4))
        _order = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]
        for _k, _d in enumerate(_order):
            _vv = _os[[_k2 for _k2, _f in enumerate(_fns) if _dm[_f] == _d]]
            _rg = np.random.default_rng(_k)
            _ap.scatter(_rg.normal(_k, 0.08, len(_vv)), _vv, s=8, alpha=0.6)
        _ap.set_xticks(range(len(_order)), _order)
        _ap.axhline(_cut, color="red", lw=2, label="corte FAR5%")
        _ap.legend()
        _ap.set_title(f"Scores por defecto {_tag}")
        _fp2.savefig(_fd / "por_defecto.png", dpi=100)
        plt.close(_fp2)
        _fm, _am = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(_oy, _pr, display_labels=["good", "bad"], ax=_am)
        _am.set_title(f"Confusion corte FAR5% {_tag}")
        _fm.savefig(_fd / "confusion.png", dpi=100)
        plt.close(_fm)
        _th = []
        for _k in range(len(_oi)):
            if (_oy[_k] == 1 and _pr[_k] == 0) or (_oy[_k] == 0 and _pr[_k] == 1 and len([_t for _t, _ in _th if _t.startswith("good")]) < 8):
                _ii = _oi[_k]
                _r = LAB.iloc[int(_ii)]
                _im0 = Image.open(BDIR / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB").resize((160, 160))
                _m = np.asarray(ec_maps[(_tag, int(_ii))], dtype=np.float32)
                _m = (_m - _m.min()) / max(_m.max() - _m.min(), 1e-8)
                _hov = _PI.fromarray((_cm.jet(_m)[:, :, :3] * 255).astype("uint8")).resize((160, 160))
                _th.append((_r["filename"], _PI.blend(_im0, _hov, 0.45)))
        if _th:
            _sh = _PI.new("RGB", (160 * len(_th), 180), "white")
            _dr = _DR.Draw(_sh)
            for _k, (_ff, _t) in enumerate(_th):
                _sh.paste(_t, (_k * 160, 20))
                _dr.text((_k * 160 + 4, 2), _ff, fill="black")
            _sh.save(_fd / "mapas.png")
        Path(f"thresholds_efficientad_{_tag}.json").write_text(_J.dumps({"cut_produccion": _cut, "regla": "media de cortes calibrados a FAR5% en train-folds (receta fase A)", "model": "efficientad-" + CSIZES[_tag], "steps": NSTEPS, "trained_on": "good-only-folds"}, indent=1))
        print(f"figs {_tag} ok cut={_cut:.4f} perdef={_pd} nmaps={len(_th)}", flush=True)
    return ec_cut5, ec_perdef, plt


@app.cell
def _(CSIZES, IMGSZ, NSTEPS, ec_cut5, ec_perdef, ec_rows):
    import mlflow
    from mlflow.tracking import MlflowClient as _MCc
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    for _tag in CSIZES:
        _sub = ec_rows[ec_rows.tag == _tag]
        _rn = f"faseC_{_tag}_efficientad"
        _hits = _MCc().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
        _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
        with _cm:
            mlflow.log_params({"model": "efficientad-" + CSIZES[_tag], "steps": NSTEPS, "batch": 1, "img": IMGSZ,
                "train": "good-only", "cv": "StratifiedKFold-5", "seed": 42, "threshold": "quantile-goodtrain",
                "engine": "anomalib-EfficientAd+Engine", "anomalib": "2.6.2", "n_good": 273, "n_bad": 40})
            for _t in _sub.itertuples():
                for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                    mlflow.log_metric(f"{_m}_fold{int(_t.fold)}_far{int(_t.far_obj * 100)}", float(getattr(_t, _m)))
            for _fobj, _sfx in [(0.05, "thr5"), (0.01, "thr1")]:
                _s2 = _sub[_sub.far_obj == _fobj]
                for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                    mlflow.log_metric(f"{_sfx}_{_m}_mean", float(_s2[_m].mean()))
                    mlflow.log_metric(f"{_sfx}_{_m}_std", float(_s2[_m].std()))
            mlflow.log_metrics({f"recall_def_{_k}": float(_v) for _k, _v in ec_perdef[_tag].items()})
            mlflow.log_metric("cut_far5", float(ec_cut5[_tag]))
            _sub.to_csv(f"fase_c_fold_metrics_{_tag}.csv", index=False)
            mlflow.log_artifact(f"fase_c_fold_metrics_{_tag}.csv")
            mlflow.log_artifact(f"thresholds_efficientad_{_tag}.json")
            for _p in ["roc.png", "scores_hist.png", "por_defecto.png", "confusion.png", "mapas.png"]:
                mlflow.log_artifact(f"mlfigs_faseC_{_tag}/" + _p)
            print(f"mlflow faseC_{_tag} ok", flush=True)
    return (mlflow,)


@app.cell
def _(CSIZES, Path, mlflow, pd, plt):
    from mlflow.tracking import MlflowClient as _MCa
    _qa = _MCa()
    def _mab(_name):
        _f = "tags.mlflow.runName = " + chr(34) + _name + chr(34)
        return _qa.search_runs(experiment_ids=["1"], filter_string=_f)[0].data.metrics
    _v2 = _mab("faseA_dinov2_logreg")
    _vb = _mab("faseA_dinov3b_logreg")
    _bw = _mab("faseB_wrn50_patchcore")
    _ce = _mab("faseC_es_efficientad")
    _cm2 = _mab("faseC_em_efficientad")
    cmp_abc = pd.DataFrame([
        {"config": "A v2-Base LogReg", "recall_far5": round(_v2["thr5_bad_recall"], 3), "auroc": round(_v2["auroc_mean"], 3), "far": round(_v2["thr5_far"], 3)},
        {"config": "A v3-Base LogReg", "recall_far5": round(_vb["thr5_bad_recall"], 3), "auroc": round(_vb["auroc_mean"], 3), "far": round(_vb["thr5_far"], 3)},
        {"config": "B wrn50 PatchCore", "recall_far5": round(_bw["thr5_bad_recall_mean"], 3), "auroc": round(_bw["thr5_auroc_mean"], 3), "far": round(_bw["thr5_far_mean"], 3)},
        {"config": "C es EfficientAD", "recall_far5": round(_ce["thr5_bad_recall_mean"], 3), "auroc": round(_ce["thr5_auroc_mean"], 3), "far": round(_ce["thr5_far_mean"], 3)},
        {"config": "C em EfficientAD", "recall_far5": round(_cm2["thr5_bad_recall_mean"], 3), "auroc": round(_cm2["thr5_auroc_mean"], 3), "far": round(_cm2["thr5_far_mean"], 3)},
    ])
    _fx, _axx = plt.subplots(figsize=(8, 3))
    _xx = range(len(cmp_abc))
    _axx.bar([_x - 0.2 for _x in _xx], cmp_abc.recall_far5, width=0.4, label="recall@FAR5%")
    _axx.bar([_x + 0.2 for _x in _xx], cmp_abc.auroc, width=0.4, label="AUROC")
    _axx.set_xticks(list(_xx), list(cmp_abc.config), rotation=15, ha="right", fontsize=8)
    _axx.legend(fontsize=8)
    _axx.set_title("A vs B vs C — desde MLflow")
    _axx.set_ylim(0, 1.05)
    plt.tight_layout()
    for _tag in CSIZES:
        _fx.savefig(Path(f"mlfigs_faseC_{_tag}") / "comparativa_abc.png", dpi=100)
        _h2 = _qa.search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + f"faseC_{_tag}_efficientad" + chr(34))
        _r2 = sorted(_h2, key=lambda _r: _r.info.start_time)[-1].info.run_id
        with mlflow.start_run(run_id=_r2):
            mlflow.log_artifact(f"mlfigs_faseC_{_tag}/comparativa_abc.png")
    plt.close(_fx)
    cmp_abc
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase C (EfficientAD S+M, 5000 steps, batch 1)" + chr(10) +
        "- em: recall 0.90@FAR5% (FAR real 0.096), AUROC 0.968 — empata a A-v3b en recall." + chr(10) +
        "- es: recall 0.60@FAR5%, AUROC 0.831 — netamente peor, el tamano importa aqui." + chr(10) +
        "- damaged_case deja de ser el peor en em (0.8): el mejor no supervisado en ese defecto." + chr(10) +
        "- Decision de produccion: sigue A-v2 (0.95 / 0.982). C-em queda segundo."
    )
    return


@app.cell
def _(CSIZES, Path, mo):
    _order = ["comparativa_abc.png", "roc.png", "scores_hist.png", "por_defecto.png", "confusion.png", "mapas.png"]
    _gal = {}
    for _tag in CSIZES:
        _d = Path(f"mlfigs_faseC_{_tag}")
        _ims = [mo.image(str((_d / _p).resolve()), width=520, caption=f"{_tag}/{_p}") for _p in _order if (_d / _p).exists()]
        _gal[_tag] = _ims
    _rows = []
    for _tag in CSIZES:
        _ims = _gal[_tag]
        _rows.append(mo.md(f"### {_tag} ({len(_ims)}/{len(_order)})"))
        _rows.extend([mo.hstack(_ims[_k:_k + 2]) for _k in range(0, len(_ims), 2)])
    mo.vstack(_rows)
    return


@app.cell
def _(CSIZES, ec_testscores, mlflow, np):
    from mlflow.tracking import MlflowClient as _MCs
    _ec_d = {}
    for (_tag, _fold), (_te, _ss) in ec_testscores.items():
        _ec_d[f"{_tag}_fold{_fold}_idx"] = np.asarray(_te)
        _ec_d[f"{_tag}_fold{_fold}_score"] = np.asarray(_ss, dtype=np.float64)
    np.savez_compressed("fase_c_oof_scores.npz", **_ec_d)
    for _tag in CSIZES:
        _h = _MCs().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + f"faseC_{_tag}_efficientad" + chr(34))
        _rid = sorted(_h, key=lambda _r: _r.info.start_time)[-1].info.run_id
        with mlflow.start_run(run_id=_rid):
            mlflow.log_artifact("fase_c_oof_scores.npz")
        print(f"oof npz -> {_rid[:8]} ({_tag})", flush=True)
    return


if __name__ == "__main__":
    app.run()
