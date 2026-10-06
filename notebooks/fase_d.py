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
    # Fase D — ensemble DINOv2 + EfficientAD-em
    Scores OOF de ambos (`models/fase_a_oof_dinov2.npz`, `models/fase_c_oof_scores.npz`).
    Fusion alpha + LogReg-2D, corte calibrado en train. Objetivo: max recall BAD con FAR<=5%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_d ready")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    ELAB = pd.read_csv(Path("data/processed/transistor_binary") / "labels.csv")
    EY = (ELAB["label"] == "bad").to_numpy(int)
    EFOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(ELAB, EY))
    _d2 = np.load("models/fase_a_oof_dinov2.npz")
    _d2map = dict(zip([str(_f) for _f in _d2["fnames"]], _d2["scores"]))
    D2 = np.array([_d2map[_f] for _f in ELAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emmap = {}
    for _f in range(5):
        for _i, _s in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emmap[int(_i)] = float(_s)
    EM = np.array([_emmap[_i] for _i in range(len(ELAB))])
    assert len(_emmap) == len(ELAB) == len(D2) == 313
    f"d2 [{D2.min():.3f},{D2.max():.3f}] em [{EM.min():.3f},{EM.max():.3f}]"
    return D2, EFOLDS, ELAB, EM, EY, Path, np, pd


@app.cell
def _(D2, EFOLDS, EM, EY, np, pd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    def _mets(_yt, _pr, _sc):
        _tp = int(((_yt == 1) & (_pr == 1)).sum())
        _fn = int(((_yt == 1) & (_pr == 0)).sum())
        _fp = int(((_yt == 0) & (_pr == 1)).sum())
        _tn = int(((_yt == 0) & (_pr == 0)).sum())
        _pc, _rc, _f1, _ = precision_recall_fscore_support(_yt, _pr, average="binary", zero_division=0)
        _fpr, _tpr, _ = roc_curve(_yt, _sc)
        return {"bad_recall": round(float(_rc), 3), "far": round(_fp / max(_fp + _tn, 1), 3),
            "frr": round(_fn / max(_fn + _tp, 1), 3), "precision": round(float(_pc), 3),
            "f1": round(float(_f1), 3), "auroc": round(float(auc(_fpr, _tpr)), 3)}
    en_rows, en_cuts, en_scores, en_alpha = [], {}, {}, {}
    for _fold, (_tr, _te) in enumerate(EFOLDS):
        _tr = np.array(_tr)
        _tel = [int(_x) for _x in _te]
        _gtr = _tr[EY[_tr] == 0]
        _yte = EY[_te]
        _med_d, _iqr_d = np.median(D2[_gtr]), np.subtract(*np.percentile(D2[_gtr], [75, 25])) + 1e-9
        _med_e, _iqr_e = np.median(EM[_gtr]), np.subtract(*np.percentile(EM[_gtr], [75, 25])) + 1e-9
        _dn_tr, _dn_te = (D2[_tr] - _med_d) / _iqr_d, (D2[_te] - _med_d) / _iqr_d
        _de_tr, _de_te = (EM[_tr] - _med_e) / _iqr_e, (EM[_te] - _med_e) / _iqr_e
        _gd_tr = _tr[EY[_tr] == 0]
        _cut_d = float(np.quantile(D2[_gd_tr], 0.95))
        _cut_e = float(np.quantile(EM[_gd_tr], 0.95))
        _sc_d, _sc_e = D2[_te], EM[_te]
        en_cuts[("dino", _fold)] = _cut_d
        en_cuts[("em", _fold)] = _cut_e
        en_scores[("dino", _fold)] = (_tel, _sc_d.copy())
        en_scores[("em", _fold)] = (_tel, _sc_e.copy())
        en_rows.append({"method": "dino", "fold": _fold, **_mets(_yte, (_sc_d >= _cut_d).astype(int), _sc_d)})
        en_rows.append({"method": "em", "fold": _fold, **_mets(_yte, (_sc_e >= _cut_e).astype(int), _sc_e)})
        _gdn = _dn_tr[EY[_tr] == 0]
        _gde = _de_tr[EY[_tr] == 0]
        _best = None
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _ftr = _a * _dn_tr + (1 - _a) * _de_tr
            _fgt = _a * _gdn + (1 - _a) * _gde
            _ca = float(np.quantile(_fgt, 0.95))
            _ptr = (_ftr >= _ca).astype(int)
            _tpr_ = ((_ptr == 1) & (EY[_tr] == 1)).sum() / max((EY[_tr] == 1).sum(), 1)
            _fpr_ = ((_ptr == 1) & (EY[_tr] == 0)).sum() / max((EY[_tr] == 0).sum(), 1)
            _key = (0 if _fpr_ <= 0.05 else 1, -_tpr_, _fpr_)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        en_alpha[_fold] = _ba
        _fte = _ba * _dn_te + (1 - _ba) * _de_te
        en_cuts[("alpha", _fold)] = _bca
        en_scores[("alpha", _fold)] = (_tel, _fte.copy())
        en_rows.append({"method": "alpha", "fold": _fold, "alpha": _ba, **_mets(_yte, (_fte >= _bca).astype(int), _fte)})
        _lr = LogisticRegression(max_iter=2000).fit(np.c_[D2[_tr], EM[_tr]], EY[_tr])
        _ltr = _lr.predict_proba(np.c_[D2[_tr], EM[_tr]])[:, 1]
        _cl = float(np.quantile(_ltr[EY[_tr] == 0], 0.95))
        _lte = _lr.predict_proba(np.c_[D2[_te], EM[_te]])[:, 1]
        en_cuts[("lr2", _fold)] = _cl
        en_scores[("lr2", _fold)] = (_tel, _lte.copy())
        en_rows.append({"method": "lr2", "fold": _fold, **_mets(_yte, (_lte >= _cl).astype(int), _lte)})
        print(f"fold {_fold} done alpha={_ba}", flush=True)
    en_rows = pd.DataFrame(en_rows)
    en_rows.groupby("method")[["bad_recall", "far", "frr", "precision", "f1", "auroc"]].agg(["mean", "std"]).round(3)
    return en_alpha, en_cuts, en_rows, en_scores


@app.cell
def _(EFOLDS, ELAB, EY, en_cuts, en_scores, pd):
    _res = {}
    for (_m, _f), (_tel, _ss) in en_scores.items():
        for _ii, _sv in zip(_tel, _ss):
            _res.setdefault(int(_ii), {})[_m] = float(_sv)
    _rows = []
    for _ii in sorted(_res.keys()):
        if EY[_ii] == 0:
            continue
        _r = ELAB.iloc[int(_ii)]
        _f = next(_ff for _ff, (_tr, _te) in enumerate(EFOLDS) if int(_ii) in set(int(_x) for _x in _te))
        _row = {"filename": _r["filename"], "defect": _r["defect"]}
        for _m in ["dino", "em", "alpha", "lr2"]:
            _sc = _res[_ii][_m]
            _row[f"{_m}_score"] = round(_sc, 4)
            _row[f"{_m}_result"] = "bad" if _sc >= en_cuts[(_m, _f)] else "good"
        _rows.append(_row)
    en_bad = pd.DataFrame(_rows)
    en_bad.to_csv("reports/fase_d_bad_table.csv", index=False)
    _dm = dict(zip(ELAB.filename, ELAB.defect))
    en_perdef = {}
    for _m in ["dino", "em", "alpha", "lr2"]:
        _pd = {}
        for _d in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
            _s = en_bad[en_bad.defect == _d]
            _pd[_d] = round(float((_s[f"{_m}_result"] == "bad").mean()), 3)
        en_perdef[_m] = _pd
    print(en_perdef)
    en_bad.head(40).to_string()
    return (en_perdef,)


@app.cell
def _(EY, en_cuts, en_scores, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from pathlib import Path as _P2
    _fd = _P2("reports/figures/mlfigs_faseD")
    _fd.mkdir(exist_ok=True)
    _pool = {}
    for _m in ["dino", "em", "alpha", "lr2"]:
        _ii, _ss = [], []
        for _f in range(5):
            _te, _sv = en_scores[(_m, _f)]
            _ii.extend([int(_x) for _x in _te])
            _ss.extend([float(_x) for _x in _sv])
        _ss = np.array(_ss)
        _yy = EY[np.array(_ii)]
        _pool[_m] = (_ss, _yy)
    for _m in ["alpha", "lr2"]:
        _ss, _yy = _pool[_m]
        _cc = np.mean([en_cuts[(_m, _f)] for _f in range(5)])
        _fg, _ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(_yy, (_ss >= _cc).astype(int), display_labels=["good", "bad"], ax=_ax)
        _ax.set_title(f"Confusion pooled {_m}")
        _fg.savefig(_fd / f"confusion_{_m}.png", dpi=100)
        plt.close(_fg)
    _sd, _yd = _pool["dino"]
    _se, _ye = _pool["em"]
    _cd = np.mean([en_cuts[("dino", _f)] for _f in range(5)])
    _ce = np.mean([en_cuts[("em", _f)] for _f in range(5)])
    _fh, _ah = plt.subplots()
    _ah.scatter(_sd[_yd == 0], _se[_yd == 0], s=10, alpha=0.4, label="good")
    _ah.scatter(_sd[(_yd == 1)], _se[(_yd == 1)], s=25, alpha=0.8, label="bad")
    _ah.axvline(_cd, color="red", lw=1)
    _ah.axhline(_ce, color="red", lw=1)
    _ah.set_xlabel("dino score")
    _ah.set_ylabel("efficientad score")
    _ah.legend(fontsize=8)
    _ah.set_title("DINO vs EfficientAD (rojo = cortes)")
    _fh.savefig(_fd / "scatter_dino_em.png", dpi=100)
    plt.close(_fh)
    print("figs D ok")
    return


@app.cell
def _(en_alpha, en_perdef, en_rows):
    import mlflow
    from mlflow.tracking import MlflowClient as _MCd
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    _rn = "faseD_ensemble"
    _hits = _MCd().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
    _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
    with _cm:
        mlflow.log_params({"base1": "dinov2-LogReg-C1.0", "base2": "efficientad-em-5000steps", "norm": "iqr-train-good",
            "alpha_grid": "0-1-step0.05", "alpha_rule": "max-train-recall-s.t.-trainFAR5%", "lr2": "LogReg-default",
            "threshold": "quantile-goodtrain-95", "cv": "StratifiedKFold-5-outer", "seed": 42, "n_good": 273, "n_bad": 40})
        mlflow.log_params({f"alpha_fold{_f}": _a for _f, _a in en_alpha.items()})
        for _m in ["dino", "em", "alpha", "lr2"]:
            _s = en_rows[en_rows.method == _m]
            for _t in _s.itertuples():
                for _k in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                    mlflow.log_metric(f"{_m}_{_k}_fold{int(_t.fold)}", float(getattr(_t, _k)))
            for _k in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                mlflow.log_metric(f"{_m}_{_k}_mean", float(_s[_k].mean()))
                mlflow.log_metric(f"{_m}_{_k}_std", float(_s[_k].std()))
            mlflow.log_metrics({f"{_m}_recall_def_{_k}": float(_v) for _k, _v in en_perdef[_m].items()})
        en_rows.to_csv("reports/fase_d_fold_metrics.csv", index=False)
        mlflow.log_artifact("reports/fase_d_fold_metrics.csv")
        mlflow.log_artifact("reports/fase_d_bad_table.csv")
        for _p in ["confusion_alpha.png", "confusion_lr2.png", "scatter_dino_em.png"]:
            mlflow.log_artifact("reports/figures/mlfigs_faseD/" + _p)
        print("mlflow faseD ok", flush=True)
    return (mlflow,)


@app.cell
def _(en_rows, mlflow, pd):
    from mlflow.tracking import MlflowClient as _MCq
    _qq = _MCq()
    def _mm(_name):
        _f = "tags.mlflow.runName = " + chr(34) + _name + chr(34)
        return _qq.search_runs(experiment_ids=["1"], filter_string=_f)[0].data.metrics
    _a = _mm("faseA_dinov2_logreg")
    _b = _mm("faseB_wrn50_patchcore")
    _c = _mm("faseC_em_efficientad")
    _dd = en_rows[en_rows.method == "alpha"]
    _dl = en_rows[en_rows.method == "lr2"]
    cmp_abcd = pd.DataFrame([
        {"config": "A dino", "recall_far5": round(_a["thr5_bad_recall"], 3), "auroc": round(_a["auroc_mean"], 3), "far": round(_a["thr5_far"], 3)},
        {"config": "B wrn50", "recall_far5": round(_b["thr5_bad_recall_mean"], 3), "auroc": round(_b["thr5_auroc_mean"], 3), "far": round(_b["thr5_far_mean"], 3)},
        {"config": "C em", "recall_far5": round(_c["thr5_bad_recall_mean"], 3), "auroc": round(_c["thr5_auroc_mean"], 3), "far": round(_c["thr5_far_mean"], 3)},
        {"config": "D alpha", "recall_far5": round(float(_dd.bad_recall.mean()), 3), "auroc": round(float(_dd.auroc.mean()), 3), "far": round(float(_dd.far.mean()), 3)},
        {"config": "D lr2", "recall_far5": round(float(_dl.bad_recall.mean()), 3), "auroc": round(float(_dl.auroc.mean()), 3), "far": round(float(_dl.far.mean()), 3)},
    ])
    import matplotlib.pyplot as _PP
    _fx, _axx = _PP.subplots(figsize=(8, 3))
    _xx = range(len(cmp_abcd))
    _axx.bar([_x - 0.2 for _x in _xx], cmp_abcd.recall_far5, width=0.4, label="recall@FAR5%")
    _axx.bar([_x + 0.2 for _x in _xx], cmp_abcd.auroc, width=0.4, label="AUROC")
    _axx.set_xticks(list(_xx), list(cmp_abcd.config), rotation=15, ha="right", fontsize=8)
    _axx.legend(fontsize=8)
    _axx.set_title("A vs B vs C vs D — MLflow + libreta")
    _axx.set_ylim(0, 1.05)
    _PP.tight_layout()
    _fx.savefig("reports/figures/mlfigs_faseD/comparativa_abcd.png", dpi=100)
    _PP.close(_fx)
    _h9 = _qq.search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + "faseD_ensemble" + chr(34))
    _r9 = sorted(_h9, key=lambda _r: _r.info.start_time)[-1].info.run_id
    with mlflow.start_run(run_id=_r9):
        mlflow.log_artifact("reports/figures/mlfigs_faseD/comparativa_abcd.png")
    cmp_abcd
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase D (ensemble dino + EfficientAD-em)" + chr(10) +
        "- alpha (0.05-0.15, EM-domina): recall 0.95, FAR 0.055, AUROC 0.980." + chr(10) +
        "- lr2: recall 0.95, FAR 0.062, AUROC 0.982." + chr(10) +
        "- Criterio: recall iguala a A (0.95), FAR a la mitad (0.113 -> 0.055), damaged 0.8-0.9." + chr(10) +
        "- EM recupera exactamente las que DINO pierde (bad_001/002 bent, bad_027 damaged); solo bad_022 se escapa a ambos." + chr(10) +
        "- Veredicto: ensemble mejor que A. Produccion: D-alpha."
    )
    return


@app.cell
def _(Path, mo):
    _order = ["comparativa_abcd.png", "scatter_dino_em.png", "confusion_alpha.png", "confusion_lr2.png"]
    _ims = [mo.image(str((Path("reports/figures/mlfigs_faseD") / _p).resolve()), width=520, caption=_p) for _p in _order if (Path("reports/figures/mlfigs_faseD") / _p).exists()]
    _rows = [mo.hstack(_ims[_k:_k + 2]) for _k in range(0, len(_ims), 2)]
    mo.vstack([mo.md(f"### Galeria D ({len(_ims)}/{len(_order)})"), *_rows])
    return


if __name__ == "__main__":
    app.run()
