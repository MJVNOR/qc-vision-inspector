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
    # Fase E — calibracion robusta de D-alpha + autopsia bad_022
    Thresholds alrededor del corte actual, solo con train. Sin tocar test ni bases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.status.toast("Paired and live!", "fase_e ready")
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
    ED2 = np.array([_d2map[_f] for _f in ELAB.filename])
    _ec = np.load("models/fase_c_oof_scores.npz")
    _emmap = {}
    for _f in range(5):
        for _i, _ss in zip(_ec[f"em_fold{_f}_idx"], _ec[f"em_fold{_f}_score"]):
            _emmap[int(_i)] = float(_ss)
    EEM = np.array([_emmap[_i] for _i in range(len(ELAB))])
    assert len(_emmap) == len(ELAB) == 313
    f"E ok d2 [{ED2.min():.3f},{ED2.max():.3f}] em [{EEM.min():.3f},{EEM.max():.3f}]"
    return ED2, EEM, EFOLDS, ELAB, EY, Path, np, pd


@app.cell
def _(ED2, EEM, EFOLDS, ELAB, EY, np, pd):
    from sklearn.metrics import precision_recall_fscore_support
    _dm = dict(zip(ELAB.filename, ELAB.defect))
    _fnames = list(ELAB.filename)
    _folddata, _foldcuts95 = {}, []
    for _fold, (_tr, _te) in enumerate(EFOLDS):
        _tr = np.array(_tr)
        _tel = [int(_x) for _x in _te]
        _gtr = _tr[EY[_tr] == 0]
        _md, _iqd = np.median(ED2[_gtr]), np.subtract(*np.percentile(ED2[_gtr], [75, 25])) + 1e-9
        _me, _iqe = np.median(EEM[_gtr]), np.subtract(*np.percentile(EEM[_gtr], [75, 25])) + 1e-9
        _dn_tr, _dn_te = (ED2[_tr] - _md) / _iqd, (ED2[_te] - _md) / _iqd
        _de_tr, _de_te = (EEM[_tr] - _me) / _iqe, (EEM[_te] - _me) / _iqe
        _gdn, _gde = _dn_tr[EY[_tr] == 0], _de_tr[EY[_tr] == 0]
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _dn_tr + (1 - _a) * _de_tr
            _ca = float(np.quantile(_a * _gdn + (1 - _a) * _gde, 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (EY[_tr] == 1)).sum() / max((EY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (EY[_tr] == 0)).sum() / max((EY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a)
        _ba = _best[1]
        _ftr = _ba * _dn_tr + (1 - _ba) * _de_tr
        _fte = _ba * _dn_te + (1 - _ba) * _de_te
        _fgt = _ftr[EY[_tr] == 0]
        _foldcuts95.append(float(np.quantile(_fgt, 0.95)))
        _folddata[_fold] = (_tel, _fgt, _fte, EY[_te])
        print(f"E fold {_fold} alpha={_ba}", flush=True)
    def _ev(_pred, _true, _tel):
        _tp = int(((_true == 1) & (_pred == 1)).sum())
        _fn = int(((_true == 1) & (_pred == 0)).sum())
        _fp = int(((_true == 0) & (_pred == 1)).sum())
        _tn = int(((_true == 0) & (_pred == 0)).sum())
        _pc, _rc, _f1, _ = precision_recall_fscore_support(_true, _pred, average="binary", zero_division=0)
        _pd = {}
        for _d in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
            _kk = [_k for _k, _ii in enumerate(_tel) if _dm[_fnames[_ii]] == _d]
            _pd[_d] = round(float(_pred[_kk].mean()), 3) if _kk else 0.0
        return {"bad_recall": round(_tp / max(_tp + _fn, 1), 3), "far": round(_fp / max(_fp + _tn, 1), 3),
            "f1": round(float(_f1), 3), **{f"def_{_k}": _v for _k, _v in _pd.items()}}
    ee_rows = []
    for _agg, _gc in [("global-mean", float(np.mean(_foldcuts95))), ("global-median", float(np.median(_foldcuts95))), ("global-max", float(np.max(_foldcuts95)))]:
        _P, _T, _I = [], [], []
        for _fold in range(5):
            _tel, _fgt, _fte, _yte = _folddata[_fold]
            _P.extend(_fte.tolist())
            _T.extend(_yte.tolist())
            _I.extend(_tel)
        _P, _T = np.array(_P), np.array(_T)
        ee_rows.append({"cut": _agg, "value": round(_gc, 4), **_ev((_P >= _gc).astype(int), _T, _I)})
    for _q in [0.93, 0.94, 0.95, 0.96, 0.97]:
        _row = {"cut": f"perfold-q{_q}", "value": "-"}
        _PP, _TT, _II, _HH = [], [], [], []
        for _fold in range(5):
            _tel, _fgt, _fte, _yte = _folddata[_fold]
            _cq = float(np.quantile(_fgt, _q))
            _HH.append(_cq)
            _PP.extend(_fte.tolist())
            _TT.extend(_yte.tolist())
            _II.extend(_tel)
        _PP, _TT = np.array(_PP), np.array(_TT)
        _row.update(_ev((_PP >= np.repeat(_HH, [len(_folddata[_f][0]) for _f in range(5)])).astype(int), _TT, _II))
        _row["value"] = round(float(np.mean(_HH)), 4)
        ee_rows.append(_row)
    ee_rows = pd.DataFrame(ee_rows)
    ee_rows
    return (ee_rows,)


@app.cell
def _(ED2, EEM, EFOLDS, ELAB, EY, Path, mo, np, pd):
    import torch as _T2
    from transformers import AutoImageProcessor as _AIP, AutoModel as _AM
    _amask = {}
    for _fold, (_tr, _te) in enumerate(EFOLDS):
        _tr = np.array(_tr)
        _gtr = _tr[EY[_tr] == 0]
        _md, _iqd = np.median(ED2[_gtr]), np.subtract(*np.percentile(ED2[_gtr], [75, 25])) + 1e-9
        _me, _iqe = np.median(EEM[_gtr]), np.subtract(*np.percentile(EEM[_gtr], [75, 25])) + 1e-9
        _dn_tr, _de_tr = (ED2[_tr] - _md) / _iqd, (EEM[_tr] - _me) / _iqe
        _gdn, _gde = _dn_tr[EY[_tr] == 0], _de_tr[EY[_tr] == 0]
        _best, _ba = None, 0.1
        for _a in [round(_x * 0.05, 2) for _x in range(21)]:
            _f = _a * _dn_tr + (1 - _a) * _de_tr
            _ca = float(np.quantile(_a * _gdn + (1 - _a) * _gde, 0.95))
            _p = (_f >= _ca).astype(int)
            _rc = ((_p == 1) & (EY[_tr] == 1)).sum() / max((EY[_tr] == 1).sum(), 1)
            _fa = ((_p == 1) & (EY[_tr] == 0)).sum() / max((EY[_tr] == 0).sum(), 1)
            _key = (0 if _fa <= 0.05 else 1, -_rc, _fa)
            if _best is None or _key < _best[0]:
                _best = (_key, _a, _ca)
        _, _ba, _bca = _best
        _dn_te = (ED2[np.array(_te)] - _md) / _iqd
        _de_te = (EEM[np.array(_te)] - _me) / _iqe
        _fte = _ba * _dn_te + (1 - _ba) * _de_te
        for _ii, _sv in zip([int(_x) for _x in _te], _fte):
            _amask[_ii] = (float(_sv), int(_sv >= _bca))
    _proc = _AIP.from_pretrained("facebook/dinov2-base")
    _bm = _AM.from_pretrained("facebook/dinov2-base")
    _bm.eval()
    for _p in _bm.parameters():
        _p.requires_grad_(False)
    _dev = _T2.device("cuda" if _T2.cuda.is_available() else "cpu")
    _bm.to(_dev)
    def _cls(_idx):
        _r = ELAB.iloc[int(_idx)]
        _im = _PIe.open(Path("data/processed/transistor_binary") / ("good" if _r["label"] == "good" else "bad") / _r["filename"]).convert("RGB")
        _inp = _proc(images=_im, return_tensors="pt")
        with _T2.inference_mode():
            return _bm(**{k: v.to(_dev) for k, v in _inp.items()}).last_hidden_state[:, 0].cpu().numpy()[0]
    from PIL import Image as _PIe
    _C = np.stack([_cls(_i) for _i in range(len(ELAB)) if EY[_i] == 0]).mean(axis=0)
    _rows = []
    for _n in [f"bad_02{_d}.png" for _d in range(10)]:
        _ii = int(ELAB[ELAB.filename == _n].index[0])
        _r = ELAB.iloc[_ii]
        _mp = Path("data/raw/mvtec_anomaly_detection/transistor/ground_truth/damaged_case") / (_r["original_path"].split("/")[-1].replace(".png", "_mask.png"))
        _mk = np.asarray(_PIe.open(_mp).convert("L")) > 127
        _im = _PIe.open(Path("data/processed/transistor_binary/bad") / _n).convert("RGB")
        _g = np.asarray(_im.convert("L"), dtype=np.float32)
        _z = _cls(_ii)
        _dist = float(np.linalg.norm(_z - _C))
        _rows.append({"file": _n, "mask_area_pct": round(100 * _mk.mean(), 2), "bright": round(_g.mean(), 1),
            "contrast": round(_g.std(), 1), "dino": round(float(ED2[_ii]), 3), "em": round(float(EEM[_ii]), 3),
            "fused": round(_amask[_ii][0], 2), "caught": bool(_amask[_ii][1]), "cls_dist": round(_dist, 2)})
    ee_aut = pd.DataFrame(_rows)
    _b22 = _PIe.open(Path("data/processed/transistor_binary/bad/bad_022.png")).convert("RGB").resize((320, 320))
    _m22 = _PIe.open(Path("data/raw/mvtec_anomaly_detection/transistor/ground_truth/damaged_case/002_mask.png")).convert("L").resize((320, 320))
    mo.vstack([mo.md("### bad_022 vs damaged_case (area de mascara %, brillo, distancia CLS, veredicto fused)"), mo.hstack([mo.image(_b22, width=280, caption="bad_022"), mo.image(_m22, width=280, caption="mask 002")]), ee_aut])
    return (ee_aut,)


@app.cell
def _(ee_aut, ee_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import mlflow
    from mlflow.tracking import MlflowClient as _MCe
    from pathlib import Path as _Pe
    _fe = _Pe("reports/figures/mlfigs_faseE")
    _fe.mkdir(exist_ok=True)
    _xs = list(range(len(ee_rows)))
    _fg, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(_xs, ee_rows.bad_recall, "o-", label="recall")
    _ax.plot(_xs, ee_rows.far, "s-", label="far")
    _ax.set_xticks(_xs, list(ee_rows.cut), rotation=20, ha="right", fontsize=7)
    _ax.axhline(0.05, color="red", lw=1, ls="--", label="FAR 5%")
    _ax.legend(fontsize=8)
    _ax.set_title("E1 sweep de cortes (pooled OOF, train-only)")
    _fg.savefig(_fe / "sweep_cuts.png", dpi=100)
    plt.close(_fg)
    _b22i = Image.open(_Pe("data/processed/transistor_binary/bad/bad_022.png")).convert("RGB").resize((280, 280))
    _m22i = Image.open(_Pe("data/raw/mvtec_anomaly_detection/transistor/ground_truth/damaged_case/002_mask.png")).convert("RGB").resize((280, 280))
    _sh = Image.new("RGB", (560, 300), "white")
    _sh.paste(_b22i, (0, 20))
    _sh.paste(_m22i, (280, 20))
    _sh.save(_fe / "autopsia_bad022.png")
    ee_rows.to_csv("reports/fase_e_sweep.csv", index=False)
    ee_aut.to_csv("reports/fase_e_autopsia.csv", index=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    _rn = "faseE_calibracion"
    _hits = _MCe().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + _rn + chr(34))
    _cm = mlflow.start_run(run_id=sorted(_hits, key=lambda _r: _r.info.start_time)[-1].info.run_id) if _hits else mlflow.start_run(run_name=_rn)
    with _cm:
        mlflow.log_params({"base": "D-alpha-frozen", "objective": "recall>=0.95,FAR<=0.05", "cands": 8,
            "verdict": "keep-5.5pct", "bad022_mask_pct": 0.63, "bad022_hyp": "defecto-mas-pequeno-diluye-score"})
        mlflow.log_artifact("reports/fase_e_sweep.csv")
        mlflow.log_artifact("reports/fase_e_autopsia.csv")
        mlflow.log_artifact("reports/figures/mlfigs_faseE/sweep_cuts.png")
        mlflow.log_artifact("reports/figures/mlfigs_faseE/autopsia_bad022.png")
        print("mlflow faseE ok", flush=True)
    return


@app.cell
def _(mo):
    mo.md(
        "### Conclusion fase E" + chr(10) +
        "- E1: ningun corte logra recall>=0.95 con FAR<=0.05. Se conserva perfold-q0.95: recall 0.95, FAR 0.055." + chr(10) +
        "- damaged_case = 0.8 en los 8 candidatos: no empeora con ningun corte." + chr(10) +
        "- 5.5 pct FAR queda documentado como mejor operating point validado." + chr(10) +
        "- E2: bad_022 tiene la mascara mas pequena (0.63 pct) y distancia CLS en rango GOOD: el defecto se diluye. Siguiente mejora: scoring sensible a defectos pequenos / mapas de mayor resolucion."
    )
    return


@app.cell
def _(Path, mo):
    _order = ["sweep_cuts.png", "autopsia_bad022.png"]
    _ims = [mo.image(str((Path("reports/figures/mlfigs_faseE") / _p).resolve()), width=560, caption=_p) for _p in _order if (Path("reports/figures/mlfigs_faseE") / _p).exists()]
    mo.vstack([mo.md(f"### Galeria E ({len(_ims)}/{len(_order)})"), mo.hstack(_ims)])
    return


if __name__ == "__main__":
    app.run()
