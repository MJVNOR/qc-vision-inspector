import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Fase A — backbone frozen + clasificador lineal (baseline supervisado)
    Backbone actual: **DINOv3** (`btag`=dinov3b (Small quedo como dinov3): cache, runs y artefactos separados de v2).
    Objetivo: maximizar deteccion de malas sin overfitting, con 273 good / 40 bad.
    Metrica guia: **recall de bad** con FAR controlado; AUROC solo para ranking.
    """)
    return


@app.cell
def md_modelo(mo):
    mo.md("""
    ### 1. Backbone congelado
    DINOv3-ViT/B en GPU, todo `requires_grad_(False)`: solo extrae features,
    aqui no se entrena nada visual. Salida por imagen: pooler (768-d).
    """)
    return


@app.cell
def modelo():

    import torch
    from transformers import AutoImageProcessor, AutoModel

    backbone_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    btag = "dinov3b" if "vitb16" in backbone_id else ("dinov3" if "dinov3" in backbone_id else "dinov2")  # tag para cache, runs y artefactos
    processor = AutoImageProcessor.from_pretrained(backbone_id)
    backbone = AutoModel.from_pretrained(backbone_id)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    # GPU si hay, si no CPU (extraccion congelada: una sola pasada)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    (backbone_id, str(device))
    return AutoModel, backbone, backbone_id, btag, device, processor, torch


@app.cell
def md_embeddings(mo):
    mo.md("""
    ### 2. Embeddings + cache
    Una pasada por las 313 imagenes (1024px) y se guarda `fase_a_emb_{btag}.pt` (una por backbone).
    Re-correr esta celda carga del cache en segundos.
    """)
    return


@app.cell
def embeddings(backbone, btag, device, processor, torch):

    from PIL import Image
    import pandas as pd
    from pathlib import Path
    import numpy as np

    base = Path("transistor_binary")
    labels = pd.read_csv(base / "labels.csv")
    cache = Path(f"fase_a_emb_{btag}.pt")  # cache por backbone: v2 y v3 no se mezclan

    def _path(_r):
        return base / ("good" if _r["label"] == "good" else "bad") / _r["filename"]

    def _embed(_im):
        _inp = processor(images=_im, return_tensors="pt")
        _out = backbone(**{k: v.to(device) for k, v in _inp.items()})
        # pooler si existe (DINOv3), si no CLS (DINOv2)
        _p = getattr(_out, "pooler_output", None)
        return (_p if _p is not None else _out.last_hidden_state[:, 0]).cpu()

    if cache.exists():
        _saved = torch.load(cache, weights_only=False)
        X = _saved["X"]
        fnames = _saved["fnames"]
        print("cache:", X.shape)
    else:
        fnames, feats = [], []
        with torch.inference_mode():
            for _i, _row in labels.iterrows():
                _im = Image.open(_path(_row)).convert("RGB")
                feats.append(_embed(_im))
                fnames.append(_row["filename"])
                if (_i + 1) % 50 == 0:
                    print(f"{_i + 1}/313")
        X = torch.cat(feats).numpy()
        torch.save({"X": X, "fnames": fnames}, cache)
    X.shape
    return Image, Path, X, base, fnames, labels, np, pd


@app.cell
def md_eval(mo):
    mo.md("""
    ### 3. Baseline: LogReg 5-fold estratificado
    `class_weight=balanced` compensa el 87/13. Threshold Youden (max tpr−fpr)
    se calibra **dentro** del train de cada fold, nunca en test. Seed 42 fija.
    """)
    return


@app.cell
def eval(X, labels, pd):

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

    y = (labels["label"] == "bad").to_numpy(int)
    # 5-fold estratificado (8 bad por fold: estimaciones ruidosas, ver std)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)
        clf.fit(X[tr], y[tr])
        p_tr = clf.predict_proba(X[tr])[:, 1]
        fpr_tr, tpr_tr, thr = roc_curve(y[tr], p_tr)
        cut = float(thr[(tpr_tr - fpr_tr).argmax()])
        p_te = clf.predict_proba(X[te])[:, 1]
        pred = (p_te >= cut).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary", zero_division=0)
        tp = int(((y[te] == 1) & (pred == 1)).sum())
        tn = int(((y[te] == 0) & (pred == 0)).sum())
        fp = int(((y[te] == 0) & (pred == 1)).sum())
        fn = int(((y[te] == 1) & (pred == 0)).sum())
        fpr, tpr, _ = roc_curve(y[te], p_te)
        fold_rows.append({"fold": fold, "cut": round(cut, 4),
            "bad_recall": round(rec, 3), "far": round(fp / max(fp + tn, 1), 3),
            "frr": round(fn / max(fn + tp, 1), 3), "precision": round(float(prec), 3),
            "f1": round(float(f1), 3), "auroc": round(float(auc(fpr, tpr)), 3)})
    results = pd.DataFrame(fold_rows)
    results
    return (
        LogisticRegression,
        StratifiedKFold,
        auc,
        precision_recall_fscore_support,
        results,
        roc_curve,
        y,
    )


@app.cell
def resumen(mo, results):
    summary = results.describe().loc[["mean", "std"]].round(3)
    # Solo media±std: con 8 bad por fold, la std importa mas que la media.
    mo.vstack([mo.md("### Baseline Youden: media ± std (5-fold, corte calibrado en train)"), summary])
    return


@app.cell
def md_mlflow(mo):
    mo.md("""
    ### 4. Registro en MLflow
    Params, metricas por fold + media/std y CSV adjunto en `transistor-binary`.
    Backend sqlite (`mlflow.db`).
    """)
    return


@app.cell
def mlflow_log(X, backbone_id, btag, device, results):

    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    with mlflow.start_run(run_name=f"faseA_{btag}_logreg"):
        mlflow.log_params({"backbone": backbone_id, "embedding": "cls", "dim": int(X.shape[1]),
            "clf": "LogisticRegression", "C": 1.0, "class_weight": "balanced",
            "cv": "StratifiedKFold-5", "seed": 42, "threshold": "youden-trainfold",
            "device": str(device), "n_good": 273, "n_bad": 40})
        for _t in results.itertuples():
            for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
                mlflow.log_metric(f"{_m}_fold{int(_t.fold)}", float(getattr(_t, _m)))
        for _m in ["bad_recall", "far", "frr", "precision", "f1", "auroc"]:
            mlflow.log_metric(f"{_m}_mean", float(results[_m].mean()))
            mlflow.log_metric(f"{_m}_std", float(results[_m].std()))
        results.to_csv(f"fase_a_fold_metrics_{btag}.csv", index=False)
        mlflow.log_artifact(f"fase_a_fold_metrics_{btag}.csv")
        print("mlflow run ok")
    return (mlflow,)


@app.cell
def md_threshold(mo):
    mo.md("""
    ### 5. Threshold orientado a recall
    Youden equilibra clases, pero aqui importa cazar malas: corte por cuantil
    de scores good de train (q95 → FAR≈5%, q99 → FAR≈1%), aplicado al test-fold.
    """)
    return


@app.cell
def threshold(
    LogisticRegression,
    StratifiedKFold,
    X,
    auc,
    np,
    pd,
    precision_recall_fscore_support,
    roc_curve,
    y,
):

    # Corte por cuantil de scores GOOD de train: el cuantil 95 deja 5% de buenas
    # arriba del corte en train (FAR objetivo), sin mirar jamas el test.
    _thr_rows = []
    _thr_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _fold, (_tr, _te) in enumerate(_thr_skf.split(X, y)):
        _clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)
        _clf.fit(X[_tr], y[_tr])
        _p_tr = _clf.predict_proba(X[_tr])[:, 1]
        _p_te = _clf.predict_proba(X[_te])[:, 1]
        for _ft in [0.05, 0.01]:
            _cut = float(np.quantile(_p_tr[y[_tr] == 0], 1 - _ft))
            _pred = (_p_te >= _cut).astype(int)
            _tp = int(((y[_te] == 1) & (_pred == 1)).sum())
            _fn = int(((y[_te] == 1) & (_pred == 0)).sum())
            _fp = int(((y[_te] == 0) & (_pred == 1)).sum())
            _tn = int(((y[_te] == 0) & (_pred == 0)).sum())
            _prec, _rec, _f1, _x = precision_recall_fscore_support(y[_te], _pred, average="binary", zero_division=0)
            _fpr, _tpr, _z = roc_curve(y[_te], _p_te)
            _thr_rows.append({"fold": _fold, "far_obj": _ft, "cut": round(_cut, 4),
                "bad_recall": round(float(_rec), 3), "far": round(_fp / max(_fp + _tn, 1), 3),
                "precision": round(float(_prec), 3), "f1": round(float(_f1), 3),
                "auroc": round(float(auc(_fpr, _tpr)), 3)})
    thr_results = pd.DataFrame(_thr_rows)
    thr_results.groupby("far_obj")[["bad_recall", "far", "precision", "f1", "auroc"]].agg(["mean", "std"]).round(3)
    return (thr_results,)


@app.cell
def oof(LogisticRegression, StratifiedKFold, X, thr_results, y):
    from sklearn.model_selection import cross_val_predict

    # Scores out-of-fold compartidos: cada imagen se predice con un modelo
    # que NO la vio en train (5-fold, seed fija para reproducibilidad).
    oof_scores = cross_val_predict(
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000),
        X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    # Corte global = media de los cortes por fold calibrados a FAR 5% en train.
    cut_far5 = float(thr_results[thr_results.far_obj == 0.05]["cut"].mean())
    return cut_far5, oof_scores


@app.cell
def md_optuna(mo):
    mo.md("""
    ### 6. Tuning con Optuna (20 trials)
    Objetivo: recall medio con FAR de train ≤5%, C de LogReg en [1e-3, 1e2].
    Resultado: C insensible (meseta ~0.95; v3b 0.975).
    """)
    return


@app.cell
def optuna_tune(
    LogisticRegression,
    StratifiedKFold,
    X,
    backbone_id,
    btag,
    mlflow,
    np,
    y,
):

    import optuna

    def _objective(_trial):
        _c = _trial.suggest_float("C", 1e-3, 1e2, log=True)
        _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _recs = []
        for _tr, _te in _skf.split(X, y):
            _cl = LogisticRegression(class_weight="balanced", C=_c, max_iter=5000)
            _cl.fit(X[_tr], y[_tr])
            _ptr = _cl.predict_proba(X[_tr])[:, 1]
            _cut = float(np.quantile(_ptr[y[_tr] == 0], 0.95))
            _pte = _cl.predict_proba(X[_te])[:, 1]
            _pr = (_pte >= _cut).astype(int)
            _tp = int(((y[_te] == 1) & (_pr == 1)).sum())
            _fn = int(((y[_te] == 1) & (_pr == 0)).sum())
            _recs.append(_tp / max(_tp + _fn, 1))
        return float(np.mean(_recs))

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=20)
    best_params = study.best_params
    best_value = float(study.best_value)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    with mlflow.start_run(run_name=f"faseA_{btag}_logreg_opt"):
        mlflow.log_params({"backbone": backbone_id, "clf": "LogReg", "tune": "C log[1e-3,1e2]",
            "trials": 20, "objective": "recall@trainFAR5%", "threshold": "quantile-neg-95"})
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("best_recall_at_far5", best_value)
        study.trials_dataframe().to_csv(f"fase_a_optuna_{btag}.csv", index=False)
        mlflow.log_artifact(f"fase_a_optuna_{btag}.csv")
        print("best:", best_params, round(best_value, 3))
    return (study,)


@app.cell
def dist_scores(cut_far5, oof_scores, pd, y):
    import altair as alt

    # Reusa los scores OOF centrales (celda oof): no recalcular aqui.
    _scores = pd.DataFrame({"score": oof_scores, "label": ["bad" if _v else "good" for _v in y]})
    _dist_base = alt.Chart(_scores).mark_bar(opacity=0.6).encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="P(bad) out-of-fold"),
        y=alt.Y("count():Q", title="imagenes"),
        color=alt.Color("label:N", title="clase"),
    ).properties(title="Scores good vs bad + corte FAR5% (media de folds)", width=450)
    _dist_rule = alt.Chart(pd.DataFrame({"cut": [cut_far5]})).mark_rule(color="red", strokeWidth=2).encode(x="cut:Q")
    _dist_base + _dist_rule
    return (alt,)


@app.cell
def errores(alt, base, cut_far5, fnames, mo, oof_scores, pd, y):
    _pred2 = (oof_scores >= cut_far5).astype(int)
    _err = pd.DataFrame({"filename": list(fnames), "score": list(oof_scores)})
    _err["real"] = ["bad" if _v else "good" for _v in y]
    _err["pred"] = ["bad" if _v else "good" for _v in _pred2]
    _cm = _err.groupby(["real", "pred"]).size().reset_index(name="n")
    _cm_chart = alt.Chart(_cm).mark_rect().encode(
        x=alt.X("pred:N", title="predicha"),
        y=alt.Y("real:N", title="real"),
        color=alt.Color("n:Q", title="n"),
    ).properties(title="Matriz real vs predicha (corte FAR5%)", width=250, height=250)
    _cm_text = alt.Chart(_cm).mark_text(fontSize=20).encode(
        x="pred:N", y="real:N", text="n:Q",
    )
    _fn = _err[(_err.real == "bad") & (_err.pred == "good")]
    _fp = _err[(_err.real == "good") & (_err.pred == "bad")]
    def _gal(_df, _tag):
        _ims = []
        for _, _r in _df.head(8).iterrows():
            _folder = "bad" if _r["real"] == "bad" else "good"
            _ims.append(mo.image(str((base / _folder / _r["filename"]).resolve()), width=160,
                caption=f"{_r['filename']} real:{_r['real']} pred:{_r['pred']} s={_r['score']:.2f}"))
        return mo.vstack([mo.md(f"### {_tag} ({len(_df)})"), mo.hstack(_ims)] if _ims else [mo.md(f"### {_tag} (0)")])
    mo.vstack([
        mo.hstack([_cm_chart + _cm_text, mo.md(f"Falsas buenas (se escapan): {len(_fn)} | Falsas malas (rechazo injusto): {len(_fp)}")]),
        _gal(_fn, "Perdidas: bad reales clasificadas good"),
        _gal(_fp, "Falsas alarmas: good reales clasificadas bad (primeras 8)"),
    ])
    return


@app.cell
def rollout(
    AutoModel,
    Image,
    backbone,
    backbone_id,
    base,
    cut_far5,
    fnames,
    mo,
    oof_scores,
    processor,
    torch,
    y,
):
    # Attention rollout sobre el ultimo bloque: a que parches miro el CLS.
    # Sin gradientes, determinista; solo para las perdidas + 4 falsas alarmas.
    import matplotlib.pyplot as _plt
    # SDPA no devuelve atenciones: copia eager en CPU solo para rollout (pesos ya en cache).
    _bm_eager = AutoModel.from_pretrained(backbone_id, attn_implementation="eager")
    _bm_eager.eval()
    def _rollout(_img):
        _inp = processor(images=_img, return_tensors="pt")
        with torch.inference_mode():
            _o = _bm_eager(_inp["pixel_values"], output_attentions=True)
        _atts = [( _a.mean(dim=1).squeeze(0) + torch.eye(_a.size(-1))) for _a in _o.attentions]
        _atts = [_a / _a.sum(dim=-1, keepdim=True) for _a in _atts]
        _j = _atts[0]
        for _a in _atts[1:]:
            _j = torch.matmul(_a, _j)
        _reg = int(getattr(backbone.config, "num_register_tokens", 0) or 0)
        _v = _j[0, 1 + _reg:]
        _g = int(round(_v.numel() ** 0.5))
        _h = _v.reshape(_g, _g).cpu().numpy()
        _h = (_h - _h.min()) / max(_h.max() - _h.min(), 1e-8)
        return _h
    _pred = (oof_scores >= cut_far5).astype(int)
    _sel = [fnames[i] for i in range(len(y)) if y[i] == 1 and _pred[i] == 0]
    _sel += [fnames[i] for i in range(len(y)) if y[i] == 0 and _pred[i] == 1][:4]
    _rows = []
    for _ff in _sel:
        _im0 = Image.open(base / ("bad" if _ff.startswith("bad") else "good") / _ff).convert("RGB")
        _h = _rollout(_im0)
        _cm = (_plt.get_cmap("jet")(_h)[:, :, :3] * 255).astype("uint8")
        _cm = Image.fromarray(_cm).resize((160, 160))
        _th = _im0.resize((160, 160))
        from PIL import Image as _I2
        _blend = _I2.blend(_th, _cm, 0.45)
        _rows.append(mo.image(_blend, width=160, caption=f"{_ff} rollout"))
    mo.vstack([mo.md("### Attention rollout en errores (rojo = miro ahi)"), mo.hstack(_rows)])
    return


@app.cell
def figs_log(
    LogisticRegression,
    StratifiedKFold,
    X,
    base,
    btag,
    cut_far5,
    fnames,
    labels,
    mlflow,
    mo,
    np,
    oof_scores,
    pd,
    thr_results,
    y,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from mlflow.tracking import MlflowClient
    from PIL import Image as _PILImage, ImageDraw as _Draw

    # --- datos compartidos ya calculados: oof_scores, cut_far5, X, y, thr_results
    _oof_pred = (oof_scores >= cut_far5).astype(int)
    _def_of = dict(zip(labels.filename, labels.defect))
    # recall por defecto al corte FAR5%
    _per_def = {}
    for _d in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
        _idx = [i for i, _f in enumerate(fnames) if _def_of[_f] == _d]
        _tp = int(sum(1 for i in _idx if _oof_pred[i] == 1))
        _per_def[_d] = round(_tp / max(len(_idx), 1), 3)

    from pathlib import Path as _Path
    _figdir = _Path(f"mlfigs_{btag}")
    _figdir.mkdir(exist_ok=True)

    # ROC por fold + PR
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4))
    _rskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _f, (_tr, _te) in enumerate(_rskf.split(X, y)):
        _c = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X[_tr], y[_tr])
        _s = _c.predict_proba(X[_te])[:, 1]
        RocCurveDisplay.from_predictions(y[_te], _s, ax=_ax[0], name=f"fold {_f}")
        _ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    from sklearn.metrics import PrecisionRecallDisplay
    for _f, (_tr, _te) in enumerate(_rskf.split(X, y)):
        _c = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X[_tr], y[_tr])
        PrecisionRecallDisplay.from_predictions(y[_te], _c.predict_proba(X[_te])[:, 1], ax=_ax[1], name=f"fold {_f}")
    _fig.suptitle("ROC y PR por fold (C=1.0)")
    _fig.savefig(_figdir / "roc_pr.png", dpi=100)
    plt.close(_fig)

    # matriz de confusion
    _fig2, _ax2 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, _oof_pred, display_labels=["good", "bad"], ax=_ax2)
    _ax2.set_title("Confusion corte FAR5%")
    _fig2.savefig(_figdir / "confusion.png", dpi=100)
    plt.close(_fig2)

    # histograma de scores
    _fig3, _ax3 = plt.subplots()
    _ax3.hist([oof_scores[y == 0], oof_scores[y == 1]], bins=30, label=["good", "bad"], alpha=0.6)
    _ax3.axvline(cut_far5, color="red", lw=2, label="corte FAR5%")
    _ax3.legend()
    _ax3.set_title("Scores OOF good vs bad")
    _fig3.savefig(_figdir / "scores_hist.png", dpi=100)
    plt.close(_fig3)

    # galeria de errores: FN todas + 8 FP, contact sheet PIL
    _fn_files = [fnames[i] for i in range(len(y)) if y[i] == 1 and _oof_pred[i] == 0]
    _fp_files = [fnames[i] for i in range(len(y)) if y[i] == 0 and _oof_pred[i] == 1][:8]
    _thumbs = []
    for _ff in _fn_files + _fp_files:
        _im = _PILImage.open(base / ("bad" if _ff.startswith("bad") else "good") / _ff).convert("RGB").resize((160, 160))
        _thumbs.append((_ff, _im))
    _sheet = _PILImage.new("RGB", (160 * max(len(_thumbs), 1), 180), "white")
    _dr = _Draw.Draw(_sheet)
    for _k, (_ff, _im) in enumerate(_thumbs):
        _sheet.paste(_im, (_k * 160, 20))
        _dr.text((_k * 160 + 4, 2), _ff, fill="black")
    _sheet.save(_figdir / "errores.png")

    # strip por defecto (faltaba en MLflow): las perdidas (FN) se ven bajo el corte
    _fig4, _ax4 = plt.subplots(figsize=(8, 4))
    _dorder = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]
    for _k, _d in enumerate(_dorder):
        _vv = oof_scores[[i for i, _f in enumerate(fnames) if _def_of[_f] == _d]]
        _rng = __import__("numpy").random.default_rng(_k)
        _ax4.scatter(_rng.normal(_k, 0.08, len(_vv)), _vv, s=8, alpha=0.6)
    _ax4.set_xticks(range(len(_dorder)), _dorder)
    _ax4.axhline(cut_far5, color="red", lw=2, label="corte FAR5%")
    _ax4.legend()
    _ax4.set_title("Scores por defecto + corte FAR5%")
    _fig4.savefig(_figdir / "por_defecto.png", dpi=100)
    plt.close(_fig4)

    # t-SNE exploratorio como artefacto (import diferido: TSNE ya lo define la celda tsne)
    _Z = __import__("sklearn.manifold", fromlist=["TSNE"]).TSNE(n_components=2, perplexity=30, random_state=42, init="random").fit_transform(X)
    _fig5, _ax5 = plt.subplots()
    for _d in _dorder:
        _ii = [i for i, _f in enumerate(fnames) if _def_of[_f] == _d]
        _ax5.scatter(_Z[_ii, 0], _Z[_ii, 1], s=10, alpha=0.7, label=_d)
    _ax5.legend(markerscale=2, fontsize=8)
    _ax5.set_title("t-SNE embeddings (exploratorio)")
    _fig5.savefig(_figdir / "tsne.png", dpi=100)
    plt.close(_fig5)

    # barrido de corte como artefacto
    _sw_rows, _ss = [], StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _f, (_tr, _te) in enumerate(_ss.split(X, y)):
        _cc = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X[_tr], y[_tr])
        _a, _b = _cc.predict_proba(X[_tr])[:, 1], _cc.predict_proba(X[_te])[:, 1]
        for _q in [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]:
            _cut = float(np.quantile(_a[y[_tr] == 0], _q))
            _pp = (_b >= _cut).astype(int)
            _tp = int(((y[_te] == 1) & (_pp == 1)).sum())
            _fn = int(((y[_te] == 1) & (_pp == 0)).sum())
            _fp = int(((y[_te] == 0) & (_pp == 1)).sum())
            _tn = int(((y[_te] == 0) & (_pp == 0)).sum())
            _sw_rows.append({"q": _q, "recall": _tp / max(_tp + _fn, 1), "far": _fp / max(_fp + _tn, 1)})
    _sw = pd.DataFrame(_sw_rows).groupby("q")[["recall", "far"]].mean()
    _fig6, _ax6 = plt.subplots()
    _ax6.plot(_sw.index, _sw.recall, "o-", label="recall")
    _ax6.plot(_sw.index, _sw.far, "s-", label="far")
    _ax6.legend()
    _ax6.set_title("Barrido del corte (media 5-fold)")
    _fig6.savefig(_figdir / "sweep.png", dpi=100)
    plt.close(_fig6)

    # --- log al run baseline existente
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _client = MlflowClient()
    _hits = _client.search_runs(experiment_ids=["1"], filter_string=f"tags.mlflow.runName = 'faseA_{btag}_logreg'")
    _rid = _hits[0].info.run_id
    with mlflow.start_run(run_id=_rid):
        _t5 = thr_results[thr_results.far_obj == 0.05][["bad_recall", "far", "precision", "f1"]].mean(numeric_only=True)
        _t1 = thr_results[thr_results.far_obj == 0.01][["bad_recall", "far", "precision", "f1"]].mean(numeric_only=True)
        mlflow.log_metrics({f"thr5_{_k}": float(_v) for _k, _v in _t5.items()})
        mlflow.log_metrics({f"thr1_{_k}": float(_v) for _k, _v in _t1.items()})
        mlflow.log_metrics({f"recall_def_{_k}": _v for _k, _v in _per_def.items()})
        for _p in ["roc_pr.png", "confusion.png", "scores_hist.png", "errores.png", "por_defecto.png", "tsne.png", "sweep.png"]:
            mlflow.log_artifact(str(_figdir / _p))
        print("figs log ok en", _rid[:8], _per_def)
    mo.md(f"### Artefactos registrados: ROC/PR, confusion, histograma, galeria ({len(_fn_files)} FN + {len(_fp_files)} FP) y recall por defecto {_per_def}")
    return


@app.cell
def modelo_final(
    LogisticRegression,
    Path,
    X,
    backbone_id,
    btag,
    cut_far5,
    mlflow,
    mo,
    y,
):
    from mlflow import sklearn as _mlsk
    from huggingface_hub import model_info as _mi
    import json as _json

    # Refit en 100% de datos con la config evaluada (C=1.0). El corte de
    # produccion es cut_far5 (media de folds, ya evaluado), no el cuantil in-sample.
    _final = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X, y)
    _ptr = _final.predict_proba(X)[:, 1]
    _train_rec = float(((_ptr >= cut_far5)[y == 1]).mean())
    _train_far = float(((_ptr >= cut_far5)[y == 0]).mean())
    _rev = _mi(backbone_id).sha
    _thr = {"cut_produccion": cut_far5, "regla": "media de cortes calibrados a FAR5% en train-folds",
        "backbone": backbone_id, "backbone_rev": _rev, "C": 1.0}
    Path(f"thresholds_{btag}.json").write_text(_json.dumps(_thr, indent=1))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("transistor-binary")
    with mlflow.start_run(run_name=f"faseA_{btag}_final"):
        mlflow.log_params({"backbone": backbone_id, "backbone_rev": _rev[:12], "C": 1.0,
            "trained_on": "full-313", "cut": cut_far5, "n_good": 273, "n_bad": 40})
        mlflow.log_metrics({"train_recall_at_cut": round(_train_rec, 3), "train_far_at_cut": round(_train_far, 3)})
        _mlsk.log_model(_final, f"logreg_{btag}", registered_model_name="transistor-logreg")
        mlflow.log_artifact(f"thresholds_{btag}.json")
        print("final ok, train recall:", round(_train_rec, 3))
    mo.md(f"### Modelo final registrado (`transistor-logreg`): train recall {_train_rec:.3f}, FAR {_train_far:.3f}, corte {cut_far5:.4f}")
    return


@app.cell
def curvas(
    LogisticRegression,
    StratifiedKFold,
    X,
    alt,
    mo,
    np,
    pd,
    results,
    roc_curve,
    y,
):
    from sklearn.metrics import precision_recall_curve

    # Curvas por fold con el mismo seed del eval: cada fold aporta su ROC y PR.
    _roc_rows, _pr_rows = [], []
    _rskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _f, (_tr, _te) in enumerate(_rskf.split(X, y)):
        _c = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X[_tr], y[_tr])
        _s = _c.predict_proba(X[_te])[:, 1]
        _fpr, _tpr, _th = roc_curve(y[_te], _s)
        _rc, _pc, _th2 = precision_recall_curve(y[_te], _s)
        for _a, _b in zip(_fpr, _tpr):
            _roc_rows.append({"fold": _f, "fpr": float(_a), "tpr": float(_b)})
        for _a, _b in zip(_rc, _pc):
            _pr_rows.append({"fold": _f, "recall": float(_a), "precision": float(_b)})
    _roc = pd.DataFrame(_roc_rows)
    _pr = pd.DataFrame(_pr_rows)
    # ROC media por interpolacion en grid comun de FPR.
    _grid = np.linspace(0, 1, 100)
    _mean_tpr = np.mean([np.interp(_grid, _roc[_roc.fold == _f].fpr, _roc[_roc.fold == _f].tpr) for _f in range(5)], axis=0)
    _mean_roc = pd.DataFrame({"fpr": _grid, "tpr": _mean_tpr})
    _roc_chart = (alt.Chart(_roc).mark_line(opacity=0.35).encode(x=alt.X("fpr:Q", title="FPR"), y=alt.Y("tpr:Q", title="TPR"), color="fold:N")
        + alt.Chart(_mean_roc).mark_line(color="black", strokeWidth=2).encode(x="fpr:Q", y="tpr:Q")
        + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q"))
    _pr_chart = alt.Chart(_pr).mark_line(opacity=0.5).encode(x=alt.X("recall:Q"), y=alt.Y("precision:Q"), color="fold:N")
    mo.vstack([
        mo.md(f"### ROC (AUROC medio {float(results.auroc.mean()):.3f}) y Precision-Recall por fold"),
        mo.hstack([_roc_chart.properties(width=300, height=300), _pr_chart.properties(width=300, height=300)]),
    ])
    return


@app.cell
def sweep(LogisticRegression, StratifiedKFold, X, alt, mo, np, pd, y):
    # Barrido del cuantil de corte: mismo modelo por fold, distinto corte.
    # Muestra el trueque recall vs FAR/precision para elegir el punto de operacion.
    _sw_rows = []
    _sskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _f, (_tr, _te) in enumerate(_sskf.split(X, y)):
        _c = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(X[_tr], y[_tr])
        _str_, _ste = _c.predict_proba(X[_tr])[:, 1], _c.predict_proba(X[_te])[:, 1]
        for _q in [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]:
            _cut = float(np.quantile(_str_[y[_tr] == 0], _q))
            _pr2 = (_ste >= _cut).astype(int)
            _tp = int(((y[_te] == 1) & (_pr2 == 1)).sum())
            _fn = int(((y[_te] == 1) & (_pr2 == 0)).sum())
            _fp = int(((y[_te] == 0) & (_pr2 == 1)).sum())
            _tn = int(((y[_te] == 0) & (_pr2 == 0)).sum())
            _sw_rows.append({"q": _q, "cut": round(_cut, 3),
                "recall": _tp / max(_tp + _fn, 1), "far": _fp / max(_fp + _tn, 1),
                "precision": _tp / max(_tp + _fp, 1)})
    _sweep = pd.DataFrame(_sw_rows).groupby("q")[["recall", "far", "precision"]].mean().reset_index()
    _melt = _sweep.melt(id_vars=["q"], value_vars=["recall", "far", "precision"], var_name="metrica", value_name="v")
    mo.vstack([
        mo.md("### Barrido del corte: recall vs FAR/precision (media 5-fold)"),
        alt.Chart(_melt).mark_line(point=True).encode(
            x=alt.X("q:Q", title="cuantil de corte (scores good train)"),
            y=alt.Y("v:Q", title="valor"), color="metrica:N").properties(width=450),
        _sweep.round(3),
    ])
    return


@app.cell
def por_defecto(alt, cut_far5, fnames, labels, mo, oof_scores, pd, y):
    # Scores OOF por tipo de defecto: que defecto se separa peor.
    _defmap = dict(zip(labels.filename, labels.defect))
    _por = pd.DataFrame({"score": list(oof_scores),
        "defect": [_defmap[_f] for _f in fnames],
        "label": ["bad" if _v else "good" for _v in y]})
    mo.vstack([
        mo.md("### Scores por defecto al corte FAR5% (`damaged_case` el mas dificil)"),
        (alt.Chart(_por).mark_tick(size=12).encode(
            x=alt.X("defect:N", title="defecto"),
            y=alt.Y("score:Q", title="P(bad) out-of-fold"),
            color=alt.Color("label:N", title="clase"),
        ).properties(width=450)
        + alt.Chart(pd.DataFrame({"cut": [cut_far5]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="cut:Q")),
    ])
    return


@app.cell
def tsne(X, alt, fnames, labels, mo, pd, y):
    from sklearn.manifold import TSNE

    # Vista 2D de los embeddings 768-d (solo exploratorio, no es el clasificador).
    _Z = TSNE(n_components=2, perplexity=30, random_state=42, init="random").fit_transform(X)
    _tdf = pd.DataFrame({"x": _Z[:, 0], "y": _Z[:, 1],
        "label": ["bad" if _v else "good" for _v in y],
        "defect": [dict(zip(labels.filename, labels.defect))[_f] for _f in fnames],
        "file": list(fnames)})
    mo.vstack([
        mo.md("### t-SNE de embeddings (solo exploratorio)"),
        alt.Chart(_tdf).mark_circle(size=40).encode(
            x=alt.X("x:Q", title="t-SNE 1"), y=alt.Y("y:Q", title="t-SNE 2"),
            color=alt.Color("defect:N", title="defecto"),
            shape=alt.Shape("label:N", title="clase"),
            tooltip=["file:N", "defect:N"],
        ).properties(width=450, height=350).interactive(),
    ])
    return


@app.cell
def optuna_plots(mo, study):
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    # Diagnostico del tuning: convergencia e importancia de C (con 1 parametro es trivial).
    mo.vstack([
        mo.md("### Optuna: historial e importancia"),
        plot_optimization_history(study),
        plot_param_importances(study),
    ])
    return


@app.cell
def ablacion_cls(mo, pd):
    from mlflow.tracking import MlflowClient as _MC2
    # Comparativa historica leida de MLflow (sin recomputar nada).
    # v2 y v3-Small quedaron fijos; el kernel vivo corre v3-Base.
    _q = _MC2()
    def _mm(_name):
        return _q.search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = '" + _name + "'")[0].data.metrics
    _a = _mm("faseA_dinov2_logreg")
    _b = _mm("faseA_dinov3_logreg")
    _c = _mm("faseA_dinov3b_logreg")
    _d = _mm("faseA_dinov3_cls_ablation")
    _cmp = pd.DataFrame([
        {"config": "v2 CLS-768", "recall_far5": round(_a["thr5_bad_recall"], 3), "auroc": round(_a["auroc_mean"], 3)},
        {"config": "v3s pooler-384", "recall_far5": round(_b["thr5_bad_recall"], 3), "auroc": round(_b["auroc_mean"], 3)},
        {"config": "v3s CLS-384", "recall_far5": round(_d["recall_at_far5_mean"], 3), "auroc": round(_d["auroc_mean"], 3)},
        {"config": "v3b pooler-768", "recall_far5": round(_c["thr5_bad_recall"], 3), "auroc": round(_c["auroc_mean"], 3)},
    ])
    mo.vstack([mo.md("### Historico v2 vs v3 (desde MLflow, sin recomputo)"), _cmp])
    return


@app.cell
def md_cierre(mo):
    mo.md("""
    ### Conclusion fase A (3 backbones, mismo pipeline por `btag`)
    - **v2-Base** (C=1.0): recall 0.95@FAR5% (FAR real 0.11, prec 0.57), AUROC 0.982.
    - **v3-Small**: 0.80 / 0.938. **v3-Base**: 0.90 / 0.970, Optuna 0.975.
    - Veredicto: empate tecnico (std ~0.1 con 8 bad/fold); C importa mas que backbone.
    - `damaged_case` el defecto mas dificil en los tres (recall 0.7-0.8).
    - Modelos en registry `transistor-logreg`; metricas y PNGs en MLflow `transistor-binary`.
    - Siguiente: fase B (PatchCore no supervisado, solo GOOD).
    """)
    return


@app.cell
def _(np):
    def _d2run():
        import torch
        import numpy as _np
        import pandas as _pd
        from pathlib import Path as _P
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.model_selection import StratifiedKFold as _SKF, cross_val_predict as _cvp
        _labels = _pd.read_csv(_P("transistor_binary") / "labels.csv")
        _y = (_labels["label"] == "bad").to_numpy(int)
        _saved = torch.load(_P("fase_a_emb_dinov2.pt"), weights_only=False)
        _s = _cvp(_LR(class_weight="balanced", C=1.0, max_iter=2000), _saved["X"], _y, cv=_SKF(n_splits=5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        _np.savez_compressed("fase_a_oof_dinov2.npz", scores=np.asarray(_s), fnames=np.asarray(_saved["fnames"]), y=_y)
        from mlflow.tracking import MlflowClient as _MCd
        import mlflow as _mlf
        _mlf.set_tracking_uri("sqlite:///mlflow.db")
        _h = _MCd().search_runs(experiment_ids=["1"], filter_string="tags.mlflow.runName = " + chr(34) + "faseA_dinov2_logreg" + chr(34))
        _rid = sorted(_h, key=lambda _r: _r.info.start_time)[-1].info.run_id
        with _mlf.start_run(run_id=_rid):
            _mlf.log_artifact("fase_a_oof_dinov2.npz")
        print(f"d2 oof ok -> {_rid[:8]}", flush=True)
        return _s, _saved["fnames"], _y
    D2OOF, D2FN, D2Y = _d2run()
    return


if __name__ == "__main__":
    app.run()
