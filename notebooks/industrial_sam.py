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
    # Template industrial con SAM (cambia carpetas y corre)
    Entrena DINO-masked + EffAD-original + fusión, exporta ONNX y deja ejemplo de inferencia.
    Todo deriva de la config de §0: para otra pieza cambia GOOD_DIR/BAD_DIR y corre de arriba a abajo.
    """)
    return


@app.cell
def _():
    # §0 Proposito: TODA la configuracion en un solo lugar + logs tic/toc. Lo unico que se edita por pieza.
    from pathlib import Path

    PROJECT_NAME = "transistor_binary"  # cambia por pieza, p.ej. "tuerca_m10"
    DATA_ROOT = Path("data/processed/transistor_binary")
    GOOD_DIR = DATA_ROOT / "good"  # solo cambiar estas dos carpetas por pieza
    BAD_DIR = DATA_ROOT / "bad"
    BACKBONE_ID = "facebook/dinov2-base"
    SAM_CHECKPOINT = "facebook/sam3"
    EFFAD_SIZE = "medium"
    STEPS = 5000
    DINO_VARIANT = "masked"  # ADOPT Fase I: dino va enmascarado
    EFFAD_VARIANT = "none"  # EffAD se queda en original (masked lo rompe: 0.625)
    TEST_SIZE = 0.2
    SEED = 42
    SAM_PADDING = 0.08
    MIN_MASK_AREA_RATIO = 0.03
    MAX_MASK_AREA_RATIO = 0.85
    N_BOOT = 200
    MLFLOW_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT = PROJECT_NAME
    MODELS_DIR = Path("models") / PROJECT_NAME / "prod"
    FIG_DIR = Path("reports/figures") / PROJECT_NAME

    # Log de corridas: toda celda pesada llama tic/toc + prints internos por fold/foto.
    import time as _tlog
    TIC = {}

    def tic(name):
        TIC[name] = _tlog.time()
        print("[%s] inicio %s" % (_tlog.strftime("%H:%M:%S"), name), flush=True)

    def toc(name):
        _dt = _tlog.time() - TIC.pop(name, _tlog.time())
        print("[%s] fin %s (%.1fs)" % (_tlog.strftime("%H:%M:%S"), name, _dt), flush=True)
    import numpy as np
    import pandas as pd
    import hashlib
    import matplotlib.pyplot as plt

    return (
        BACKBONE_ID,
        BAD_DIR,
        DINO_VARIANT,
        EFFAD_SIZE,
        EFFAD_VARIANT,
        FIG_DIR,
        GOOD_DIR,
        MAX_MASK_AREA_RATIO,
        MIN_MASK_AREA_RATIO,
        MLFLOW_EXPERIMENT,
        MLFLOW_URI,
        MODELS_DIR,
        PROJECT_NAME,
        SAM_CHECKPOINT,
        SAM_PADDING,
        SEED,
        STEPS,
        TEST_SIZE,
        hashlib,
        np,
        pd,
        plt,
        tic,
        toc,
    )


@app.cell
def _(mo):
    mo.md("""
    ### §0 Datos: escanea carpetas, arma labels y fingerprint (aborta si falta algo)
    """)
    return


@app.cell
def _(BAD_DIR, GOOD_DIR, MODELS_DIR, SEED, TEST_SIZE, hashlib, pd):
    # §0 Proposito: inventario del dataset (filename, label, path) + split train/test + fingerprint.

    from sklearn.model_selection import StratifiedShuffleSplit

    _recs = [{"filename": p.name, "label": "good", "path": str(p)} for p in sorted(GOOD_DIR.glob("*")) if p.is_file()]
    _recs += [{"filename": p.name, "label": "bad", "path": str(p)} for p in sorted(BAD_DIR.glob("*")) if p.is_file()]
    assert _recs, f"carpetas vacias: {GOOD_DIR} {BAD_DIR}"
    df_files = pd.DataFrame(_recs)
    assert not df_files.filename.duplicated().any(), "filenames duplicados entre good/bad"
    print(f"dataset: {(df_files.label == 'good').sum()} GOOD + {(df_files.label == 'bad').sum()} BAD")
    fingerprint = hashlib.sha1("|".join(sorted(f"{r.filename}:{(GOOD_DIR / r.filename if r.label == 'good' else BAD_DIR / r.filename).stat().st_size}" for r in df_files.itertuples())).encode()).hexdigest()[:16]
    print("fingerprint:", fingerprint)
    _y0 = (df_files.label == "bad").to_numpy(int)
    _tr, _te = next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(df_files, _y0))
    train_files = df_files.iloc[_tr].reset_index(drop=True)
    test_files = df_files.iloc[_te].reset_index(drop=True)
    print(f"split: {len(train_files)} train ({(train_files.label == 'bad').sum()} BAD) + {len(test_files)} test ({(test_files.label == 'bad').sum()} BAD)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return df_files, fingerprint, test_files, train_files


@app.cell
def _(mo):
    mo.md("""
    ### §1 SAM: segmenta una vez con logs (reusa cache si el fingerprint coincide)
    """)
    return


@app.cell
def _(SAM_CHECKPOINT, np, tic, toc):
    # §1 Proposito: carga SAM3 una vez + funciones mascara/bbox/crop. Solo localiza, nunca clasifica.
    import time

    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor, AutoImageProcessor, AutoModel

    tic("sam_load")
    _sam_proc = Sam3Processor.from_pretrained(SAM_CHECKPOINT)
    _sam_model = Sam3Model.from_pretrained(SAM_CHECKPOINT).eval()
    for _sp in _sam_model.parameters():
        _sp.requires_grad_(False)
    _sam_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sam_model.to(_sam_dev)
    print("sam device:", _sam_dev)


    def segment_part(image, prompt="transistor"):
        _t0 = time.perf_counter()
        _inp = _sam_proc(images=image, text=prompt, return_tensors="pt")
        _inp = {k: (v.to(_sam_dev) if torch.is_tensor(v) else v) for k, v in _inp.items()}
        with torch.inference_mode():
            _out = _sam_model(**_inp)
        _qi = int(_out.pred_logits[0].argmax())
        _score = float(_out.pred_logits[0][_qi])
        _m = torch.sigmoid(_out.pred_masks[0][_qi]).cpu().numpy() > 0.5
        _mask = np.array(Image.fromarray((_m * 255).astype("uint8")).resize(image.size, Image.NEAREST)) > 0
        return {"mask": _mask, "score": _score, "latency": time.perf_counter() - _t0}


    def mask_to_bbox(mask):
        _ys, _xs = np.where(np.asarray(mask))
        if len(_xs) == 0:
            return None
        return (int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max()))


    def expand_bbox(bbox, image_shape, padding_ratio=0.08):
        _h, _w = image_shape[:2]
        _x0, _y0, _x1, _y1 = bbox
        _bw, _bh = _x1 - _x0, _y1 - _y0
        _px, _py = int(_bw * padding_ratio), int(_bh * padding_ratio)
        return (max(0, _x0 - _px), max(0, _y0 - _py), min(_w, _x1 + _px), min(_h, _y1 + _py))

    toc("sam_load")
    return (
        AutoImageProcessor,
        AutoModel,
        Image,
        expand_bbox,
        mask_to_bbox,
        segment_part,
        torch,
    )


@app.cell
def _(
    Image,
    MODELS_DIR,
    df_files,
    fingerprint,
    mask_to_bbox,
    np,
    pd,
    segment_part,
    tic,
    toc,
    torch,
):
    # §1 Proposito: mascaras de TODO el dataset con log cada 10 (o reusa cache por fingerprint).

    tic("sam_cache")
    _sam_dir = MODELS_DIR / "sam"
    _sam_dir.mkdir(parents=True, exist_ok=True)
    _csv, _pt = _sam_dir / "masks.csv", _sam_dir / "masks256.pt"
    _fp_file = _sam_dir / "fingerprint.txt"
    if _csv.exists() and _fp_file.exists() and _fp_file.read_text().strip() == fingerprint:
        sam_masks = pd.read_csv(_csv)
        _mp = torch.load(_pt, map_location="cpu", weights_only=False)
        sam_masks256, _mfn = _mp["masks"], _mp["fnames"]
        print(f"sam cache hit: {len(sam_masks)} filas")
    else:
        _rows, _ml, _mfn = [], [], []
        for _k, _r in enumerate(df_files.itertuples()):
            _im = Image.open(_r.path).convert("RGB")
            _s = segment_part(_im)
            _bb = mask_to_bbox(_s["mask"])
            _m = _s["mask"]
            _rows.append({"filename": _r.filename, "label": _r.label, "score": round(_s["score"], 3),
                           "area": round(float(_m.mean()), 4), "bbox": _bb is not None,
                           "edge": bool(_m[0, :].any() or _m[-1, :].any() or _m[:, 0].any() or _m[:, -1].any()),
                           "lat": round(_s["latency"], 2),
                           "x0": _bb[0] if _bb else -1, "y0": _bb[1] if _bb else -1,
                           "x1": _bb[2] if _bb else -1, "y1": _bb[3] if _bb else -1})
            _ml.append(torch.from_numpy(np.array(Image.fromarray(_m.astype("uint8") * 255).resize((256, 256))) > 0))
            _mfn.append(_r.filename)
            if (_k + 1) % 10 == 0:
                print(f"sam {_k + 1}/{len(df_files)}", flush=True)
        sam_masks = pd.DataFrame(_rows)
        sam_masks.to_csv(_csv, index=False)
        sam_masks256 = torch.stack(_ml)
        torch.save({"masks": sam_masks256, "fnames": _mfn}, _pt)
        _fp_file.write_text(fingerprint)
        print(f"saved {len(sam_masks)} mascaras")
    print(f"bbox_ok={sam_masks.bbox.mean():.3f} edge={sam_masks.edge.mean():.3f} lat_media={sam_masks.lat.mean():.2f}s total={sam_masks.lat.sum() / 60:.1f}min")
    toc("sam_cache")
    return sam_masks, sam_masks256


@app.cell
def _(mo):
    mo.md("""
    ### §2 Variantes none/crop/masked + media GOOD-train (train-only, sin fuga)
    """)
    return


@app.cell
def _(
    Image,
    MAX_MASK_AREA_RATIO,
    MIN_MASK_AREA_RATIO,
    MODELS_DIR,
    SAM_PADDING,
    expand_bbox,
    np,
    sam_masks,
    sam_masks256,
    tic,
    toc,
    train_files,
):
    # §2 Proposito: construye cada variante + fondo masked = media de GOOD-train (nunca test).
    tic("variants")
    _pos = dict(zip(sam_masks.filename, range(len(sam_masks))))
    _good_tr = train_files[train_files.label == "good"]
    _mean = np.zeros((1024, 1024, 3), np.float64)
    for _gf in _good_tr.itertuples():
        _mean += np.asarray(Image.open(_gf.path).convert("RGB"), np.float64)
    good_mean = (_mean / len(_good_tr)).astype("uint8")
    print(f"media GOOD-train de {len(_good_tr)} fotos")
    _qc = {r.filename: (r.area, r.edge, (r.x0, r.y0, r.x1, r.y1) if r.bbox else None) for r in sam_masks.itertuples()}
    _n_fb = []

    def variant_image(filename, path, variant):
        _im = Image.open(path).convert("RGB")
        if variant == "none":
            return _im
        _area, _edge, _bb = _qc[filename]
        if _bb is None or _bb[0] < 0 or _area < MIN_MASK_AREA_RATIO or _area > MAX_MASK_AREA_RATIO or _edge:
            _n_fb.append(filename)
            return _im  # fallback QC: original
        if variant == "crop":
            from PIL import Image as _I  # noqa

            return _im.crop(expand_bbox(_bb, (_im.size[1], _im.size[0]), SAM_PADDING))
        _mk = np.array(Image.fromarray((sam_masks256[_pos[filename]].numpy() * 255).astype("uint8")).resize((1024, 1024))) > 0
        return Image.fromarray(np.where(_mk[:, :, None], np.asarray(_im.resize((1024, 1024))), good_mean).astype("uint8"))

    print("fallback QC activable en:", sorted(set(_n_fb)) if False else "(se cuenta al embeddar)")
    np.save(MODELS_DIR / "good_mean.npy", good_mean)
    print("saved good_mean.npy", good_mean.shape)
    toc("variants")
    return (variant_image,)


@app.cell(hide_code=True)
def viz_crops(
    FIG_DIR,
    Image,
    SAM_PADDING,
    expand_bbox,
    mask_to_bbox,
    mo,
    np,
    plt,
    segment_part,
    tic,
    toc,
    train_files,
    variant_image,
):
    # Proposito: 2 mosaicos (10 GOOD y 10 BAD): original | overlay+bbox | crop p08 | masked. Solo muestra.
    tic('viz_crops')
    def _mosaico(_rows, _titulo, _fname):
        _f, _axs = plt.subplots(len(_rows), 6, figsize=(12, 1.6 * len(_rows)))
        _f.patch.set_facecolor("white")
        for _k, _r in enumerate(_rows):
            _im = Image.open(_r.path).convert("RGB")
            _s = segment_part(_im)
            _bb = mask_to_bbox(_s["mask"])
            _row = _axs[_k] if len(_rows) > 1 else _axs
            _row[0].imshow(_im)
            _row[0].set_title(_r.filename, fontsize=8, color="black")
            _row[1].imshow(_im)
            _row[1].imshow(np.ma.masked_where(~_s["mask"][::4, ::4], np.ones((256, 256))), cmap="jet", alpha=0.4, extent=(0, _im.size[0], _im.size[1], 0))
            if _bb:
                _x0, _y0, _x1, _y1 = _bb
                _row[1].plot([_x0, _x1, _x1, _x0, _x0], [_y0, _y0, _y1, _y1, _y0], "r-")
            _row[1].set_title("overlay", fontsize=8, color="black")
            _row[2].imshow(variant_image(_r.filename, _r.path, "crop"))
            _row[2].set_title("crop", fontsize=8, color="black")
            _row[3].imshow(variant_image(_r.filename, _r.path, "masked"))
            _row[3].set_title("masked", fontsize=8, color="black")
            _full = np.array(Image.fromarray((_s["mask"] * 255).astype("uint8")).resize((1024, 1024))) > 0
            _row[4].imshow(np.where(_full[:, :, None], np.asarray(_im.resize((1024, 1024))), np.zeros((1024, 1024, 3), np.uint8)))
            _row[4].set_title("fondo negro", fontsize=8, color="black")
            _cropmask = Image.fromarray(np.where(_full[:, :, None], np.asarray(_im.resize((1024, 1024))), np.zeros((1024, 1024, 3), np.uint8))).crop(expand_bbox(_bb, (1024, 1024), SAM_PADDING)) if _bb else _im
            _row[5].imshow(_cropmask)
            _row[5].set_title("crop+mask", fontsize=8, color="black")
            _row[3].set_title("masked", fontsize=8, color="black")
            [a.axis("off") for a in _row]
        _f.suptitle(_titulo, fontsize=12, color="black")
        _f.subplots_adjust(wspace=0.05, hspace=0.3, top=0.95)
        _f.savefig(FIG_DIR / _fname, dpi=90)
        print("saved", _fname)
        return _f
    _goods = list(train_files[train_files.label == "good"].itertuples())[:10]
    _bads = list(train_files[train_files.label == "bad"].itertuples())[:10]
    _fg = _mosaico(_goods, "10 GOOD: orig | overlay | crop | masked", "ejemplos_good.png")
    _fb = _mosaico(_bads, "10 BAD: orig | overlay | crop | masked", "ejemplos_bad.png")
    toc('viz_crops')
    mo.vstack([_fg, _fb])
    return


@app.cell
def _(mo):
    mo.md("""
    ### §3 DINO en la variante elegida + LogReg + export ONNX
    """)
    return


@app.cell
def _(
    AutoImageProcessor,
    AutoModel,
    BACKBONE_ID,
    DINO_VARIANT,
    MODELS_DIR,
    df_files,
    fingerprint,
    tic,
    toc,
    torch,
    variant_image,
):
    # §3 Proposito: embeddings CLS DINOv2 de la variante DINO_VARIANT (cache por fingerprint).

    tic("dino_embed")

    _ck = MODELS_DIR / f"dino_{DINO_VARIANT}.pt"
    if _ck.exists() and torch.load(_ck, map_location="cpu", weights_only=False).get("fingerprint") == fingerprint:
        _dp = torch.load(_ck, map_location="cpu", weights_only=False)
        dino_X, dino_fnames = _dp["X"], _dp["fnames"]
        print("dino cache hit:", dino_X.shape)
    else:
        _proc = AutoImageProcessor.from_pretrained(BACKBONE_ID)
        _model = AutoModel.from_pretrained(BACKBONE_ID).eval()
        for _pp in _model.parameters():
            _pp.requires_grad_(False)
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_dev)
        _xs, _fn = [], []
        for _k, _r in enumerate(df_files.itertuples()):
            _ci = variant_image(_r.filename, _r.path, DINO_VARIANT)
            _in = _proc(images=_ci, size={"height": 224, "width": 224}, return_tensors="pt")
            with torch.inference_mode():
                _xs.append(_model(**{k: v.to(_dev) for k, v in _in.items()}).last_hidden_state[:, 0, :].cpu())
            _fn.append(_r.filename)
            if (_k + 1) % 50 == 0:
                print(f"dino {_k + 1}/{len(df_files)}", flush=True)
        import numpy as _np

        dino_X, dino_fnames = _np.concatenate([x.numpy() for x in _xs]), _fn
        torch.save({"X": dino_X, "fnames": dino_fnames, "fingerprint": fingerprint}, _ck)
        print("saved", dino_X.shape)
    toc("dino_embed")
    try:
        _model.to("cpu")
        print("DINO descargado")
    except NameError:
        print("DINO ya estaba descargado (cache hit)")
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    print("DINO descargado, VRAM:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
    return dino_X, dino_fnames


@app.cell
def _(
    MODELS_DIR,
    df_files,
    dino_X,
    dino_fnames,
    np,
    test_files,
    tic,
    toc,
    train_files,
):
    # §3 Proposito: LogReg en train, metricas en test, scores y export a ONNX.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report as _crN, roc_auc_score as _aucN

    tic("dino_train")
    _y_all = (df_files.set_index("filename").loc[dino_fnames, "label"] == "bad").to_numpy(int)
    _pos = {f: i for i, f in enumerate(dino_fnames)}
    _tri = [_pos[f] for f in train_files.filename]
    _tei = [_pos[f] for f in test_files.filename]
    _clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(dino_X[_tri], _y_all[_tri])
    dino_scores = _clf.predict_proba(dino_X)[:, 1]
    _cut = float(np.quantile(dino_scores[_tri][_y_all[_tri] == 0], 0.95))
    _pr = (dino_scores[_tei] >= _cut).astype(int)
    _yte = _y_all[_tei]
    print(_crN(_yte, _pr, target_names=["good", "bad"], zero_division=0))
    print(f"dino-test: auroc={_aucN(_yte, dino_scores[_tei]):.4f} cut={_cut:.3f}")
    np.savez(MODELS_DIR / "dino.npz", fnames=np.array(dino_fnames), y=_y_all, scores=dino_scores,
             train_idx=np.array(_tri), test_idx=np.array(_tei), cut=_cut)
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    _onnx = to_onnx(_clf, initial_types=[("input", FloatTensorType([None, dino_X.shape[1]]))], options={"zipmap": False})
    (MODELS_DIR / "dino_logreg.onnx").write_bytes(_onnx.SerializeToString())
    print("saved dino.npz + dino_logreg.onnx")
    toc("dino_train")
    return (dino_scores,)


@app.cell(hide_code=True)
def tsne_dino(FIG_DIR, df_files, dino_X, dino_fnames, mo, plt, tic, toc):
    # Proposito: t-SNE de embeddings DINO-masked (separacion GOOD/BAD de un vistazo). Solo visualiza.
    tic('tsne')
    from sklearn.manifold import TSNE as _TSNE
    _yy = (df_files.set_index("filename").loc[dino_fnames, "label"] == "bad").to_numpy(int)
    _Z = _TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(dino_X)
    print("tsne listo:", _Z.shape, flush=True)
    import seaborn as _snsT
    _snsT.set_theme(style="whitegrid")
    _ft, _at = plt.subplots(figsize=(7, 5))
    _snsT.scatterplot(x=_Z[:, 0], y=_Z[:, 1], hue=["bad" if y else "good" for y in _yy], palette={"good": "#4C78A8", "bad": "#F58518"}, s=18, ax=_at)
    _at.set(title="t-SNE DINO-masked (313 fotos)", xlabel="t1", ylabel="t2")
    _ft.tight_layout()
    _ft.savefig(FIG_DIR / "tsne_dino.png", dpi=100)
    print("saved tsne_dino.png")
    toc('tsne')
    mo.vstack([_ft])
    return


@app.cell(hide_code=True)
def cm_dino(FIG_DIR, MODELS_DIR, mo, np, plt, tic, toc):
    # Proposito: matriz de confusion de DINO en test (EffAD/fusion llegan tras el entreno). Solo muestra.
    tic('cm_dino')
    from sklearn.metrics import confusion_matrix as _cmx9
    import seaborn as _sns9
    _sns9.set_theme(style="whitegrid")
    _dz = np.load(MODELS_DIR / "dino.npz")
    _yte9 = _dz["y"][_dz["test_idx"]].astype(int)
    _pr9 = (_dz["scores"][_dz["test_idx"]] >= float(_dz["cut"])).astype(int)
    _tn, _fp, _fn, _tp = _cmx9(_yte9, _pr9).ravel()
    print(f"dino-test: tn={_tn} fp={_fp} fn={_fn} tp={_tp}")
    _fc9, _ac9 = plt.subplots(figsize=(4.5, 4))
    _sns9.heatmap([[_tn, _fp], [_fn, _tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=_ac9)
    _ac9.set(title="DINO-masked test (63 fotos)", xlabel="pred", ylabel="real")
    _ac9.set_xticklabels(["good", "bad"])
    _ac9.set_yticklabels(["good", "bad"], rotation=0)
    _fc9.tight_layout()
    _fc9.savefig(FIG_DIR / "cm_dino.png", dpi=100)
    print("saved cm_dino.png")
    toc('cm_dino')
    mo.vstack([_fc9])
    return


@app.cell
def _(mo):
    mo.md("""
    ### §4 EffAD en original, 5000 steps con logs + export ONNX
    """)
    return


@app.cell
def _(
    EFFAD_SIZE,
    EFFAD_VARIANT,
    MODELS_DIR,
    SEED,
    STEPS,
    np,
    tic,
    toc,
    torch,
    train_files,
    variant_image,
):
    # §4 Proposito: entrena EfficientAD solo en GOOD-train (variante EFFAD_VARIANT) y exporta a ONNX.
    import glob

    tic("effad_train")
    from lightning.pytorch import seed_everything
    from torch.utils.data import DataLoader
    from lightning.pytorch import LightningDataModule
    from anomalib.data.dataclasses.torch import ImageBatch
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine

    seed_everything(SEED)
    _good_tr = train_files[train_files.label == "good"]

    def _tensor(_path, _fn):
        _im = variant_image(_fn, _path, EFFAD_VARIANT).resize((256, 256))
        return torch.from_numpy(np.asarray(_im, dtype="float32") / 255.0).permute(2, 0, 1)

    class _DM(LightningDataModule):
        def __init__(self, _t):
            super().__init__()
            self.train_batch_size = 1
            self.name = EFFAD_VARIANT
            self.category = EFFAD_VARIANT
            self._t = _t

        def train_dataloader(self):
            return DataLoader(self._t, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))

    from lightning.pytorch.callbacks import Callback as _CB
    class _LogSteps(_CB):
        def on_train_batch_end(self, trainer, *_a, **_k):
            _gs = int(trainer.global_step)
            if _gs % 250 == 0 and _gs > 0:
                _mm = {k: float(v) for k, v in trainer.callback_metrics.items()}
                _msg = " ".join(k + "=" + format(v, ".4f") for k, v in _mm.items()) or "(sin metricas)"
                print("effad step %d/%d %s" % (_gs, STEPS, _msg), flush=True)
    _model = EfficientAd(model_size=EFFAD_SIZE, visualizer=False, evaluator=False)
    _eng = Engine(logger=True, default_root_dir=str(MODELS_DIR / "effad"), max_steps=STEPS, callbacks=[_LogSteps()])
    print(f"entrenando EffAD-{EFFAD_SIZE} en {len(_good_tr)} GOOD x {STEPS} steps (log cada N steps en light_logs/)", flush=True)

    from lightning.pytorch.callbacks import Callback as _CB
    class _LogSteps(_CB):
        def on_train_batch_end(self, trainer, *_a, **_k):
            _gs = int(trainer.global_step)
            if _gs % 250 == 0 and _gs > 0:
                _mm = {k: float(v) for k, v in trainer.callback_metrics.items()}
                _msg = " ".join(k + "=" + format(v, ".4f") for k, v in _mm.items()) or "(sin metricas)"
                print("effad step %d/%d %s" % (_gs, STEPS, _msg), flush=True)
    print("callback cada 250 steps enganchado")
    import glob as _gb2
    _ckpts = sorted(_gb2.glob(str(MODELS_DIR / "effad/**/*.ckpt"), recursive=True))
    _resume = _ckpts[-1] if _ckpts else None
    print("resume desde:", _resume, flush=True)
    _eng.fit(model=_model, datamodule=_DM([_tensor(r.path, r.filename) for r in _good_tr.itertuples()]), ckpt_path=_resume)
    _ckpt = sorted(glob.glob(str(MODELS_DIR / "effad/**/*.ckpt"), recursive=True))[-1]
    print("ckpt:", _ckpt)

    # Export ONNX: wrapper que devuelve pred_score como tensor.
    import torch.nn as _nn

    class _EffWrap(_nn.Module):
        def __init__(self, _m):
            super().__init__()
            self._m = _m

        def forward(self, _x):
            return self._m(_x).pred_score

    _model.eval()
    _dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(_EffWrap(_model).eval(), _dummy, str(MODELS_DIR / "effad.onnx"),
                      input_names=["input"], output_names=["score"], opset_version=17, dynamo=False)
    print("saved effad.onnx")
    toc("effad_train")
    return EfficientAd, glob


@app.cell
def _(mo):
    mo.md("""
    ### §5 Fusión dino + effad (alpha en train) y scores test
    """)
    return


@app.cell
def _(
    EfficientAd,
    MODELS_DIR,
    df_files,
    dino_fnames,
    dino_scores,
    glob,
    np,
    test_files,
    tic,
    toc,
    torch,
    train_files,
    variant_image,
):
    # §5 Proposito: normaliza con train-GOOD, elige alpha en train, corta q95 y scorea test (dino+effad+fusion).

    tic("fusion")

    _y = (df_files.set_index("filename").loc[dino_fnames, "label"] == "bad").to_numpy(int)
    _pos = {f: i for i, f in enumerate(dino_fnames)}
    _tri = [_pos[f] for f in train_files.filename]
    _tei = [_pos[f] for f in test_files.filename]
    _ckpt = sorted(glob.glob(str(MODELS_DIR / "effad/**/*.ckpt"), recursive=True))[-1]
    _em = EfficientAd.load_from_checkpoint(_ckpt, map_location="cpu").eval()
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _em.to(_dev)
    _es = []
    with torch.inference_mode():
        for _s in range(0, len(dino_fnames), 16):
            _xb = torch.stack([torch.from_numpy(np.asarray(variant_image(f, df_files.set_index("filename").loc[f, "path"], "none").resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1) for f in dino_fnames[_s:_s + 16]]).to(_dev)
            _es.extend(_em(_xb).pred_score.flatten().cpu().tolist())
            if (_s // 16 + 1) % 5 == 0:
                print(f"effad score {_s + 16}/{len(dino_fnames)}", flush=True)
    effad_scores = np.array(_es)
    _gd = [i for i in _tri if _y[i] == 0]
    _md = float(np.median(dino_scores[_gd]))
    _id = float(max(np.subtract(*np.percentile(dino_scores[_gd], [75, 25])), 1e-9))
    _me = float(np.median(effad_scores[_gd]))
    _ie = float(max(np.subtract(*np.percentile(effad_scores[_gd], [75, 25])), 1e-9))
    _dtr, _etr = (dino_scores[_tri] - _md) / _id, (effad_scores[_tri] - _me) / _ie
    _best = None
    for _a in np.arange(0, 1.05, 0.05).round(2):
        _ftr = _a * _dtr + (1 - _a) * _etr
        _thr = float(np.quantile(_ftr[_y[_tri] == 0], 0.95))
        _pr = (_ftr >= _thr).astype(int)
        _rec = float((_pr[_y[_tri] == 1] == 1).mean())
        _far = float((_pr[_y[_tri] == 0] == 1).mean())
        if _best is None or (_rec, -_far) > (_best[1], -_best[2]):
            _best = (float(_a), _rec, _far, _thr)
    _alpha, _, _, _cut = _best
    fused_scores = _alpha * (dino_scores - _md) / _id + (1 - _alpha) * (effad_scores - _me) / _ie
    np.savez(MODELS_DIR / "fusion.npz", alpha=_alpha, cut=_cut, md=_md, id=_id, me=_me, ie=_ie,
             fnames=np.array(dino_fnames), y=_y, dino=dino_scores, effad=effad_scores, fused=fused_scores,
             train_idx=np.array(_tri), test_idx=np.array(_tei))
    print(f"alpha={_alpha} cut={_cut:.3f} saved fusion.npz")
    _em.to("cpu"); torch.cuda.empty_cache()
    print("EffAD descargado")
    toc("fusion")
    return


@app.cell
def _(mo):
    mo.md("""
    ### §6 Evaluación: reportes, matrices, McNemar, bootstrap + figuras mostradas
    """)
    return


@app.cell
def _(FIG_DIR, MODELS_DIR, mo, np, pd, plt, tic, toc):
    # §6 Proposito: metricas test por brazo + figuras seaborn (barras + matrices + McNemar). Sin hardcodes.
    tic("eval")
    import seaborn as sns
    from sklearn.metrics import classification_report as _cr6, confusion_matrix as _cmx6, roc_auc_score as _auc6
    sns.set_theme(style="whitegrid")
    _fz = np.load(MODELS_DIR / "fusion.npz")
    _y = _fz["y"].astype(int)
    _te = _fz["test_idx"]
    _yte = _y[_te]
    _pal = {"dino": "#4C78A8", "effad": "#F58518", "fusion": "#54A24B"}
    _rows, _preds = [], {}
    for _m in ("dino", "effad", "fused"):
        _sc = _fz[{"dino": "dino", "effad": "effad", "fused": "fused"}[_m]]
        _cut = float(np.quantile(_sc[_fz["train_idx"]][_y[_fz["train_idx"]] == 0], 0.95))
        _pr = (_sc[_te] >= _cut).astype(int)
        _preds[_m] = _pr
        _rep = _cr6(_yte, _pr, target_names=["good", "bad"], output_dict=True, zero_division=0)
        _rows.append({"method": _m, "auroc": round(float(_auc6(_yte, _sc[_te])), 4), "recall": round(float(_rep["bad"]["recall"]), 3), "far": round(float((_pr[_yte == 0] == 1).mean()), 3), "prec_bad": round(float(_rep["bad"]["precision"]), 3), "f1_bad": round(float(_rep["bad"]["f1-score"]), 3)})
        print(_m, _rows[-1], flush=True)
    eval_df = pd.DataFrame(_rows)
    eval_df.to_csv(MODELS_DIR / "eval.csv", index=False)
    _mlab = ["dino", "effad", "fusion"]
    import math as _m6
    def _mcn6(_a, _b):
        _n01 = int(((_a == 0) & (_b == 1)).sum())
        _n10 = int(((_a == 1) & (_b == 0)).sum())
        _n = _n01 + _n10
        return 1.0 if _n == 0 else round(min(1.0, 2.0 * sum(_m6.comb(_n, _k) for _k in range(0, min(_n01, _n10) + 1)) / 2.0 ** _n), 4)
    print("McNemar:", {p: _mcn6(_preds[p[0]], _preds[p[1]]) for p in [("dino", "effad"), ("dino", "fused"), ("effad", "fused")]}, flush=True)
    _fe, _ae = plt.subplots(figsize=(8, 4))
    sns.barplot(data=pd.DataFrame({"method": _mlab, "recall": [r["recall"] for r in _rows]}), x="method", y="recall", hue="method", legend=False, palette=_pal, ax=_ae, order=_mlab)
    _ae.set(title="BAD recall test por brazo", ylim=(0, 1.05))
    _fm, _am = plt.subplots(1, 3, figsize=(12, 3.5))
    for _ax, _m in zip(_am, _mlab):
        _tn, _fp, _fn, _tp = _cmx6(_yte, _preds[{"dino": "dino", "effad": "effad", "fusion": "fused"}[_m]]).ravel()
        sns.heatmap([[_tn, _fp], [_fn, _tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=_ax)
        _ax.set(title=_m, xlabel="pred", ylabel="real")
        _ax.set_xticklabels(["good", "bad"])
        _ax.set_yticklabels(["good", "bad"], rotation=0)
    _fm.suptitle("Matrices test por brazo")
    _fm.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _fe.savefig(FIG_DIR / "recall.png", dpi=100)
    _fm.savefig(FIG_DIR / "cm.png", dpi=100)
    print("saved eval.csv, recall.png, cm.png")
    toc("eval")
    mo.vstack([_fe, _fm])
    return (eval_df,)


@app.cell(hide_code=True)
def figs_dist(FIG_DIR, MODELS_DIR, mo, np, plt, tic, toc):
    # Proposito: 4 figuras diagnosticas test (distribuciones, ROC, strip por clase, scatter dino-vs-effad). Solo muestra.
    tic("figs_dist")
    import seaborn as _snsD
    from sklearn.metrics import roc_curve as _rcc, roc_auc_score as _au9
    _snsD.set_theme(style="whitegrid")
    _fz9 = np.load(MODELS_DIR / "fusion.npz")
    _y9 = _fz9["y"].astype(int)
    _tr9, _te9 = _fz9["train_idx"], _fz9["test_idx"]
    _S9 = {"dino": _fz9["dino"], "effad": _fz9["effad"], "fusion": _fz9["fused"]}
    _cuts9 = {m: float(np.quantile(_S9[m][_tr9][_y9[_tr9] == 0], 0.95)) for m in _S9}
    print("cortes:", {m: round(c, 3) for m, c in _cuts9.items()}, flush=True)
    _lab9 = np.array(["bad" if y else "good" for y in _y9[_te9]])
    _f1, _a1 = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _m in zip(_a1, ["dino", "effad", "fusion"]):
        _snsD.histplot(x=_S9[_m][_te9], hue=_lab9, palette={"good": "#4C78A8", "bad": "#F58518"}, bins=30, ax=_ax)
        _ax.axvline(_cuts9[_m], color="k", ls="--")
        _ax.set(title=_m + " corte=" + format(_cuts9[_m], ".2f"), xlabel="score", ylabel="n")
    _f1.suptitle("Score test por brazo (GOOD/BAD + corte q95-train)")
    _f1.tight_layout()
    _f1.savefig(FIG_DIR / "dist_scores.png", dpi=100)
    print("saved dist_scores.png", flush=True)
    _f2, _a2 = plt.subplots(figsize=(7, 5))
    for _m, _c in [("dino", "#4C78A8"), ("effad", "#F58518"), ("fusion", "#54A24B")]:
        _fpr, _tpr, _ = _rcc(_y9[_te9], _S9[_m][_te9])
        _a2.plot(_fpr, _tpr, color=_c, label=_m + " AUROC=" + format(_au9(_y9[_te9], _S9[_m][_te9]), ".3f"))
    _a2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    _a2.set(title="ROC test por brazo", xlabel="FPR", ylabel="TPR (recall)")
    _a2.legend(fontsize=9)
    _f2.tight_layout()
    _f2.savefig(FIG_DIR / "roc.png", dpi=100)
    print("saved roc.png", flush=True)
    _f3, _a3 = plt.subplots(figsize=(9, 4))
    _snsD.stripplot(x=_lab9, y=_S9["dino"][_te9], hue=_lab9, palette={"good": "#4C78A8", "bad": "#F58518"}, legend=False, s=6, jitter=0.25, ax=_a3)
    _a3.axhline(_cuts9["dino"], color="r")
    _a3.set(title="DINO test por clase (linea roja = corte FAR5%)", ylabel="score")
    _f3.tight_layout()
    _f3.savefig(FIG_DIR / "strip_dino.png", dpi=100)
    print("saved strip_dino.png", flush=True)
    _f4, _a4 = plt.subplots(figsize=(7, 6))
    _snsD.scatterplot(x=_S9["dino"][_te9], y=_S9["effad"][_te9], hue=_lab9, palette={"good": "#4C78A8", "bad": "#F58518"}, s=20, ax=_a4)
    _a4.axvline(_cuts9["dino"], color="r", label="corte DINO")
    _a4.axhline(_cuts9["effad"], color="r", ls="--", label="corte EffAD")
    _a4.set(title="DINO vs EffAD test", xlabel="dino", ylabel="effad")
    _a4.legend(fontsize=8)
    _f4.tight_layout()
    _f4.savefig(FIG_DIR / "scatter_arms.png", dpi=100)
    print("saved scatter_arms.png")
    toc("figs_dist")
    mo.vstack([_f1, _f2, _f3, _f4])
    return


@app.cell
def _(mo):
    mo.md("""
    ### §7 MLflow: params, métricas, ONNXs, figuras y fingerprint en un run
    """)
    return


@app.cell
def _(
    BACKBONE_ID,
    DINO_VARIANT,
    EFFAD_SIZE,
    EFFAD_VARIANT,
    FIG_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    PROJECT_NAME,
    SEED,
    STEPS,
    eval_df,
    fingerprint,
    tic,
    toc,
):
    # §7 Proposito: deja todo trazado en MLflow (reproducible por fingerprint).
    tic("mlflow_final")

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{PROJECT_NAME}_prod"):
        mlflow.log_params({"backbone": BACKBONE_ID, "dino_variant": DINO_VARIANT, "effad_variant": EFFAD_VARIANT,
                            "effad_size": EFFAD_SIZE, "steps": STEPS, "seed": SEED, "fingerprint": fingerprint})
        for _, _r in eval_df.iterrows():
            mlflow.log_metrics({f"{_r.method}_recall": _r.recall, f"{_r.method}_far": _r.far, f"{_r.method}_auroc": _r.auroc})
        for _f in ["dino.npz", "fusion.npz", "eval.csv", "dino_logreg.onnx", "effad.onnx", "good_mean.npy"]:
            mlflow.log_artifact(str(MODELS_DIR / _f))
        for _g in FIG_DIR.glob("*.png"):
            mlflow.log_artifact(str(_g))
        mlflow.set_tag("fingerprint", fingerprint)
    print("mlflow listo")
    toc("mlflow_final")
    return


@app.cell
def _(mo):
    mo.md("""
    ### §8 Ejemplo de inferencia end-to-end (1 GOOD + 1 BAD con SAM en vivo)
    """)
    return


@app.cell
def _(
    AutoImageProcessor,
    AutoModel,
    BACKBONE_ID,
    Image,
    MODELS_DIR,
    mask_to_bbox,
    np,
    segment_part,
    test_files,
    tic,
    toc,
    torch,
):
    # §8 Proposito: prueba que el template sirve: carga ONNXs + SAM en vivo y decide 2 fotos nuevas.
    tic("infer_ejemplo")

    import onnxruntime as _ort

    _fz = np.load(MODELS_DIR / "fusion.npz")
    _sess_dino = _ort.InferenceSession(str(MODELS_DIR / "dino_logreg.onnx"))
    _sess_effad = _ort.InferenceSession(str(MODELS_DIR / "effad.onnx"))

    _proc = AutoImageProcessor.from_pretrained(BACKBONE_ID)
    _bm = AutoModel.from_pretrained(BACKBONE_ID).eval()
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bm.to(_dev)
    _pos = dict(zip(_fz["fnames"], range(len(_fz["fnames"]))))
    for _lb in ("good", "bad"):
        _row = test_files[test_files.label == _lb].iloc[0]
        _im0 = Image.open(_row.path).convert("RGB")
        # DINO va enmascarado: SAM en vivo (igual que en produccion real).
        _r = segment_part(_im0)
        _bb = mask_to_bbox(_r["mask"])
        from PIL import Image as _I

        _m = np.array(_I.fromarray((_r["mask"] * 255).astype("uint8")).resize((1024, 1024))) > 0
        _mean = np.load(MODELS_DIR / "good_mean.npy")  # media GOOD-train guardada (igual que en batch)
        _masked = _I.fromarray(np.where(_m[:, :, None], np.asarray(_im0.resize((1024, 1024))), _mean).astype("uint8"))
        _in = _proc(images=_masked, size={"height": 224, "width": 224}, return_tensors="pt")
        with torch.inference_mode():
            _cls = _bm(**{k: v.to(_dev) for k, v in _in.items()}).last_hidden_state[:, 0, :].cpu().numpy()
        _pd = float(_sess_dino.run(None, {"input": _cls.astype("float32")})[1][0][1])
        _xb = torch.from_numpy(np.asarray(_im0.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(_dev)
        with torch.inference_mode():
            _pe = float(torch.from_numpy(_sess_effad.run(None, {"input": _xb.cpu().numpy()})[0]).flatten()[0])
        _f = _fz["alpha"] * (_pd - _fz["md"]) / _fz["id"] + (1 - _fz["alpha"]) * (_pe - _fz["me"]) / _fz["ie"]
        print(f"{_row.filename} real={_lb} dino={_pd:.3f} effad={_pe:.4f} fusion={_f:.3f} -> {'BAD' if _f >= _fz['cut'] else 'GOOD'} (corte {_fz['cut']:.3f})")
    toc("infer_ejemplo")
    return


if __name__ == "__main__":
    app.run()
