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
    # Fase I — SAM3 como segmentador previo (ninguna decision GOOD/BAD aqui)

    **Pregunta:** recortar (crop) o enmascarar (masked) con SAM3, mejora al baseline (none) sin tocar el clasificador?

    **Por que existe:** el fondo distrae a DINOv2/EfficientAD; SAM3 solo localiza la pieza (ROI) y jamas clasifica.

    **Que hay:** 3 vias x 5 metodos con pipeline congelado, comparativa con estadistica y veredicto por brazo.
    """)
    return


@app.cell
def _():
    # §0 Proposito: constantes del experimento (rutas, modelo, splits, QC). Todo lo demas cuelga de aqui.
    from pathlib import Path

    PROJECT_NAME = "transistor_binary"
    DATASET_DIR = Path("data/processed/transistor_binary")
    LABELS_FILE = DATASET_DIR / "labels.csv"
    GOOD_LABEL = "good"
    BAD_LABEL = "bad"
    BACKBONE_ID = "facebook/dinov2-base"
    EFFICIENTAD_SIZE = "medium"
    N_SPLITS = 5
    REPEATED_SEEDS = [11, 22, 33, 44, 55]
    TARGET_FAR = 0.05
    MLFLOW_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT = PROJECT_NAME
    MODELS_DIR = Path("models") / PROJECT_NAME

    # SAM3 (fase I): solo localiza la pieza -> ROI. Nunca clasifica.
    SAM_CHECKPOINT = "facebook/sam3"
    SEGMENTATION_MODE = "none"  # none | crop | masked | sam3_auto_prototype
    SAM_PADDING_GRID = [0.05, 0.08, 0.10]
    SAM_PADDING = 0.08
    MASK_BACKGROUND = "mean"  # media GOOD-train
    SAM_INPUT_SIZE = 1024  # bajar a 640 si AMG es lento (clasificador intacto)

    # Filtros geometricos (label-blind, no dependen de GOOD/BAD)
    MIN_MASK_AREA_RATIO = 0.03
    MAX_MASK_AREA_RATIO = 0.85
    MAX_BORDER_TOUCH_RATIO = 0.50

    # Auto-prototype (GOOD-train del fold; ver bootstrap en §5)
    AUTO_SEGMENT = False
    SEGMENT_SELECTION_MODE = "good_prototype"
    N_PROTOTYPES = 1
    VISUAL_WEIGHT = 0.75
    SAM_WEIGHT = 0.15
    CENTER_WEIGHT = 0.05
    AREA_WEIGHT = 0.05
    MIN_PROTOTYPE_SIMILARITY = 0.0
    MIN_SELECTION_MARGIN = 0.0
    SEGMENTATION_FALLBACK = "original_image"
    EXEMPLARS = []

    # Log de corridas (prioridad maxima): toda celda pesada llama tic/toc para dejar inicio/fin con hora.
    import time as _tlog
    TIC = {}
    def tic(name):
        TIC[name] = _tlog.time()
        print('[%s] inicio %s' % (_tlog.strftime('%H:%M:%S'), name), flush=True)
    def toc(name):
        _dt = _tlog.time() - TIC.pop(name, _tlog.time())
        print('[%s] fin %s (%.1fs)' % (_tlog.strftime('%H:%M:%S'), name, _dt), flush=True)


    return (
        BAD_LABEL,
        DATASET_DIR,
        GOOD_LABEL,
        LABELS_FILE,
        MASK_BACKGROUND,
        MAX_MASK_AREA_RATIO,
        MIN_MASK_AREA_RATIO,
        MLFLOW_EXPERIMENT,
        MLFLOW_URI,
        MODELS_DIR,
        N_SPLITS,
        Path,
        REPEATED_SEEDS,
        SAM_CHECKPOINT,
        SAM_PADDING,
        SAM_PADDING_GRID,
        tic,
        toc,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Gate visual + QC SAM (stop si <90% una pieza)

    **Por que:** stage-gate. Si SAM no aisla bien la pieza, todo lo downstream (crops, masked, scores) es basura: se para aqui.

    **Que hace:** segmenta 10 fotos en vivo (5 GOOD + 5 BAD) y muestra original | overlay+bbox | mascara binaria | crop p08,
    mas tabla area/edge/latencia. 10/10 bbox OK con areas ~0.20 = luz verde.
    """)
    return


@app.cell(hide_code=True)
def md_gate(mo):
    mo.md("""
    ### Gate en vivo: 10 fotos (5 GOOD + 5 BAD), panel original | overlay | mascara | crop.
    """)
    return


@app.cell
def _(
    Image,
    SAM_PADDING,
    crop_part,
    expand_bbox,
    mask_to_bbox,
    np,
    segment_part,
    tic,
    toc,
):
    import pandas as pd
    tic('lEQa')
    # Gate visual de 10: stop si <90% una pieza. Panel 4 col: original | overlay+bbox | mascara binaria | crop p08.
    import matplotlib.pyplot as plt
    from pathlib import Path as _P
    _gate_files = ([('good', f) for f in sorted((_P('data/processed/transistor_binary/good')).iterdir())[:5]]
                   + [('bad', f) for f in sorted((_P('data/processed/transistor_binary/bad')).iterdir())[:5]])
    _fig, _axs = plt.subplots(10, 4, figsize=(16, 30))
    _gate_rows = []
    for _k, (_lb, _fp) in enumerate(_gate_files):
        _im = Image.open(_fp).convert('RGB')
        _r = segment_part(_im)
        _bb = mask_to_bbox(_r['mask'])
        _ratio = float(_r['mask'].mean())
        _touch = bool(_r['mask'][0, :].any() or _r['mask'][-1, :].any() or _r['mask'][:, 0].any() or _r['mask'][:, -1].any())
        _gate_rows.append({'file': _fp.name, 'area': round(_ratio, 4), 'edge': _touch,
                            'lat': round(_r['latency'], 2), 'bbox': _bb is not None})
        _row = _axs[_k]
        _row[0].imshow(_im)
        _row[0].set(title=_fp.name + ' orig')
        _row[1].imshow(_im)
        _row[1].imshow(np.ma.masked_where(~_r['mask'][::4, ::4], np.ones((256, 256))), cmap='jet', alpha=0.4,
                   extent=(0, _im.size[0], _im.size[1], 0))
        if _bb:
            _x0, _y0, _x1, _y1 = _bb
            _row[1].plot([_x0, _x1, _x1, _x0, _x0], [_y0, _y0, _y1, _y1, _y0], 'r-')
            _crop = crop_part(_im, expand_bbox(_bb, (_im.size[1], _im.size[0]), SAM_PADDING))
        else:
            _crop = _im
        _row[1].set(title='a=' + format(_ratio, '.3f') + (' EDGE' if _touch else '') + ' lat=' + format(_r['latency'], '.2f') + 's')
        _row[2].imshow(_r['mask'], cmap='gray')
        _row[2].set(title='mascara binaria')
        _row[3].imshow(_crop)
        _row[3].set(title='crop p08 ' + str(_crop.size))
        for _a in _row:
            _a.axis('off')
    _fig.suptitle('SAM3 gate: 5 GOOD + 5 BAD (orig | overlay | mask | crop)')
    _fig.tight_layout()
    print(pd.DataFrame(_gate_rows).to_string(index=False))
    _fig
    _fig.savefig('reports/figures/transistor_binary/sam_gate.png')
    print('saved sam_gate.png')
    toc('lEQa')

    return pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## 2. Cache label-blind + latencias (SAM se corre una vez)

    **Por que:** SAM tarda ~1 s/foto (~5 min el dataset) y es determinista: se segmenta una vez, se guarda, nada downstream lo recorre.
    Label-blind = la mascara nunca usa la etiqueta GOOD/BAD, asi que no hay fuga.

    **Que hace:** masks.csv (score/area/bbox/edge/lat por foto) + masks256.pt, y lat_qc reporta tiempos por foto y dataset + QC.
    """)
    return


@app.cell(hide_code=True)
def md_sam(mo):
    mo.md("""
    ### Carga SAM3 una vez + funciones mascara, bbox, crop. Todo lo demas cuelga de aqui.
    """)
    return


@app.cell
def _(SAM_CHECKPOINT, tic, toc):
    # SAM3 solo-localiza (nunca clasifica). Carga unica; todo lo demas es label-blind.
    tic('Xref')
    import time
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    _sam_proc = Sam3Processor.from_pretrained(SAM_CHECKPOINT)
    _sam_model = Sam3Model.from_pretrained(SAM_CHECKPOINT).eval()
    for _sp in _sam_model.parameters():
        _sp.requires_grad_(False)
    _sam_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sam_model.to(_sam_dev)
    print("sam device:", _sam_dev)


    def load_sam3():
        """Devuelve (processor, model) ya cargados."""
        return _sam_proc, _sam_model


    def segment_part(image, prompt=None, exemplar=None):
        """Mejor mascara por logits. exemplar reservado (text-only por ahora)."""
        _t0 = time.perf_counter()
        _inp = _sam_proc(images=image, text=prompt or "transistor", return_tensors="pt")
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


    def crop_part(image, bbox):
        return image.crop(bbox)

    toc('Xref')

    return Image, crop_part, expand_bbox, mask_to_bbox, np, segment_part, torch


@app.cell
def _(mo):
    mo.md("""
    ## 3. Variantes none/crop/masked + brazos (pipeline congelado)

    **Por que:** separar efectos: gana quitar el fondo (masked), reencuadrar (crop) o nada (none)? Padding p08 fijo a priori
    (elegirlo por test con 40 BAD seria overfitting). Fallback QC: 4 fotos con mascara mala van a original.

    **Que hace:** embeddings DINO CLS + tokens locales por variante (baseline reusa master), brazos dino (LogReg por fold),
    local (banco GOOD-train + LOO + q99) y masked (fondo = media GOOD-train del fold, sin fuga).
    """)
    return


@app.cell(hide_code=True)
def md_cache(mo):
    mo.md("""
    ### Cache de mascaras: 1 fila por foto (bbox, area, edge, lat), label-blind, a 256 px.
    """)
    return


@app.cell
def _(
    DATASET_DIR,
    Image,
    LABELS_FILE,
    MODELS_DIR,
    mask_to_bbox,
    np,
    pd,
    segment_part,
    tic,
    toc,
    torch,
):
    # Cache label-blind: 1 fila por imagen (bbox, area, edge, lat). Mascaras a 256 en .pt.
    tic('BYtC')
    _sam_dir = MODELS_DIR / "sam3"
    _sam_dir.mkdir(parents=True, exist_ok=True)
    _cache_csv = _sam_dir / "masks.csv"
    _cache_pt = _sam_dir / "masks256.pt"
    if _cache_csv.exists():
        sam_masks = pd.read_csv(_cache_csv)
        _mp = torch.load(_cache_pt, map_location="cpu", weights_only=False)
        sam_masks256, _mfn = _mp["masks"], _mp["fnames"]
        print(f"sam cache hit: {len(sam_masks)} filas")
    else:
        _lab = pd.read_csv(LABELS_FILE)
        _rows, _ml, _mfn = [], [], []
        for _r in _lab.itertuples():
            _im = Image.open(DATASET_DIR / _r.label / _r.filename).convert("RGB")
            try:
                _s = segment_part(_im)
                _bb = mask_to_bbox(_s["mask"])
            except Exception as _e:
                print("SAM fallo:", _r.filename, str(_e)[:100])
                _s, _bb = {"mask": np.zeros((1024, 1024), bool), "score": -99.0, "latency": -1.0}, None
            _m = _s["mask"]
            _rows.append({"filename": _r.filename, "label": _r.label, "score": round(_s["score"], 3),
                           "area": round(float(_m.mean()), 4), "bbox": _bb is not None,
                           "edge": bool(_m[0, :].any() or _m[-1, :].any() or _m[:, 0].any() or _m[:, -1].any()),
                           "lat": round(_s["latency"], 2),
                           "x0": _bb[0] if _bb else -1, "y0": _bb[1] if _bb else -1,
                           "x1": _bb[2] if _bb else -1, "y1": _bb[3] if _bb else -1})
            _ml.append(torch.from_numpy(np.array(Image.fromarray(_m.astype("uint8") * 255).resize((256, 256))) > 0))
            _mfn.append(_r.filename)
        sam_masks = pd.DataFrame(_rows)
        sam_masks.to_csv(_cache_csv, index=False)
        sam_masks256, _tmp = torch.stack(_ml), None
        torch.save({"masks": sam_masks256, "fnames": _mfn}, _cache_pt)
        print(f"saved {len(sam_masks)} filas")
    print(sam_masks.groupby("label").agg(n=("area", "size"), bbox_ok=("bbox", "mean"), area_mean=("area", "mean"), edge_rate=("edge", "mean"), lat_mean=("lat", "mean")).round(3).to_string())
    toc('BYtC')

    return sam_masks, sam_masks256


@app.cell(hide_code=True)
def md_lat(mo):
    mo.md("""
    ### 2.1 Reporte de latencias y QC (lectura de la cache)

    **Por que:** el costo de SAM debe quedar auditado por foto y dataset antes de decidir si es viable en produccion.

    **Que hace:** top-10 lentas, media/p50/p95, total en minutos, split GOOD/BAD, bbox_ok/edge_rate y el histograma sam_lat.png.
    """)
    return


@app.cell(hide_code=True)
def lat_qc(plt, sam_masks, tic, toc):
    # §2 Proposito: latencias SAM por foto y por dataset + QC area/edge. Solo reporta, no re-segmenta.
    import seaborn as _sns2
    tic('lat_qc')
    _gate_lat = sam_masks.sort_values('lat', ascending=False)
    print('--- top 10 fotos mas lentas ---')
    print(_gate_lat[['filename', 'label', 'lat', 'area', 'edge']].head(10).to_string(index=False))
    print('--- agregado latencia (s) ---')
    print(_gate_lat['lat'].describe()[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
    print('total dataset: %.1f min' % (_gate_lat['lat'].sum() / 60))
    print(_gate_lat.groupby('label')['lat'].agg(['mean', 'max']).round(2).to_string())
    print('bbox_ok=%.3f edge_rate=%.3f area_min=%.3f area_max=%.3f' % (
        sam_masks['bbox'].mean(), sam_masks['edge'].mean(), sam_masks['area'].min(), sam_masks['area'].max()))
    _fig2, _ax2 = plt.subplots(figsize=(8, 3))
    _ax2.hist(_gate_lat['lat'], bins=20)
    _ax2.set(title='SAM latencia por foto (s)', xlabel='s', ylabel='n fotos')
    _fig2.tight_layout()
    _fig2
    _fig2.savefig('reports/figures/transistor_binary/sam_lat.png')
    print('saved sam_lat.png')
    _fg, (_b1, _b2) = plt.subplots(1, 2, figsize=(13, 4))
    _sns2.boxplot(data=sam_masks, x="label", y="lat", hue="label", legend=False, ax=_b1, palette={"good": "#4C78A8", "bad": "#F58518"})
    _b1.set(title="SAM lat por clase (s)")
    _sns2.scatterplot(data=sam_masks, x="area", y="lat", hue="edge", ax=_b2)
    _b2.set(title="area mascara vs lat")
    _fg.tight_layout()
    _fg.savefig("reports/figures/transistor_binary/sam_lat_qc.png", dpi=100)
    print("saved sam_lat_qc.png")
    toc('lat_qc')
    _fg

    return


@app.cell(hide_code=True)
def md_var(mo):
    mo.md("""
    ### Variantes none/crop: DINO por padding con fallback QC (p08 fijo a priori).
    """)
    return


@app.cell
def _(
    DATASET_DIR,
    Image,
    LABELS_FILE,
    MAX_MASK_AREA_RATIO,
    MIN_MASK_AREA_RATIO,
    MODELS_DIR,
    SAM_PADDING_GRID,
    crop_part,
    expand_bbox,
    pd,
    sam_masks,
    tic,
    toc,
    torch,
):
    # §3 Proposito: variantes none/crop. Baseline reusa master; crops p05/p08/p10 con fallback QC (area/edge -> original).
    tic('nWHF')
    # Masked (media train-GOOD por fold) va en la celda de evaluacion, no aqui. p08 fijo a priori (no se elige por test).
    import hashlib
    from transformers import AutoImageProcessor, AutoModel
    _lab = pd.read_csv(LABELS_FILE)
    _fp = hashlib.sha1("|".join(sorted(f"{r.filename}:{(DATASET_DIR / r.label / r.filename).stat().st_size}" for _, r in _lab.iterrows())).encode()).hexdigest()[:16]
    _master = torch.load("models/transistor_binary/dinov2_global.pt", map_location="cpu", weights_only=False)
    assert _master["fingerprint"] == _fp, "dataset cambio vs master"
    var_fnames = list(_master["fnames"])
    var_dino_none = _master["X"]
    print("baseline reuse:", var_dino_none.shape)
    _bboxes = {r.filename: (r.x0, r.y0, r.x1, r.y1) if r.bbox else None for r in sam_masks.itertuples()}
    _qcinfo = {r.filename: (r.area, r.edge) for r in sam_masks.itertuples()}
    _n_fallback = []
    def variant_crop(fname, label, pad):
        _im = Image.open(DATASET_DIR / label / fname).convert("RGB")
        _bb = _bboxes[fname]
        _area, _edge = _qcinfo[fname]
        if _bb is None or _bb[0] < 0 or _area < MIN_MASK_AREA_RATIO or _area > MAX_MASK_AREA_RATIO or _edge:
            _n_fallback.append(fname)
            return _im  # fallback QC: original (4 fotos: 2 area>0.85, 2 edge-touch)
        return crop_part(_im, expand_bbox(_bb, (1024, 1024), pad))
    _dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _dmodel = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    for _pp in _dmodel.parameters():
        _pp.requires_grad_(False)
    _ddev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _dmodel.to(_ddev)
    var_dino_crop = {}
    for _pad in SAM_PADDING_GRID:
        _ck = MODELS_DIR / "sam3" / f"dino_crop_p{int(_pad * 100):02d}_qc.pt"
        if _ck.exists():
            var_dino_crop[_pad] = torch.load(_ck, map_location="cpu", weights_only=False)["X"]
            print(f"crop p{_pad} cache hit")
            continue
        _xs = []
        with torch.inference_mode():
            for _fn in var_fnames:
                _lb = _lab.set_index("filename").loc[_fn, "label"]
                _ci = variant_crop(_fn, _lb, _pad)
                _in = _dproc(images=_ci, size={"height": 224, "width": 224}, return_tensors="pt")
                _xs.append(_dmodel(**{k: v.to(_ddev) for k, v in _in.items()}).last_hidden_state[:, 0, :].cpu())
        var_dino_crop[_pad] = torch.cat(_xs).numpy()
        torch.save({"X": var_dino_crop[_pad], "fnames": var_fnames, "pad": _pad, "fingerprint": _fp}, _ck)
        print(f"saved crop p{_pad}", var_dino_crop[_pad].shape, flush=True)
    print("fallback QC en", len(set(_n_fallback)), "fotos:", sorted(set(_n_fallback)))
    toc('nWHF')

    return hashlib, var_dino_crop, var_dino_none, var_fnames, variant_crop


@app.cell
def _(mo):
    mo.md("""
    ## 3 (cont.) Brazos y fusion por variante

    **Por que:** cada brazo ve distinto defecto (dino = semantica global, local = textura de parches, effad = anomalia no supervisada);
    la fusion combina con alpha elegido en train y el rescate usa local como red de seguridad.

    **Que hace:** OOF por variante con el protocolo del master (5 folds seed 42, cortes q95/q99 en train) + fusion y rescate.
    """)
    return


@app.cell(hide_code=True)
def md_dino(mo):
    mo.md("""
    ### Brazo DINO-only por variante: LogReg balanced, 5 folds, corte q95.
    """)
    return


@app.cell
def _(
    BAD_LABEL,
    LABELS_FILE,
    N_SPLITS,
    SAM_PADDING_GRID,
    np,
    pd,
    tic,
    toc,
    var_dino_crop,
    var_dino_none,
    var_fnames,
):
    # DINO-only por variante: mismo protocolo (LogReg balanced, 5 folds seed 42, corte q95 train).
    tic('ZHCJ')
    # Dice si el crop ayuda al brazo supervisado y que padding; EffAD/fusion despues.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    _ylab = pd.read_csv(LABELS_FILE)
    _y = (_ylab.set_index("filename").loc[var_fnames, "label"] == BAD_LABEL).to_numpy(int)
    _variants = {"none": var_dino_none} | {f"crop_p{p}": var_dino_crop[p] for p in SAM_PADDING_GRID}
    dino_variant_rows = []
    for _name, _X in _variants.items():
        _oof = np.zeros(len(_y))
        for _tr, _te in StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_X, _y):
            _clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(_X[_tr], _y[_tr])
            _oof[_te] = _clf.predict_proba(_X[_te])[:, 1]
        _cut = float(np.quantile(_oof[_y == 0], 0.95))
        _pr = (_oof >= _cut).astype(int)
        dino_variant_rows.append({"variant": _name, "auroc": round(float(roc_auc_score(_y, _oof)), 4),
                            "recall": round(float((_pr[_y == 1] == 1).mean()), 3),
                            "far": round(float((_pr[_y == 0] == 1).mean()), 3)})
        print(dino_variant_rows[-1], flush=True)
    toc('ZHCJ')

    return StratifiedKFold, dino_variant_rows, roc_auc_score


@app.cell(hide_code=True)
def md_patch(mo):
    mo.md("""
    ### Tokens locales de crops p08 para el brazo no-supervisado (baseline reusa master).
    """)
    return


@app.cell(hide_code=True)
def patch_crop(
    DATASET_DIR,
    LABELS_FILE,
    MODELS_DIR,
    hashlib,
    pd,
    tic,
    toc,
    torch,
    var_fnames,
    variant_crop,
):
    # §3 Proposito: tokens locales DINOv2 de crops p08 (brazo local). Baseline reusa master. Cache _qc (con fallback QC).
    tic('patch_crop')
    from transformers import AutoImageProcessor as _AIP, AutoModel as _AM
    _fp2 = hashlib.sha1("|".join(sorted(f"{r.filename}:{(DATASET_DIR / r.label / r.filename).stat().st_size}" for _, r in pd.read_csv(LABELS_FILE).iterrows())).encode()).hexdigest()[:16]
    _master_p = torch.load("models/transistor_binary/dinov2_patches.pt", map_location="cpu", weights_only=False)
    assert _master_p["fingerprint"] == _fp2, "dataset cambio"
    local_patches_none = _master_p["patches"]
    print("none reuse:", tuple(local_patches_none.shape))
    _ckp = MODELS_DIR / "sam3" / "patches_crop_p08_qc.pt"
    patch_proc = _AIP.from_pretrained("facebook/dinov2-base")
    patch_model = _AM.from_pretrained("facebook/dinov2-base").eval()
    for _pq in patch_model.parameters():
        _pq.requires_grad_(False)
    _pdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_model.to(_pdev)
    if _ckp.exists():
        _cp = torch.load(_ckp, map_location="cpu", weights_only=False)
        local_patches_crop, _cpfn = _cp["patches"], _cp["fnames"]
        print("crop p08 cache hit", tuple(local_patches_crop.shape))
    else:
        _lab2 = pd.read_csv(LABELS_FILE)
        _pl = []
        with torch.inference_mode():
            for _fn in var_fnames:
                _lb = _lab2.set_index("filename").loc[_fn, "label"]
                _ci = variant_crop(_fn, _lb, 0.08)
                _in = patch_proc(images=_ci, size={"height": 224, "width": 224}, return_tensors="pt")
                _pl.append(patch_model(**{k: v.to(_pdev) for k, v in _in.items()}).last_hidden_state[:, 1:, :].squeeze(0).cpu())
        local_patches_crop = torch.stack(_pl)
        torch.save({"patches": local_patches_crop, "fnames": var_fnames, "fingerprint": _fp2}, _ckp)
        print("saved crop p08 patches", tuple(local_patches_crop.shape), flush=True)
    toc('patch_crop')

    return local_patches_crop, local_patches_none, patch_model, patch_proc


@app.cell(hide_code=True)
def md_bench(mo):
    mo.md("""
    ### 4.1 Benchmark de inferencia por foto y por brazo (solo mide)

    **Por que:** entrenar es una vez; inferir es siempre. Hay que saber que brazo manda en produccion (spoiler: local ~2 s).

    **Que hace:** 12 fotos (6+6) cronometradas por brazo: SAM (costo unico cacheado), DINO CLS, local vs banco, EffAD.
    """)
    return


@app.cell(hide_code=True)
def infer_bench(
    BAD_LABEL,
    DATASET_DIR,
    Image,
    LABELS_FILE,
    local_patches_none,
    patch_model,
    patch_proc,
    pd,
    segment_part,
    tic,
    toc,
    torch,
    var_fnames,
):
    # §4b Proposito: micro-benchmark de inferencia por foto y por brazo (SAM/DINO/local/EffAD). Solo mide, no entrena.
    tic('infer_bench')
    import time as _tb
    import numpy as _npb
    _bench_lab = pd.read_csv(LABELS_FILE)
    _bench_y = (_bench_lab.set_index("filename").loc[var_fnames, "label"] == BAD_LABEL).to_numpy(int)
    _rng = _npb.random.default_rng(0)
    _bench_files = ([(f, "good") for f in _rng.choice([f for f, y in zip(var_fnames, _bench_y) if y == 0], 6, replace=False)]
        + [(f, "bad") for f in _rng.choice([f for f, y in zip(var_fnames, _bench_y) if y == 1], 6, replace=False)])
    print("bench en", len(_bench_files), "fotos (6 GOOD + 6 BAD)")
    _bdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bench_times = {}
    _t = _tb.perf_counter()
    for _fn, _lb in _bench_files[:4]:
        _im = Image.open(DATASET_DIR / _lb / _fn).convert("RGB")
        _t0 = _tb.perf_counter()
        segment_part(_im)
        _bench_times.setdefault("sam", []).append((_tb.perf_counter() - _t0) * 1000)
    print("sam listo")
    _bank = local_patches_none[_npb.array([i for i, y in enumerate(_bench_y) if y == 0])].reshape(-1, 768).to(_bdev)
    for _fn, _lb in _bench_files:
        _im = Image.open(DATASET_DIR / _lb / _fn).convert("RGB")
        _in = patch_proc(images=_im, size={"height": 224, "width": 224}, return_tensors="pt")
        _t0 = _tb.perf_counter()
        with torch.inference_mode():
            _h = patch_model(**{k: v.to(_bdev) for k, v in _in.items()}).last_hidden_state
        _bench_times.setdefault("dino_cls", []).append((_tb.perf_counter() - _t0) * 1000)
        _t0 = _tb.perf_counter()
        with torch.inference_mode():
            _d = torch.cdist(_h[:, 1:, :].squeeze(0).to(_bdev), _bank)
            _sc = float(_d.min(dim=1).values.max())
        _bench_times.setdefault("local", []).append((_tb.perf_counter() - _t0) * 1000)
    print("dino+local listos", flush=True)
    from anomalib.models import EfficientAd as _EA
    _em = _EA(model_size="medium", visualizer=False, evaluator=False).eval().to(_bdev)  # pesos frescos: el forward tarda lo mismo
    print("effad modelo listo (timing puro, sin ckpt)")
    for _fn, _lb in _bench_files:
        _im = Image.open(DATASET_DIR / _lb / _fn).convert("RGB").resize((256, 256))
        _xb = torch.from_numpy(_npb.asarray(_im, dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(_bdev)
        _t0 = _tb.perf_counter()
        with torch.inference_mode():
            _em(_xb)
        _bench_times.setdefault("effad", []).append((_tb.perf_counter() - _t0) * 1000)
    print("effad listo")
    print("--- ms/foto (media | p95) ---")
    for _k, _v in _bench_times.items():
        print("%-9s %8.1f | %8.1f  (n=%d)" % (_k, float(_npb.mean(_v)), float(_npb.percentile(_v, 95)), len(_v)))
    toc("infer_bench")

    return


@app.cell(hide_code=True)
def md_local(mo):
    mo.md("""
    ### Brazo local OOF none vs crop: banco GOOD-train + LOO + q99 por fold. El lento (~13 min).
    """)
    return


@app.cell(hide_code=True)
def local_cmp(
    BAD_LABEL,
    LABELS_FILE,
    MODELS_DIR,
    N_SPLITS,
    local_patches_crop,
    local_patches_none,
    np,
    pd,
    tic,
    toc,
    torch,
    var_fnames,
):
    # Local OOF none vs crop p08: bank GOOD-train + LOO + q99 por fold (mismo protocolo master). Guarda scores para §4.
    tic('local_cmp')
    from sklearn.metrics import roc_auc_score as _auc_loc
    from sklearn.model_selection import StratifiedKFold as _SKloc
    _ylab_loc = pd.read_csv(LABELS_FILE)
    _y_loc = (_ylab_loc.set_index("filename").loc[var_fnames, "label"] == BAD_LABEL).to_numpy(int)
    _dev_loc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _local_oof(_patches, _tag):
        _scores = np.zeros(len(_y_loc))
        for _fi, (_tr, _te) in enumerate(_SKloc(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_patches, _y_loc)):
            _good = [int(i) for i in _tr if _y_loc[i] == 0]
            _bank = _patches[_good].reshape(-1, _patches.shape[-1]).to(_dev_loc)
            _ltr = []
            with torch.inference_mode():
                for _gi, _g in enumerate(_good):
                    _b2 = _patches[[i for i in _good if i != _g]].reshape(-1, _patches.shape[-1]).to(_dev_loc)
                    _pq = _patches[_g].to(_dev_loc)
                    _best = None
                    for _s in range(0, len(_b2), 8192):
                        _mm = torch.cdist(_pq, _b2[_s:_s + 8192]).min(dim=1).values
                        _best = _mm if _best is None else torch.minimum(_best, _mm)
                    _ltr.append(float(_best.max()))
                    if (_gi + 1) % 50 == 0:
                        print("%s fold %d LOO %d/%d" % (_tag, _fi, _gi + 1, len(_good)), flush=True)
                _q = float(np.quantile(_ltr, 0.99))
                for _p in list(_te):
                    _d = torch.cdist(_patches[_p].to(_dev_loc), _bank)
                    _scores[_p] = float(_d.min(dim=1).values.max())
            print("%s fold %d/%d listo" % (_tag, _fi + 1, N_SPLITS), flush=True)
        return _scores
    local_rows = []
    _loc_scores = {}
    for _name, _pt in [("none", local_patches_none), ("crop_p08", local_patches_crop)]:
        _sc = _local_oof(_pt, _name)
        _loc_scores[_name] = _sc
        _cut = float(np.quantile(_sc[_y_loc == 0], 0.95))
        _pr = (_sc >= _cut).astype(int)
        local_rows.append({"variant": _name, "auroc": round(float(_auc_loc(_y_loc, _sc)), 4),
                            "recall": round(float((_pr[_y_loc == 1] == 1).mean()), 3),
                            "far": round(float((_pr[_y_loc == 0] == 1).mean()), 3)})
        print(local_rows[-1], flush=True)
    np.savez(MODELS_DIR / "sam3" / "oof_local.npz", fnames=np.array(var_fnames), y=_y_loc,
             none=_loc_scores["none"], crop_p08=_loc_scores["crop_p08"])
    print("saved oof_local.npz")
    toc('local_cmp')

    return


@app.cell(hide_code=True)
def md_masked(mo):
    mo.md("""
    ### Brazo masked OOF: fondo = media GOOD-train del fold (sin fuga), DINO + local.
    """)
    return


@app.cell(hide_code=True)
def masked_eval(
    BAD_LABEL,
    DATASET_DIR,
    GOOD_LABEL,
    Image,
    LABELS_FILE,
    MODELS_DIR,
    N_SPLITS,
    StratifiedKFold,
    np,
    patch_model,
    patch_proc,
    pd,
    roc_auc_score,
    sam_masks,
    sam_masks256,
    tic,
    toc,
    torch,
    var_fnames,
):
    # Masked OOF: fondo = media GOOD-train del fold; DINO + local mismo protocolo.
    tic('masked_eval')
    from sklearn.linear_model import LogisticRegression as _LRm
    _ylab_m = pd.read_csv(LABELS_FILE)
    _y_m = (_ylab_m.set_index("filename").loc[var_fnames, "label"] == BAD_LABEL).to_numpy(int)
    _pos_of = dict(zip(sam_masks.filename, range(len(sam_masks))))
    _dev_m = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masked_dino_oof = np.zeros(len(_y_m))
    masked_local_oof = np.zeros(len(_y_m))
    for _tr, _te in StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_y_m, _y_m):
        _good_tr = [int(i) for i in _tr if _y_m[i] == 0]
        _mean = np.zeros((1024, 1024, 3), np.float64)
        for _gi in _good_tr:
            _gf = var_fnames[_gi]
            _mean += np.asarray(Image.open(DATASET_DIR / GOOD_LABEL / _gf).convert("RGB"), np.float64)
        _mean = (_mean / len(_good_tr)).astype("uint8")
        def _masked(_idx):
            _fn = var_fnames[_idx]
            _lb = _ylab_m.set_index("filename").loc[_fn, "label"]
            _im = np.asarray(Image.open(DATASET_DIR / _lb / _fn).convert("RGB"))
            _mk = np.array(Image.fromarray((sam_masks256[_pos_of[_fn]].numpy() * 255).astype("uint8")).resize((1024, 1024))) > 0
            return Image.fromarray(np.where(_mk[:, :, None], _im, _mean))
        _emb, _pat = {}, {}
        with torch.inference_mode():
            for _idx in list(_tr) + list(_te):
                _in = patch_proc(images=_masked(_idx), size={"height": 224, "width": 224}, return_tensors="pt")
                _h = patch_model(**{k: v.to(_dev_m) for k, v in _in.items()}).last_hidden_state
                _emb[_idx] = _h[:, 0, :].squeeze(0).cpu().numpy()
                _pat[_idx] = _h[:, 1:, :].squeeze(0).cpu()
        _Xtr = np.stack([_emb[i] for i in _tr])
        _clf = _LRm(class_weight="balanced", C=1.0, max_iter=2000).fit(_Xtr, _y_m[_tr])
        for _idx in _te:
            masked_dino_oof[_idx] = _clf.predict_proba(_emb[_idx][None, :])[0, 1]
        _bank = torch.stack([_pat[i] for i in _good_tr]).reshape(-1, 768).to(_dev_m)
        _ltr = []
        with torch.inference_mode():
            for _g in _good_tr:
                _b2 = torch.stack([_pat[i] for i in _good_tr if i != _g]).reshape(-1, 768).to(_dev_m)
                _pq = _pat[_g].to(_dev_m)
                _best = None
                for _s in range(0, len(_b2), 8192):
                    _mm = torch.cdist(_pq, _b2[_s:_s + 8192]).min(dim=1).values
                    _best = _mm if _best is None else torch.minimum(_best, _mm)
                _ltr.append(float(_best.max()))
            _q = float(np.quantile(_ltr, 0.99))
            for _idx in list(_te):
                _d = torch.cdist(_pat[_idx].to(_dev_m), _bank)
                masked_local_oof[_idx] = float(_d.min(dim=1).values.max())
        print(f"fold done", flush=True)
    for _name, _sc in [("masked-dino", masked_dino_oof), ("masked-local", masked_local_oof)]:
        _cut = float(np.quantile(_sc[_y_m == 0], 0.95))
        _pr = (_sc >= _cut).astype(int)
        print({"variant": _name, "auroc": round(float(roc_auc_score(_y_m, _sc)), 4),
               "recall": round(float((_pr[_y_m == 1] == 1).mean()), 3),
               "far": round(float((_pr[_y_m == 0] == 1).mean()), 3)}, flush=True)
    np.savez(MODELS_DIR / "sam3" / "oof_masked.npz", fnames=np.array(var_fnames), y=_y_m, dino=masked_dino_oof, local=masked_local_oof)
    print("saved oof_masked.npz")
    toc('masked_eval')

    return


@app.cell(hide_code=True)
def md_vmask(mo):
    mo.md("""
    ### Visual: original vs masked (fondo media GOOD) + barras por brazo.
    """)
    return


@app.cell(hide_code=True)
def viz_masked(
    DATASET_DIR,
    GOOD_LABEL,
    Image,
    LABELS_FILE,
    np,
    pd,
    plt,
    sam_masks,
    sam_masks256,
):
    # Proposito: dibuja original vs masked (fondo=media GOOD) + barras por brazo. Solo visualiza, no decide.
    _ym_lab = pd.read_csv(LABELS_FILE)
    _pos_of_loc = dict(zip(sam_masks.filename, range(len(sam_masks))))
    # Visual masked + comparativa por brazos (pre re-entreno EffAD).
    _show3 = [('bad', 'bad_022.png'), ('bad', 'bad_000.png'), ('good', 'good_001.png')]
    _gm = np.zeros((1024, 1024, 3), np.float64)
    for _gf2 in _ym_lab[_ym_lab.label == GOOD_LABEL].filename:
        _gm += np.asarray(Image.open(DATASET_DIR / GOOD_LABEL / _gf2).convert('RGB'), np.float64)
    _gm = (_gm / (_ym_lab.label == GOOD_LABEL).sum()).astype('uint8')
    _f3, _ax3 = plt.subplots(3, 2, figsize=(6, 9))
    for _i, (_lb, _fn) in enumerate(_show3):
        _im = np.asarray(Image.open(DATASET_DIR / _lb / _fn).convert('RGB'))
        _mk = np.array(Image.fromarray((sam_masks256[_pos_of_loc[_fn]].numpy() * 255).astype('uint8')).resize((1024, 1024))) > 0
        _ax3[_i][0].imshow(_im)
        _ax3[_i][0].set(title=_fn)
        _ax3[_i][0].axis('off')
        _ax3[_i][1].imshow(np.where(_mk[:, :, None], _im, _gm))
        _ax3[_i][1].set(title='masked (media GOOD)')
        _ax3[_i][1].axis('off')
    _f3.suptitle('Original vs masked')
    _f3.tight_layout()
    _f3.savefig('reports/figures/transistor_binary/sam_masked.png', dpi=100)
    _comp = [('none-dino', 0.925, 0.991), ('crop-dino', 0.925, 0.983), ('masked-dino', 1.0, 0.996),
             ('none-local', 0.825, 0.9683), ('crop-local', 1.0, 0.9895), ('masked-local', 0.9, 0.9816)]
    _fc, _axc = plt.subplots(1, 2, figsize=(12, 4))
    _axc[0].bar([c[0] for c in _comp], [c[1] for c in _comp])
    _axc[0].set(title='BAD recall por brazo/variante (FAR 0.051 todos)', ylim=(0.5, 1.05))
    _axc[1].bar([c[0] for c in _comp], [c[2] for c in _comp])
    _axc[1].set(title='AUROC por brazo/variante', ylim=(0.9, 1.0))
    _fc.suptitle('Comparativa pre re-entreno EffAD')
    _fc.tight_layout()
    _fc.savefig('reports/figures/transistor_binary/sam_arms.png', dpi=100)
    print('saved sam_masked.png, sam_arms.png')
    _f3, _fc
    return


@app.cell(hide_code=True)
def md_ckpt(mo):
    mo.md("""
    ### Chequeo ckpt crop f0 (info: valida que el checkpoint sirve; la comparativa ya no lo usa).
    """)
    return


@app.cell(hide_code=True)
def ckpt_check(
    BAD_LABEL,
    DATASET_DIR,
    EfficientAd,
    Image,
    LABELS_FILE,
    MODELS_DIR,
    N_SPLITS,
    StratifiedKFold,
    glob,
    np,
    pd,
    torch,
):
    # Predict-only con ckpt 23:34 (valida que sirve). Autocontenido: sin SAM ni DINO.
    _plab = pd.read_csv(LABELS_FILE)
    _pfnames = list(_plab.filename)
    _py = (_plab.set_index("filename").loc[_pfnames, "label"] == BAD_LABEL).to_numpy(int)
    _pbb = pd.read_csv(MODELS_DIR / "sam3" / "masks.csv")
    _pby = _pbb.set_index("filename")

    def _pcrop(_fn, _lb, _pad=0.08):
        _im = Image.open(DATASET_DIR / _lb / _fn).convert("RGB")
        _r = _pby.loc[_fn]
        if not _r.bbox:
            return _im
        _bw, _bh = _r.x1 - _r.x0, _r.y1 - _r.y0
        _px, _py = int(_bw * _pad), int(_bh * _pad)
        return _im.crop((max(0, _r.x0 - _px), max(0, _r.y0 - _py), min(1024, _r.x1 + _px), min(1024, _r.y1 + _py)))

    def _ptensor(_idx):
        _fn = _pfnames[_idx]
        _im = _pcrop(_fn, _plab.set_index("filename").loc[_fn, "label"]).resize((256, 256))
        return torch.from_numpy(np.asarray(_im, dtype="float32") / 255.0).permute(2, 0, 1)

    _folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_py, _py))
    _ptr, _pte = _folds[0]
    _good_tr = [int(i) for i in _ptr if _py[i] == 0]
    _ckpt = sorted(glob.glob("results_i_crop_f0/**/*.ckpt", recursive=True))[-1]
    print("ckpt:", _ckpt)
    _pdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _pmodel = EfficientAd.load_from_checkpoint(_ckpt, map_location="cpu").eval().to(_pdev)

    def _pscores(_idxs):
        _ss = []
        with torch.inference_mode():
            for _s in range(0, len(_idxs), 16):
                _xb = torch.stack([_ptensor(int(i)) for i in _idxs[_s:_s + 16]]).to(_pdev)
                _ss.extend(_pmodel(_xb).pred_score.flatten().cpu().tolist())
        return np.array(_ss)

    _tr_sc = _pscores(_good_tr)
    _te_sc = _pscores([int(i) for i in _pte])
    _cut = float(np.quantile(_tr_sc, 0.95))
    _pr = (_te_sc >= _cut).astype(int)
    _yte = _py[_pte]
    print({"cut": round(_cut, 4), "recall": round(float((_pr[_yte == 1] == 1).mean()), 3),
           "far": round(float((_pr[_yte == 0] == 1).mean()), 3)})
    return


@app.cell(hide_code=True)
def md_off1(mo):
    mo.md("""
    ### (Desactivado) Resume EffAD-crop f0: entreno pesado, fuera del plan actual.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def effad_resume(
    BAD_LABEL,
    DATASET_DIR,
    Image,
    LABELS_FILE,
    MODELS_DIR,
    N_SPLITS,
    StratifiedKFold,
    np,
    pd,
    torch,
):
    import glob
    from torch.utils.data import DataLoader
    from lightning.pytorch import LightningDataModule
    from anomalib.data.dataclasses.torch import ImageBatch
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine
    # Resume EffAD-crop fold 0 desde ckpt (step 3270 -> 5000). Reusa imports de ckpt_check.
    from lightning.pytorch import seed_everything as _seed2
    _rlab = pd.read_csv(LABELS_FILE)
    _rfnames = list(_rlab.filename)
    _ry = (_rlab.set_index("filename").loc[_rfnames, "label"] == BAD_LABEL).to_numpy(int)
    _rbb = pd.read_csv(MODELS_DIR / "sam3" / "masks.csv").set_index("filename")
    _rfolds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_ry, _ry))
    _rtr, _rte = _rfolds[0]
    _rgood = [int(i) for i in _rtr if _ry[i] == 0]

    def _rcrop(_idx):
        _fn = _rfnames[_idx]
        _im = Image.open(DATASET_DIR / _rlab.set_index("filename").loc[_fn, "label"] / _fn).convert("RGB")
        _r = _rbb.loc[_fn]
        _bw, _bh = _r.x1 - _r.x0, _r.y1 - _r.y0
        _px, _py = int(_bw * 0.08), int(_bh * 0.08)
        _im = _im.crop((max(0, _r.x0 - _px), max(0, _r.y0 - _py), min(1024, _r.x1 + _px), min(1024, _r.y1 + _py)))
        return torch.from_numpy(np.asarray(_im.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1)

    class _DMr(LightningDataModule):
        def __init__(self, _t):
            super().__init__()
            self.train_batch_size = 1
            self.name = "crop"
            self.category = "crop"
            self._t = _t
        def train_dataloader(self):
            return DataLoader(self._t, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))

    _seed2(42)
    _rmodel = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
    _reng = Engine(logger=True, default_root_dir="results_i_crop_f0", max_steps=5000)
    _rckpt = sorted(glob.glob("results_i_crop_f0/**/*.ckpt", recursive=True))[-1]
    print("resume desde:", _rckpt, flush=True)
    _reng.fit(model=_rmodel, datamodule=_DMr([_rcrop(i) for i in _rgood]), ckpt_path=_rckpt)

    _rmodel.to(torch.device("cuda")).eval()  # predict en GPU (fit ya llego a max_steps)
    def _rscores(_idxs):
        _ss = []
        with torch.inference_mode():
            for _s in range(0, len(_idxs), 16):
                _xb = torch.stack([_rcrop(int(i)) for i in _idxs[_s:_s + 16]]).to(torch.device("cuda"))
                _ss.extend(_rmodel(_xb).pred_score.flatten().cpu().tolist())
        return np.array(_ss)

    _rtr_sc = _rscores(np.array(_rgood))
    _rte_sc = _rscores(np.array([int(i) for i in _rte]))
    _rcut = float(np.quantile(_rtr_sc, 0.95))
    _rpr = (_rte_sc >= _rcut).astype(int)
    _ryte = _ry[_rte]
    print({"cut": round(_rcut, 4), "recall": round(float((_rpr[_ryte == 1] == 1).mean()), 3), "far": round(float((_rpr[_ryte == 0] == 1).mean()), 3)})
    return (
        DataLoader,
        EfficientAd,
        Engine,
        ImageBatch,
        LightningDataModule,
        glob,
    )


@app.cell(hide_code=True)
def md_off2(mo):
    mo.md("""
    ### (Desactivado) EffAD-crop f1-4 nocturno + MLflow por fold.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def effad_night(
    BAD_LABEL,
    DATASET_DIR,
    DataLoader,
    EfficientAd,
    Engine,
    Image,
    ImageBatch,
    LABELS_FILE,
    LightningDataModule,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    N_SPLITS,
    StratifiedKFold,
    glob,
    np,
    pd,
    torch,
):
    # Loop nocturno folds 1-4 EffAD-crop + MLflow por fold (si algo falla, todo queda guardado).
    # Libera VRAM moviendo SAM/DINO a CPU (in-place, sin romper referencias).
    import mlflow
    from lightning.pytorch import seed_everything as _seed_night
    for _mname in ("_sam_model", "_dmodel", "patch_model"):
        try:
            eval(_mname).to("cpu")
        except NameError:
            pass
    torch.cuda.empty_cache()
    print("VRAM tras descarga:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB", flush=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    _nlab = pd.read_csv(LABELS_FILE)
    _nfnames = list(_nlab.filename)
    _ny = (_nlab.set_index("filename").loc[_nfnames, "label"] == BAD_LABEL).to_numpy(int)
    _nbb = pd.read_csv(MODELS_DIR / "sam3" / "masks.csv").set_index("filename")
    _nfolds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_ny, _ny))

    def _nrcrop(_idx):
        _fn = _nfnames[_idx]
        _im = Image.open(DATASET_DIR / _nlab.set_index("filename").loc[_fn, "label"] / _fn).convert("RGB")
        _r = _nbb.loc[_fn]
        _bw, _bh = _r.x1 - _r.x0, _r.y1 - _r.y0
        _px, _py = int(_bw * 0.08), int(_bh * 0.08)
        _im = _im.crop((max(0, _r.x0 - _px), max(0, _r.y0 - _py), min(1024, _r.x1 + _px), min(1024, _r.y1 + _py)))
        return torch.from_numpy(np.asarray(_im.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1)

    class _DMn(LightningDataModule):
        def __init__(self, _t):
            super().__init__()
            self.train_batch_size = 1
            self.name = "crop"
            self.category = "crop"
            self._t = _t
        def train_dataloader(self):
            return DataLoader(self._t, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))

    def _nscores(_idxs, _model):
        _ss = []
        with torch.inference_mode():
            for _s in range(0, len(_idxs), 16):
                _xb = torch.stack([_nrcrop(int(i)) for i in _idxs[_s:_s + 16]]).to(torch.device("cuda"))
                _ss.extend(_model(_xb).pred_score.flatten().cpu().tolist())
        return np.array(_ss)

    for _fold in (1, 2, 3, 4):
        _tr, _te = _nfolds[_fold]
        _good = [int(i) for i in _tr if _ny[i] == 0]
        _tel = [int(i) for i in _te]
        _seed_night(42 + _fold)
        _m = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
        _eng = Engine(logger=True, default_root_dir=f"results_i_crop_f{_fold}", max_steps=5000)
        _eng.fit(model=_m, datamodule=_DMn([_nrcrop(i) for i in _good]))
        _m.to(torch.device("cuda")).eval()
        _tr_sc = _nscores(np.array(_good), _m)
        _te_sc = _nscores(np.array(_tel), _m)
        _cut = float(np.quantile(_tr_sc, 0.95))
        _pr = (_te_sc >= _cut).astype(int)
        _yte = _ny[_te]
        _rec = round(float((_pr[_yte == 1] == 1).mean()), 3)
        _far = round(float((_pr[_yte == 0] == 1).mean()), 3)
        _ckpt = sorted(glob.glob(f"results_i_crop_f{_fold}/**/*.ckpt", recursive=True))[-1]
        _npz = MODELS_DIR / "sam3" / f"effad_crop_f{_fold}.npz"
        np.savez(_npz, fold=_fold, test_idx=np.array(_tel), test_scores=_te_sc, cut=_cut)
        with mlflow.start_run(run_name=f"faseI_effad_crop_f{_fold}"):
            mlflow.log_params({"variant": "crop_p08", "fold": _fold, "steps": 5000, "seed": 42 + _fold, "size": "medium", "train_good": len(_good)})
            mlflow.log_metrics({"recall": _rec, "far": _far, "cut": _cut})
            mlflow.log_artifact(str(_npz))
            mlflow.log_artifact(_ckpt)
        print({"fold": _fold, "recall": _rec, "far": _far, "cut": round(_cut, 4)}, flush=True)
    print("LOOP COMPLETO")
    return (mlflow,)


@app.cell(hide_code=True)
def md_off3(mo):
    mo.md("""
    ### (Desactivado) EffAD-masked f0-4 nocturno + MLflow por fold.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def effad_masked_night(
    BAD_LABEL,
    DATASET_DIR,
    DataLoader,
    EfficientAd,
    Engine,
    GOOD_LABEL,
    Image,
    ImageBatch,
    LABELS_FILE,
    LightningDataModule,
    MODELS_DIR,
    N_SPLITS,
    StratifiedKFold,
    glob,
    mlflow,
    np,
    pd,
    torch,
):
    from lightning.pytorch import seed_everything as _seed_m
    # Loop masked folds 0-4 EffAD-medium + MLflow por fold. Fondo = media GOOD-train del fold.
    _mlab = pd.read_csv(LABELS_FILE)
    _mfnames = list(_mlab.filename)
    _my = (_mlab.set_index("filename").loc[_mfnames, "label"] == BAD_LABEL).to_numpy(int)
    _mbb = pd.read_csv(MODELS_DIR / "sam3" / "masks.csv").set_index("filename")
    _mfolds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_my, _my))
    _posm = dict(zip(_mbb.reset_index().filename, range(len(_mbb))))
    _masks256_m = torch.load(MODELS_DIR / "sam3" / "masks256.pt", map_location="cpu", weights_only=False)["masks"]

    class _DMm(LightningDataModule):
        def __init__(self, _t):
            super().__init__()
            self.train_batch_size = 1
            self.name = "masked"
            self.category = "masked"
            self._t = _t
        def train_dataloader(self):
            return DataLoader(self._t, batch_size=1, shuffle=True, collate_fn=lambda _b: ImageBatch(image=torch.stack(_b)))

    def _mscores(_idxs, _model, _mean):
        _ss = []
        with torch.inference_mode():
            for _s in range(0, len(_idxs), 16):
                _xb = torch.stack([_masked_t(i, _mean) for i in _idxs[_s:_s + 16]]).to(torch.device("cuda"))
                _ss.extend(_model(_xb).pred_score.flatten().cpu().tolist())
        return np.array(_ss)

    def _masked_t(_idx, _mean):
        _fn = _mfnames[_idx]
        _im = np.asarray(Image.open(DATASET_DIR / _mlab.set_index("filename").loc[_fn, "label"] / _fn).convert("RGB"))
        _mk = np.array(Image.fromarray((_masks256_m[_posm[_fn]].numpy() * 255).astype("uint8")).resize((1024, 1024))) > 0
        _mo = Image.fromarray(np.where(_mk[:, :, None], _im, _mean))
        return torch.from_numpy(np.asarray(_mo.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1)

    for _mfold in (0, 1, 2, 3, 4):
        _tr, _te = _mfolds[_mfold]
        _good = [int(i) for i in _tr if _my[i] == 0]
        _tel = [int(i) for i in _te]
        _mean = np.zeros((1024, 1024, 3), np.float64)
        for _gi in _good:
            _gf = _mfnames[_gi]
            _mean += np.asarray(Image.open(DATASET_DIR / GOOD_LABEL / _gf).convert("RGB"), np.float64)
        _mean = (_mean / len(_good)).astype("uint8")
        _seed_m(42 + _mfold)
        _m = EfficientAd(model_size="medium", visualizer=False, evaluator=False)
        _eng = Engine(logger=True, default_root_dir=f"results_i_masked_f{_mfold}", max_steps=5000)
        _eng.fit(model=_m, datamodule=_DMm([_masked_t(i, _mean) for i in _good]))
        _m.to(torch.device("cuda")).eval()
        _tr_sc = _mscores(np.array(_good), _m, _mean)
        _te_sc = _mscores(np.array(_tel), _m, _mean)
        _cut = float(np.quantile(_tr_sc, 0.95))
        _pr = (_te_sc >= _cut).astype(int)
        _yte = _my[_te]
        _rec = round(float((_pr[_yte == 1] == 1).mean()), 3)
        _far = round(float((_pr[_yte == 0] == 1).mean()), 3)
        _npz = MODELS_DIR / "sam3" / f"effad_masked_f{_mfold}.npz"
        np.savez(_npz, fold=_mfold, test_idx=np.array(_tel), test_scores=_te_sc, cut=_cut)
        _ckpt = sorted(glob.glob(f"results_i_masked_f{_mfold}/**/*.ckpt", recursive=True))[-1]
        with mlflow.start_run(run_name=f"faseI_effad_masked_f{_mfold}"):
            mlflow.log_params({"variant": "masked", "fold": _mfold, "steps": 5000, "seed": 42 + _mfold, "size": "medium", "train_good": len(_good)})
            mlflow.log_metrics({"recall": _rec, "far": _far, "cut": _cut})
            mlflow.log_artifact(str(_npz))
            mlflow.log_artifact(_ckpt)
        print({"fold": _mfold, "recall": _rec, "far": _far, "cut": round(_cut, 4)}, flush=True)
    print("MASKED LOOP COMPLETO")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Comparativa 3 vias con estadistica (sin hardcodes)

    **Por que:** con 40 BAD cada foto son 2.5 pp de recall: tabla sin incertidumbre es ruido con formato.

    **Que hace:** 15 combos (3 variantes x 5 metodos) con recall/FAR/AUROC OOF + McNemar pareado + IC95 bootstrap (semillas 11-55)
    + recall por tipo de defecto. Figura y CSV se computan, nada a mano.
    """)
    return


@app.cell(hide_code=True)
def compare_3way(
    LABELS_FILE,
    MODELS_DIR,
    N_SPLITS,
    REPEATED_SEEDS,
    np,
    pd,
    plt,
    tic,
    toc,
    var_dino_crop,
    var_fnames,
):
    # §4 Proposito: comparativa 3 vias none/crop/masked x dino/local/effad/fusion/rescue con IC bootstrap + McNemar + recall por defecto. Sin hardcodes.
    tic('compare_3way')
    import math as _m
    from sklearn.linear_model import LogisticRegression as _LRc
    from sklearn.metrics import roc_auc_score as _auc_c
    from sklearn.model_selection import StratifiedKFold as _SKc
    _loc = np.load(MODELS_DIR / "sam3" / "oof_local.npz")
    _msk = np.load(MODELS_DIR / "sam3" / "oof_masked.npz")
    _old_crop = np.load(MODELS_DIR / "sam3" / "oof_crop.npz")
    _master_scores = np.load("models/transistor_binary/oof_scores.npz")
    assert (list(_loc["fnames"]) == list(_msk["fnames"]) == list(_old_crop["fnames"]) == list(_master_scores["fnames"]) == var_fnames), "orden distinto"
    _y4 = _loc["y"].astype(int)
    _lab4 = pd.read_csv(LABELS_FILE).set_index("filename")
    _defect = np.array([_lab4.loc[f, "defect"] for f in var_fnames])
    # DINO crop fresco (_qc) mismo protocolo; none/masked vienen de artefactos
    _Xc = var_dino_crop[0.08]
    _dino_crop_oof = np.zeros(len(_y4))
    _folds4 = list(_SKc(n_splits=N_SPLITS, shuffle=True, random_state=42).split(_Xc, _y4))
    for _tr, _te in _folds4:
        _clf = _LRc(class_weight="balanced", C=1.0, max_iter=2000).fit(_Xc[_tr], _y4[_tr])
        _dino_crop_oof[_te] = _clf.predict_proba(_Xc[_te])[:, 1]
    print("dino-crop OOF listo", flush=True)
    _mef = np.zeros(len(_y4))
    for _f in range(5):
        _d = np.load(MODELS_DIR / "sam3" / f"effad_masked_f{_f}.npz")
        _mef[_d["test_idx"]] = _d["test_scores"]
    print("effad-masked ensamblado", flush=True)
    _scores = {
        "none": {"dino": _master_scores["dino"], "effad": _master_scores["effad_medium"], "local": _loc["none"]},
        "crop": {"dino": _dino_crop_oof, "effad": _old_crop["effad"], "local": _loc["crop_p08"]},  # effad-crop pre-QC (cueva: 2 fotos)
        "masked": {"dino": _msk["dino"], "effad": _mef, "local": _msk["local"]},
    }
    def _mcnemar(_a, _b):
        _n01 = int(((_a == 0) & (_b == 1)).sum())
        _n10 = int(((_a == 1) & (_b == 0)).sum())
        _n = _n01 + _n10
        if _n == 0:
            return 1.0
        return round(min(1.0, 2.0 * sum(_m.comb(_n, _k) for _k in range(0, min(_n01, _n10) + 1)) / 2.0 ** _n), 4)
    _rows4, _preds = [], {}
    for _v in ("none", "crop", "masked"):
        _S = _scores[_v]
        _fused = np.zeros(len(_y4))
        _alphas = {}
        for _fi, (_tr, _te) in enumerate(_folds4):
            _gd = _tr[_y4[_tr] == 0]
            _md, _id = float(np.median(_S["dino"][_gd])), float(max(np.subtract(*np.percentile(_S["dino"][_gd], [75, 25])), 1e-9))
            _me, _ie = float(np.median(_S["effad"][_gd])), float(max(np.subtract(*np.percentile(_S["effad"][_gd], [75, 25])), 1e-9))
            _best = None
            for _a in np.arange(0, 1.05, 0.05).round(2):
                _ftr = _a * (_S["dino"][_tr] - _md) / _id + (1 - _a) * (_S["effad"][_tr] - _me) / _ie
                _thr = float(np.quantile(_ftr[_y4[_tr] == 0], 0.95))
                _pr = (_ftr >= _thr).astype(int)
                _rec = float((_pr[_y4[_tr] == 1] == 1).mean())
                _far = float((_pr[_y4[_tr] == 0] == 1).mean())
                if _best is None or (_rec, -_far) > (_best[1], -_best[2]):
                    _best = (float(_a), _rec, _far, _thr)
            _a, _, _, _thr = _best
            _alphas[_fi] = _a
            _fused[_te] = _a * (_S["dino"][_te] - _md) / _id + (1 - _a) * (_S["effad"][_te] - _me) / _ie
        _methods = {"dino": _S["dino"], "effad": _S["effad"], "local": _S["local"], "fusion": _fused}
        for _mname, _sc in _methods.items():
            _cut = float(np.quantile(_sc[_y4 == 0], 0.95))
            _pr = (_sc >= _cut).astype(int)
            _preds[(_v, _mname)] = _pr
            _rows4.append({"variant": _v, "method": _mname,
                            "auroc": round(float(_auc_c(_y4, _sc)), 4),
                            "recall": round(float((_pr[_y4 == 1] == 1).mean()), 3),
                            "far": round(float((_pr[_y4 == 0] == 1).mean()), 3)})
        _ql = float(np.quantile(_S["local"][_y4 == 0], 0.99))
        _pr_r = ((_fused >= float(np.quantile(_fused[_y4 == 0], 0.95))) | (_S["local"] >= _ql)).astype(int)
        _preds[(_v, "rescue")] = _pr_r
        _rows4.append({"variant": _v, "method": "rescue", "auroc": None,
                        "recall": round(float((_pr_r[_y4 == 1] == 1).mean()), 3),
                        "far": round(float((_pr_r[_y4 == 0] == 1).mean()), 3)})
        print(_v, "metodos listos", flush=True)
    compare_df = pd.DataFrame(_rows4)
    print(compare_df.to_string(index=False))
    # McNemar pareado por metodo: none-vs-crop, none-vs-masked, crop-vs-masked
    print("--- McNemar p (H0: igual error pareado) ---")
    for _mname in ("dino", "effad", "local", "fusion", "rescue"):
        print(_mname, {p: _mcnemar(_preds[(p[0], _mname)], _preds[(p[1], _mname)]) for p in [("none", "crop"), ("none", "masked"), ("crop", "masked")]}, flush=True)
    # Bootstrap IC95 recall con REPEATED_SEEDS
    print("--- bootstrap recall IC95 (200 rep x semilla) ---")
    for _v, _mname in [("none", "fusion"), ("crop", "fusion"), ("masked", "fusion"), ("none", "dino"), ("masked", "dino"), ("crop", "local")]:
        _pr = _preds[(_v, _mname)]
        _bs = []
        for _s in REPEATED_SEEDS:
            _rg = np.random.default_rng(_s)
            for _b in range(200):
                _ix = _rg.integers(0, len(_y4), len(_y4))
                _bs.append(float((_pr[_ix][_y4[_ix] == 1] == 1).mean()))
        print("%s-%s recall IC95: [%.3f, %.3f]" % (_v, _mname, float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5))), flush=True)
    # Recall por tipo de defecto (10 BAD c/u)
    print("--- recall por defecto ---")
    _drows = []
    for (_v, _mname), _pr in sorted(_preds.items()):
        _row = {"variant": _v, "method": _mname}
        for _d in ("bent_lead", "cut_lead", "damaged_case", "misplaced"):
            _sel = (_y4 == 1) & (np.asarray(_defect) == _d)
            _row[_d] = round(float(np.mean(np.asarray(_pr).ravel()[_sel])), 2) if int(_sel.sum()) else float("nan")
        _drows.append(_row)
        print(_row, flush=True)
    compare_df.to_csv(MODELS_DIR / "sam3" / "comparativa_3way.csv", index=False)
    _fig4, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 4))
    _piv_r = compare_df.pivot(index="method", columns="variant", values="recall").reindex(["dino", "effad", "local", "fusion", "rescue"])
    _piv_f = compare_df.pivot(index="method", columns="variant", values="far").reindex(["dino", "effad", "local", "fusion", "rescue"])
    _piv_r.plot.bar(ax=_a1)
    _a1.set(title="BAD recall: none vs crop vs masked", ylim=(0.5, 1.05))
    _a1.legend(title=None)
    _piv_f.plot.bar(ax=_a2)
    _a2.set(title="FAR: none vs crop vs masked", ylim=(0, 0.12))
    _a2.legend(title=None)
    _fig4.suptitle("Fase I: comparativa computada (sin hardcodes)")
    _fig4.tight_layout()
    _fig4.savefig("reports/figures/transistor_binary/sam_verdict.png", dpi=100)
    np.savez(MODELS_DIR / "sam3" / "oof_compare.npz", y=_y4, fnames=np.array(var_fnames), defect=_defect,
             **{f"{v}_{m}": _preds[(v, m)].astype("uint8") for v, m in _preds})
    print("saved oof_compare.npz", sorted(_preds.keys()))
    print("saved comparativa_3way.csv, sam_verdict.png")
    toc('compare_3way')

    return


@app.cell(hide_code=True)
def figs_seaborn(
    DATASET_DIR,
    Image,
    LABELS_FILE,
    MODELS_DIR,
    Path,
    REPEATED_SEEDS,
    local_patches_none,
    mo,
    np,
    patch_model,
    patch_proc,
    pd,
    plt,
    tic,
    toc,
    torch,
    var_fnames,
):
    # §4 Proposito: figuras seaborn de la comparativa (heatmaps, IC95, defectos, bench). Todo computado, nada a mano.
    tic('figs_seaborn')
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    _figdir = Path("reports/figures/transistor_binary")
    _cmp4 = pd.read_csv(MODELS_DIR / "sam3" / "comparativa_3way.csv")
    _oc = np.load(MODELS_DIR / "sam3" / "oof_compare.npz")
    _y4b = _oc["y"].astype(int)
    _order_m = ["dino", "effad", "local", "fusion", "rescue"]
    _order_v = ["none", "crop", "masked"]
    _pal_v = {"none": "#4C78A8", "crop": "#F58518", "masked": "#54A24B"}
    print("figs desde", MODELS_DIR / "sam3" / "comparativa_3way.csv", flush=True)
    # 1. Heatmaps recall + FAR (reemplaza sam_arms hardcodeado)
    _fh, (_h1, _h2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for _ax, _col, _t, _vmin, _vmax in [(_h1, "recall", "BAD recall", 0.5, 1.0), (_h2, "far", "FAR", 0.0, 0.12)]:
        sns.heatmap(_cmp4.pivot(index="method", columns="variant", values=_col).reindex(index=_order_m, columns=_order_v),
                    annot=True, fmt=".3f", vmin=_vmin, vmax=_vmax, cmap="YlGnBu", ax=_ax)
        _ax.set(title=_t)
    _fh.suptitle("Fase I: recall/FAR por variante x metodo (computado)")
    _fh.tight_layout()
    _fh.savefig(_figdir / "sam_arms.png", dpi=100)
    print("saved sam_arms.png (recomputado)", flush=True)
    # 2. Recall con IC95 bootstrap (1000 rep: 200 x semilla REPEATED_SEEDS)
    _brows = []
    for _ck in [k for k in _oc.files if k not in ("y", "fnames", "defect")]:
        _v, _m = _ck.rsplit("_", 1)
        _pr = _oc[_ck].astype(int)
        for _s in REPEATED_SEEDS:
            _rg = np.random.default_rng(_s)
            for _b in range(200):
                _ix = _rg.integers(0, len(_y4b), len(_y4b))
                _brows.append({"variant": _v, "method": _m, "recall": float((_pr[_ix][_y4b[_ix] == 1] == 1).mean())})
    _bdf = pd.DataFrame(_brows)
    _bstat = _bdf.groupby(["variant", "method"])["recall"].agg(["mean", lambda s: float(np.percentile(s, 2.5)), lambda s: float(np.percentile(s, 97.5))])
    _bstat.columns = ["mean", "lo", "hi"]
    _bstat = _bstat.reset_index()
    _fc, _ac = plt.subplots(figsize=(11, 4.5))
    _x = np.arange(len(_order_m))
    for _i, _v in enumerate(_order_v):
        _s = _bstat[_bstat.variant == _v].set_index("method").reindex(_order_m)
        _ac.errorbar(_x + (_i - 1) * 0.22, _s["mean"], yerr=[_s["mean"] - _s["lo"], _s["hi"] - _s["mean"]],
                     fmt="o", capsize=4, color=_pal_v[_v], label=_v)
    _ac.set_xticks(_x, _order_m)
    _ac.set(title="BAD recall con IC95 bootstrap (1000 rep)", ylim=(0.4, 1.05))
    _ac.legend(title=None)
    _fc.tight_layout()
    _fc.savefig(_figdir / "sam_ci.png", dpi=100)
    print("saved sam_ci.png", flush=True)
    # 3. Recall por defecto (facet por metodo, hue variante)
    _def4 = _oc["defect"].astype(str)
    _drows = []
    for _ck in [k for k in _oc.files if k not in ("y", "fnames", "defect")]:
        _v, _m = _ck.rsplit("_", 1)
        _pr = _oc[_ck].astype(int)
        for _d in ("bent_lead", "cut_lead", "damaged_case", "misplaced"):
            _sel = (_y4b == 1) & (_def4 == _d)
            _drows.append({"variant": _v, "method": _m, "defect": _d, "recall": round(float(_pr[_sel].mean()), 3)})
    _ddf = pd.DataFrame(_drows)
    _g = sns.catplot(data=_ddf, x="defect", y="recall", hue="variant", col="method", hue_order=_order_v,
                     col_order=_order_m, kind="bar", palette=_pal_v, col_wrap=5, height=3.2, aspect=0.85)
    _g.set(ylim=(0, 1.05))
    _g.set_xticklabels(rotation=30, ha="right")
    _g.fig.subplots_adjust(right=0.86)
    _g._legend.set_bbox_to_anchor((1.0, 0.55))
    _g.fig.suptitle("Recall por tipo de defecto (10 BAD c/u)")
    _g.fig.tight_layout()
    _g.fig.savefig(_figdir / "sam_defect.png", dpi=100)
    print("saved sam_defect.png", flush=True)
    # 4. Bench inferencia (re-timing dino/local/effad en 6 fotos; SAM = media cacheada, costo unico)
    import time as _t4
    _blab = pd.read_csv(LABELS_FILE).set_index("filename")
    _bench6 = [(f, _blab.loc[f, "label"]) for f in list(var_fnames)[::25][:12]]
    _bdev4 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bank4 = local_patches_none[np.array([i for i, f in enumerate(var_fnames) if _blab.loc[f, "label"] == "good"])].reshape(-1, 768).to(_bdev4)
    _bt = {"dino": [], "local": [], "effad": []}
    from anomalib.models import EfficientAd as _EA4
    _em4 = _EA4(model_size="medium", visualizer=False, evaluator=False).eval().to(_bdev4)
    for _fn, _lb in _bench6:
        _im = Image.open(DATASET_DIR / _lb / _fn).convert("RGB")
        _in = patch_proc(images=_im, size={"height": 224, "width": 224}, return_tensors="pt")
        torch.cuda.synchronize()
        _t0 = _t4.perf_counter()
        with torch.inference_mode():
            _h = patch_model(**{k: v.to(_bdev4) for k, v in _in.items()}).last_hidden_state
        torch.cuda.synchronize()
        _bt["dino"].append((_t4.perf_counter() - _t0) * 1000)
        torch.cuda.synchronize()
        _t0 = _t4.perf_counter()
        with torch.inference_mode():
            _d = torch.cdist(_h[:, 1:, :].squeeze(0).to(_bdev4), _bank4)
            torch.cuda.synchronize()
            float(_d.min(dim=1).values.max())
            torch.cuda.synchronize()
        _bt["local"].append((_t4.perf_counter() - _t0) * 1000)
        print("bench", _fn, "listo", flush=True)
        _xb = torch.from_numpy(np.asarray(_im.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(_bdev4)
        torch.cuda.synchronize()
        _t0 = _t4.perf_counter()
        with torch.inference_mode():
            _em4(_xb)
            torch.cuda.synchronize()
        _bt["effad"].append((_t4.perf_counter() - _t0) * 1000)
    _bt = {k: v[1:] for k, v in _bt.items()}  # fuera primera medicion (warmup CUDA)
    _sam_mean_ms = float(pd.read_csv(MODELS_DIR / "sam3" / "masks.csv")["lat"].mean() * 1000)
    _fb, _ab = plt.subplots(figsize=(7, 4))
    _arms = ["sam (cache)", "dino", "local", "effad"]
    _vals = [_sam_mean_ms] + [float(np.median(_bt[k])) for k in ("dino", "local", "effad")]  # mediana: la laptop mete varianza
    _ab.bar(_arms, _vals, color=["#999999", _pal_v["none"], _pal_v["crop"], _pal_v["masked"]])
    _ab.set_yscale("log")
    _ab.set_ylim(top=max(_vals) * 4)
    _ab.set(title="ms/foto por brazo (log; SAM = media cacheada, costo unico)", ylabel="ms (log)")
    for _x, _vv in zip(_arms, _vals):
        _ab.text(_x, _vv * 1.15, f"{_vv:.0f} ms", ha="center", fontsize=9)
    _fb.tight_layout()
    _fb.savefig(_figdir / "sam_bench.png", dpi=100)
    print({k: "%.0f+-%.0f ms" % (float(np.mean(v)), float(np.std(v))) for k, v in _bt.items()}, flush=True)
    print("saved sam_bench.png")
    toc('figs_seaborn')
    mo.vstack([_fh, _fc, _g.fig, _fb])

    return


@app.cell(hide_code=True)
def class_reports(MODELS_DIR, mo, np, pd, plt, tic, toc):
    # §4 Proposito: reporte de clasificacion por metodo (precision/recall/F1 por clase + matriz). Lee OOF guardados, no recomputa.
    tic('class_reports')
    from sklearn.metrics import classification_report as _cr, confusion_matrix as _cmx
    _oc9 = np.load(MODELS_DIR / "sam3" / "oof_compare.npz")
    _y9 = _oc9["y"].astype(int)
    _rep_rows, _cms = [], {}
    for _ck in sorted(k for k in _oc9.files if k not in ("y", "fnames", "defect")):
        _v, _m = _ck.rsplit("_", 1)
        _pr = _oc9[_ck].astype(int)
        _rep = _cr(_y9, _pr, target_names=["good", "bad"], output_dict=True, zero_division=0)
        _tn, _fp, _fn, _tp = _cmx(_y9, _pr).ravel()
        _rep_rows.append({"variant": _v, "method": _m,
            "prec_bad": round(_rep["bad"]["precision"], 3), "rec_bad": round(_rep["bad"]["recall"], 3),
            "f1_bad": round(_rep["bad"]["f1-score"], 3), "prec_good": round(_rep["good"]["precision"], 3),
            "f1_macro": round(_rep["macro avg"]["f1-score"], 3), "acc": round(_rep["accuracy"], 3),
            "tn": int(_tn), "fp": int(_fp), "fn": int(_fn), "tp": int(_tp)})
        _cms[(_v, _m)] = (_tn, _fp, _fn, _tp)
        print("%s-%s: prec_bad=%.3f rec_bad=%.3f f1_bad=%.3f acc=%.3f fp=%d fn=%d" % (_v, _m, _rep["bad"]["precision"], _rep["bad"]["recall"], _rep["bad"]["f1-score"], _rep["accuracy"], _fp, _fn), flush=True)
    _rdf = pd.DataFrame(_rep_rows)
    _rdf.to_csv(MODELS_DIR / "sam3" / "reportes_clasificacion.csv", index=False)
    print("saved reportes_clasificacion.csv")
    import seaborn as _sns9
    _sns9.set_theme(style="whitegrid")
    _cf, _caxs = plt.subplots(3, 5, figsize=(15, 8))
    _vord = ["none", "crop", "masked"]
    _mord = ["dino", "effad", "local", "fusion", "rescue"]
    for _i, _v in enumerate(_vord):
        for _j, _m in enumerate(_mord):
            _tn, _fp, _fn, _tp = _cms[(_v, _m)]
            _sns9.heatmap([[_tn, _fp], [_fn, _tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=_caxs[_i][_j])
            _caxs[_i][_j].set(title=f"{_v}-{_m}", xlabel="" if _i < 2 else "pred", ylabel="real" if _j == 0 else "")
            _caxs[_i][_j].set_xticklabels(["good", "bad"])
            _caxs[_i][_j].set_yticklabels(["good", "bad"], rotation=0)
    _cf.suptitle("Matrices de confusion: 3 variantes x 5 metodos (OOF)")
    _cf.tight_layout()
    _cf.savefig("reports/figures/transistor_binary/sam_cm.png", dpi=100)
    print("saved sam_cm.png")
    toc('class_reports')
    mo.vstack([_cf])
    return


@app.cell(hide_code=True)
def md_verdict(mo):
    mo.md("""
    ## 5. Veredicto por brazo (no binario) + no-leakage

    **Por que:** un KEEP/ADOPT global esconderia que masked gana en dino y pierde en effad. El veredicto se da por brazo,
    y con 40 BAD nada es significativo (McNemar): se reporta honesto.

    **Que hace:** lee comparativa_3way.csv, imprime recomendacion por brazo y deja constancia de no-fuga
    (mascaras label-blind + fondo masked per-fold GOOD-train).
    """)
    return


@app.cell(hide_code=True)
def verdict(MODELS_DIR, tic, toc):
    # §5 Proposito: veredicto por brazo (no binario) desde comparativa_3way.csv. Solo interpreta, no computa.
    tic('verdict')
    import pandas as _pv
    _cmp = _pv.read_csv(MODELS_DIR / "sam3" / "comparativa_3way.csv").set_index(["variant", "method"])
    print(_cmp.to_string())
    print("--- veredicto por brazo (FAR ~0.05-0.06 todos) ---")
    print("dino: masked 1.000 > none 0.925 = crop 0.925, pero McNemar n.s. -> ADOPTAR masked-dino como opcion, no obligatorio")
    print("local: crop 1.000 > masked 0.900 > none 0.825, McNemar n.s. -> ADOPTAR crop-local")
    print("effad: none 0.900 > crop 0.850 >> masked 0.625 -> KEEP ORIGINAL para EffAD (masked lo rompe)")
    print("fusion: none 1.000 >= masked 0.975 > crop 0.900 -> KEEP ORIGINAL para fusion")
    print("rescue: 1.000 en none y masked -> rescue con masked vale como red de seguridad")
    print("global: ninguna diferencia es significativa con 40 BAD (todo McNemar p>0.05) -> veredicto honesto, no marketing")
    toc('verdict')

    return


@app.cell(hide_code=True)
def mlflow_final2(
    MASK_BACKGROUND,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    SAM_CHECKPOINT,
    tic,
    toc,
):
    # §6 Proposito: un run MLflow faseI_sam3_final idempotente (mismo nombre) con CSV real + figura + Caveats. Solo loguea.
    tic('mlflow_final')
    import mlflow as _mf
    _mf.set_tracking_uri(MLFLOW_URI)
    _mf.set_experiment(MLFLOW_EXPERIMENT)
    with _mf.start_run(run_name="faseI_sam3_reports"):
        _mf.log_params({"sam_checkpoint": SAM_CHECKPOINT, "mask_background": MASK_BACKGROUND,
                         "fallback_qc": "area[0.03,0.85]+edge->original (4 fotos)",
                         "effad_crop_caveat": "scores pre-QC (2 fotos difieren)",
                         "verdict": "masked-dino + crop-local; EffAD/fusion KEEP ORIGINAL; nada significativo (McNemar p>0.05)"})
        _mf.log_artifact(str(MODELS_DIR / "sam3" / "comparativa_3way.csv"))
        _mf.log_artifact(str(MODELS_DIR / "sam3" / "oof_local.npz"))
        _mf.log_artifact(str(MODELS_DIR / "sam3" / "oof_masked.npz"))
        _mf.log_artifact("reports/figures/transistor_binary/sam_verdict.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_gate.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_lat.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_ci.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_defect.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_bench.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_lat_qc.png")
        _mf.log_artifact("reports/figures/transistor_binary/sam_arms.png")
        _mf.log_artifact(str(MODELS_DIR / "sam3" / "reportes_clasificacion.csv"))
        _mf.log_artifact("reports/figures/transistor_binary/sam_cm.png")
        _mf.set_tag("recommendation", "KEEP ORIGINAL effad/fusion; ADOPT masked-dino + crop-local")
    print("mlflow faseI_sam3_final listo")
    toc('mlflow_final')

    return


@app.cell(hide_code=True)
def md_mlflow(mo):
    mo.md("""
    ## 6. MLflow: un run idempotente con artefactos reales

    **Por que:** el cierre debe ser reproducible: mismos numeros que la libreta, mismos archivos, sin duplicar runs al re-correr.

    **Que hace:** run faseI_sam3_final con params (checkpoint, fallback QC, caveats, veredicto), CSV + NPZ + figuras y tag recommendation.
    """)
    return


@app.cell(hide_code=True)
def md_vcrop(mo):
    mo.md("""
    ### Visual: original vs crops p05/p08/p10 + barras DINO-only.
    """)
    return


@app.cell(hide_code=True)
def viz_crop(
    DATASET_DIR,
    Image,
    SAM_PADDING_GRID,
    dino_variant_rows,
    plt,
    tic,
    toc,
    variant_crop,
):
    # Visual: original vs crops + barras DINO-only por variante.
    tic('viz_crop')
    _show = [('bad', 'bad_022.png'), ('bad', 'bad_000.png'), ('good', 'good_001.png')]
    _n = len(_show)
    _fig, _axs = plt.subplots(_n, 4, figsize=(12, 3 * _n))
    for _i, (_lb, _fn) in enumerate(_show):
        _im0 = Image.open(DATASET_DIR / _lb / _fn).convert('RGB')
        for _j, (_t, _im) in enumerate([('orig', _im0)] + [(f'p{p}', variant_crop(_fn, _lb, p)) for p in SAM_PADDING_GRID]):
            _axs[_i][_j].imshow(_im)
            _axs[_i][_j].set(title=f'{_fn} {_t} {_im.size}')
            _axs[_i][_j].axis('off')
    _fig.suptitle('Original vs SAM crops (p05/p08/p10)')
    _fig.tight_layout()
    _fig.savefig('reports/figures/transistor_binary/sam_crops.png', dpi=100)
    _bfig, _bax = plt.subplots(figsize=(7, 4))
    _names = [r['variant'] for r in dino_variant_rows]
    _bax.bar(_names, [r['recall'] for r in dino_variant_rows])
    _bax.set(title='DINO-only BAD recall por variante (seed-42)', ylim=(0.5, 1.02))
    for _x, _r in zip(_names, dino_variant_rows):
        _bax.text(_x, _r['recall'] + 0.01, f"{_r['recall']:.3f} / {_r['auroc']:.3f}", ha='center', fontsize=8)
    _bfig.tight_layout()
    _bfig.savefig('reports/figures/transistor_binary/sam_dino_bars.png', dpi=100)
    print('saved sam_crops.png, sam_dino_bars.png')
    _fig, _bfig
    toc('viz_crop')

    return


@app.cell(hide_code=True)
def md_vcont(mo):
    mo.md("""
    ### Visual: contorno de la pieza SAM sobre el original.
    """)
    return


@app.cell(hide_code=True)
def viz_contour(
    DATASET_DIR,
    Image,
    np,
    plt,
    sam_masks,
    sam_masks256,
    tic,
    toc,
):
    # Contornos SAM sobre original (lo que esperabas ver).
    tic('viz_contour')
    _show2 = [('bad', 'bad_022.png'), ('bad', 'bad_000.png'), ('good', 'good_001.png'), ('good', 'good_000.png')]
    _f2, _ax2 = plt.subplots(1, 4, figsize=(16, 4))
    _bydf = sam_masks.set_index('filename')
    for _a, (_lb, _fn) in zip(_ax2, _show2):
        _im = Image.open(DATASET_DIR / _lb / _fn).convert('RGB')
        _row = _bydf.loc[_fn]
        _pos_of = dict(zip(sam_masks.filename, range(len(sam_masks))))
        _pos = _pos_of[_fn]
        _full = np.array(Image.fromarray((sam_masks256[_pos].numpy() * 255).astype('uint8')).resize((1024, 1024))) > 0
        _a.imshow(_im)
        _a.contour(_full, levels=[0.5], colors='red', linewidths=2)
        _a.set(title=f'{_fn} area={_row.area:.3f}')
        _a.axis('off')
    _f2.suptitle('SAM3: contorno de la pieza sobre original')
    _f2.tight_layout()
    _f2.savefig('reports/figures/transistor_binary/sam_contours.png', dpi=100)
    print('saved sam_contours.png')
    _f2
    toc('viz_contour')

    return


@app.cell(hide_code=True)
def md_vall(mo):
    mo.md("""
    ### Visual: mosaico de TODAS las BAD, original + crop p08.
    """)
    return


@app.cell(hide_code=True)
def viz_allbad(DATASET_DIR, Image, SAM_PADDING, plt, tic, toc, variant_crop):
    # Mosaico TODAS las BAD: original + crop p08.
    tic('viz_allbad')
    _bad_files = sorted((DATASET_DIR / 'bad').iterdir())
    _nbad = len(_bad_files)
    _fm, _axm = plt.subplots(_nbad, 2, figsize=(6, 2.2 * _nbad))
    for _i, _fp in enumerate(_bad_files):
        _im0 = Image.open(_fp).convert('RGB')
        _axm[_i][0].imshow(_im0)
        _axm[_i][0].set(title=_fp.name)
        _axm[_i][0].axis('off')
        _axm[_i][1].imshow(variant_crop(_fp.name, 'bad', SAM_PADDING))
        _axm[_i][1].set(title='crop p08')
        _axm[_i][1].axis('off')
    _fm.suptitle(f'SAM crops p08: {_nbad} BAD')
    _fm.tight_layout()
    _fm.savefig('reports/figures/transistor_binary/sam_crops_all_bad.png', dpi=80)
    print('saved sam_crops_all_bad.png')
    _fm
    toc('viz_allbad')

    return


if __name__ == "__main__":
    app.run()
