import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Industrial inspector — Nsight Systems + NVTX

    Profiling ejecutado del pipeline de `industrial_sam.py` §1 y §8:
    SAM3 + DINOv2 en CUDA, EfficientAD y regresión ONNX en CPU, fusión congelada.

    Incluye benchmark histórico, verificación numérica, diagnóstico de subnormales,
    matriz de hilos, dos capturas CUDA/NVTX válidas y comparación sin profiler en
    procesos aislados. El cierre y las tablas están al final de la libreta.
    Las celdas largas requieren sus botones; no se repiten al reconectar.

    Los originales y los artefactos de producción se conservan. La variante
    FTZ + 4 hilos es un experimento identificado y no cambia industrial_sam.py.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experimentos y evidencia

    1. Referencia inspeccionada mediante MCP y artefactos identificados por SHA-256.
    2. Manifest del test congelado, precisión y preprocesamiento conservados.
    3. Carga y calentamiento separados de las inferencias medidas.
    4. Diecisiete rangos NVTX para etapas, transferencias y frame completo.
    5. Benchmark histórico: 10 imágenes, 30 mediciones y memoria CUDA.
    6. Diagnóstico CPU, matriz FTZ de 1/2/4 hilos y confirmación residente.
    7. Capturas reales de referencia y variante, validadas por CUDA, NVTX y PID.
    8. Comparación aislada sin profiler, scores y decisiones comprobados.
    9. Informe CSV/JSON/Markdown y cierre del kernel.

    La mediana y p95 históricos corresponden a aquella corrida. La comparación final
    es pequeña y descriptiva; no produce un p95 estable ni una evaluación de precisión.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Referencia y alcance

    La implementación se basa en las secciones 1 y 8 de industrial_sam.py,
    inspeccionadas mediante el MCP. Conserva el camino de inferencia individual:
    prompt transistor, selección SAM por mayor logit, máscara redimensionada con
    vecino más próximo y fondo good_mean. La caja se calcula pero no se recorta,
    igual que en §8.

    DINOv2 usa CLS y regresión ONNX; EfficientAD recibe la imagen original de 256 px,
    incluido el round-trip GPU de §8. La fusión usa fusion.npz sin recalibración.
    Los scores DINO/EffAD/fusión y las decisiones se comparan con la referencia.

    La variante FTZ cambia deliberadamente el tratamiento numérico de subnormales
    y la configuración de hilos; sus resultados se identifican por separado.
    La equivalencia de las imágenes ensayadas no demuestra precisión global.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Entorno, artefactos e imágenes

    Registraremos GPU, driver, versiones de PyTorch/CUDA y Nsight Systems,
    dispositivo, precisión, tamaños de entrada y batch. Confirmaremos CUDA
    antes de medir; un fallback a CPU no se mezclará con resultados GPU.

    Localizaremos los artefactos existentes y su configuración. No habrá
    entrenamiento, descarga de modelos ni recalibración como efecto de medir.
    Los originales en `data/raw/` no se modificarán.

    Usaremos una lista reproducible de imágenes reales GOOD y BAD, con su
    identificación y fingerprint del dataset. Repetir imágenes permite medir
    variabilidad temporal, pero no añade muestras independientes para evaluar
    detección. El número de imágenes únicas y las repeticiones se reportarán
    por separado.
    """)
    return


@app.cell
def profile_diagnostics():
    # Diagnóstico visible: no carga modelos, no entrena y no descarga pesos.
    from pathlib import Path
    import os
    import platform
    import shutil
    import subprocess
    import time
    import json
    import hashlib
    import numpy as np
    import pandas as pd
    import torch
    import onnxruntime as ort

    profile_root = Path.cwd()
    profile_models_dir = profile_root / "models" / "transistor_binary" / "prod"
    profile_required = ["good_mean.npy", "fusion.npz", "dino_logreg.onnx", "effad.onnx"]
    profile_artifacts = pd.DataFrame([
        {"artifact": _name, "exists": (profile_models_dir / _name).is_file(),
         "bytes": (profile_models_dir / _name).stat().st_size if (profile_models_dir / _name).is_file() else None}
        for _name in profile_required
    ])
    profile_nsys = shutil.which("nsys")
    profile_nsys_candidates = sorted(Path("C:/Program Files/NVIDIA Corporation").glob("Nsight Systems*/target-windows-x64/nsys.exe"))
    if profile_nsys is None and profile_nsys_candidates:
        profile_nsys = str(profile_nsys_candidates[-1])
    profile_environment = {
        "kernel_pid": os.getpid(), "cwd": str(profile_root), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "onnxruntime": ort.__version__, "ort_providers": ort.get_available_providers(),
        "nsys": profile_nsys,
    }
    print(json.dumps(profile_environment, indent=2, ensure_ascii=False))
    print(profile_artifacts.to_string(index=False))
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range("profile/nvtx_probe"):
            pass
        print("NVTX: disponible en PyTorch")
    else:
        print("GPU no disponible: no ejecutar benchmark GPU.")
    if not profile_nsys:
        print("Nsight Systems no encontrado en PATH ni en la ruta estándar comprobada.")

    return (
        Path,
        hashlib,
        json,
        np,
        ort,
        os,
        pd,
        profile_artifacts,
        profile_environment,
        profile_models_dir,
        profile_nsys,
        profile_required,
        profile_root,
        shutil,
        subprocess,
        time,
        torch,
    )


@app.cell
def profile_controls(mo, os):
    # Los controles evitan repetir la corrida larga al abrir/reconectar la libreta.
    profile_capture_mode = os.environ.get("QC_PROFILE_CAPTURE", "0") == "1"
    profile_allow_full = mo.ui.checkbox(value=False, label="Habilitar la serie completa de 30 mediciones (la corrida anterior tardó 72 min)")
    profile_short_button = mo.ui.run_button(label="Prueba corta CPU: EfficientAD con 1, 2 y 4 hilos")
    mo.vstack([
        mo.md("## Controles de ejecución\n\nLa serie larga requiere habilitación explícita. La prueba corta crea sesiones CPU de diagnóstico; no sustituye los modelos de referencia. La captura de Nsight se prepara en §9 y se ejecuta en un proceso lanzado bajo el profiler."),
        profile_allow_full, profile_short_button,
    ])

    return profile_allow_full, profile_capture_mode, profile_short_button


@app.cell
def profile_config(hashlib, np, pd, profile_models_dir, profile_root):
    # Configuración reproducible de la primera medición. No entrena ni recalibra.
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor, AutoImageProcessor, AutoModel
    from contextlib import contextmanager, nullcontext
    from datetime import datetime, timezone

    PROFILE_CONFIG = {
        "source": "industrial_sam.py §1 + §8",
        "backbone": "facebook/dinov2-base", "sam_checkpoint": "facebook/sam3",
        "prompt": "transistor", "seed": 42, "batch_size": 1,
        "images_per_class": 5, "warmup_frames": 3, "repeats": 3,
        "precision": "float32", "ort_providers": ["CPUExecutionProvider"],
        "scenario": "resident_models_reference_section8",
        "sampling": "balanced subset of frozen test split; not production prevalence",
    }
    # Se usa el test_idx congelado, sin volver a generar splits ni seleccionar por score.
    with np.load(profile_models_dir / "fusion.npz", allow_pickle=False) as _saved:
        profile_fusion = {k: _saved[k].copy() for k in _saved.files}
    profile_data_root = profile_root / "data" / "processed" / "transistor_binary"
    _profile_test_idx = profile_fusion["test_idx"].astype(int)
    _profile_test_records = []
    for _idx in _profile_test_idx:
        _fname = str(profile_fusion["fnames"][_idx])
        _label = "bad" if int(profile_fusion["y"][_idx]) else "good"
        _path = profile_data_root / _label / _fname
        if not _path.is_file():
            raise FileNotFoundError(_path)
        _profile_test_records.append({"filename": _fname, "label": _label, "path": str(_path), "frozen_index": int(_idx)})
    profile_test_manifest = pd.DataFrame(_profile_test_records)
    profile_manifest = pd.concat([
        profile_test_manifest[profile_test_manifest.label == _label].head(PROFILE_CONFIG["images_per_class"])
        for _label in ("good", "bad")
    ], ignore_index=True)
    assert set(profile_manifest.label) == {"good", "bad"}
    profile_manifest_fingerprint = hashlib.sha256(profile_manifest.to_csv(index=False).encode()).hexdigest()
    print("Test congelado:", len(profile_test_manifest), "imágenes; benchmark:", len(profile_manifest))
    print(profile_manifest[["filename", "label"]].to_string(index=False))
    print("Los tiempos incluyen lectura/decodificación y el round-trip GPU de EffAD de §8.")
    print("Se conserva ONNX en CPU; SAM/DINO en CUDA, sin AMP ni cambio de resolución.")

    return (
        AutoImageProcessor,
        AutoModel,
        Image,
        PROFILE_CONFIG,
        Sam3Model,
        Sam3Processor,
        contextmanager,
        datetime,
        nullcontext,
        profile_fusion,
        profile_manifest,
        profile_manifest_fingerprint,
        timezone,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Arranque, calentamiento y modelos residentes

    Separaremos tres escenarios:

    - **Arranque:** carga de artefactos, inicialización de CUDA y preparación
      del pipeline. Distinguiremos cachés existentes de una ejecución fría.
    - **Calentamiento:** inferencias previas con las mismas formas y ruta de
      ejecución que el benchmark; no entran en las estadísticas finales.
    - **Régimen estable:** inferencia con los modelos que el pipeline mantiene
      residentes, sin volver a cargar pesos por cada imagen.

    Verificaremos si SAM se descarga de VRAM en la referencia. Si mantenerlo
    residente cambia la política original, mediremos ambos escenarios con sus
    respectivos costes de memoria. No sumaremos máximos de VRAM de modelos
    que nunca estuvieron residentes al mismo tiempo.

    Ejecutaremos los modelos en modo evaluación y sin gradientes. No habrá
    otras celdas de entrenamiento o inferencia compitiendo por la GPU durante
    la medición.
    """)
    return


@app.cell
def profile_load_models(
    AutoImageProcessor,
    AutoModel,
    PROFILE_CONFIG,
    Sam3Model,
    Sam3Processor,
    mo,
    ort,
    pd,
    profile_artifacts,
    profile_capture_ftz_mode,
    profile_model_registry,
    profile_models_dir,
    profile_tool_probe_mode,
    time,
    torch,
):
    mo.stop(profile_tool_probe_mode, mo.md("Sonda de Nsight: se omite la carga de modelos."))

    # Un solo juego de modelos por kernel; evita duplicaciones al reejecutar.
    if not profile_model_registry.get("_initialized",False):
        # Modelos residentes: medir la carga con caché local, sin descargar ni entrenar.
        # Las credenciales del entorno nunca se imprimen ni se guardan en reportes.
        assert torch.cuda.is_available(), "Este benchmark requiere CUDA."
        assert profile_artifacts["exists"].all(), "Faltan artefactos de la referencia."
        profile_device = torch.device("cuda:0")
        profile_load_records = []
        _profile_load_start = time.perf_counter()
        print("Cargando SAM3 desde caché local…", flush=True)
        _profile_t0 = time.perf_counter()
        profile_sam_processor = Sam3Processor.from_pretrained(PROFILE_CONFIG["sam_checkpoint"], local_files_only=True)
        profile_sam_model = Sam3Model.from_pretrained(PROFILE_CONFIG["sam_checkpoint"], local_files_only=True).eval().requires_grad_(False).to(profile_device)
        torch.cuda.synchronize(profile_device)
        profile_load_records.append({"stage": "sam_load", "seconds": time.perf_counter()-_profile_t0})
        print("SAM listo. Cargando DINOv2…", flush=True)
        _profile_t0 = time.perf_counter()
        profile_dino_processor = AutoImageProcessor.from_pretrained(PROFILE_CONFIG["backbone"], local_files_only=True)
        profile_dino_model = AutoModel.from_pretrained(PROFILE_CONFIG["backbone"], local_files_only=True).eval().requires_grad_(False).to(profile_device)
        torch.cuda.synchronize(profile_device)
        profile_load_records.append({"stage": "dino_load", "seconds": time.perf_counter()-_profile_t0})
        print("DINO listo. Cargando ONNX CPU…", flush=True)
        _profile_t0 = time.perf_counter()
        profile_logreg_session = ort.InferenceSession(str(profile_models_dir / "dino_logreg.onnx"), providers=PROFILE_CONFIG["ort_providers"])
        _profile_effad_options = ort.SessionOptions()
        if profile_capture_ftz_mode:
            _profile_effad_options.intra_op_num_threads = 4
            _profile_effad_options.inter_op_num_threads = 1
            _profile_effad_options.add_session_config_entry("session.set_denormal_as_zero","1")
            _profile_effad_options.add_session_config_entry("session.intra_op.allow_spinning","0")
            _profile_effad_options.add_session_config_entry("session.inter_op.allow_spinning","0")
        profile_effad_session = ort.InferenceSession(str(profile_models_dir / "effad.onnx"), sess_options=_profile_effad_options, providers=PROFILE_CONFIG["ort_providers"])
        profile_load_records.append({"stage": "onnx_load", "seconds": time.perf_counter()-_profile_t0})
        profile_load_seconds = time.perf_counter() - _profile_load_start
        profile_load_table = pd.DataFrame(profile_load_records)
        print(profile_load_table.to_string(index=False))
        print(f"Carga total (cachés locales, proceso ya iniciado): {profile_load_seconds:.3f} s")
        print("ONNX providers:", profile_logreg_session.get_providers(), profile_effad_session.get_providers())
        print("VRAM asignada/reservada GB:", torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)
        profile_model_registry.update({"_initialized":True,"_capture_ftz":profile_capture_ftz_mode,'profile_device':profile_device,'profile_load_records':profile_load_records,'profile_sam_processor':profile_sam_processor,'profile_sam_model':profile_sam_model,'profile_dino_processor':profile_dino_processor,'profile_dino_model':profile_dino_model,'profile_logreg_session':profile_logreg_session,'profile_effad_session':profile_effad_session,'profile_load_seconds':profile_load_seconds,'profile_load_table':profile_load_table})
    else:
        assert profile_model_registry["_capture_ftz"]==profile_capture_ftz_mode, "Cambiar variante en un proceso nuevo, no recargar modelos GPU."
        profile_device = profile_model_registry['profile_device']
        profile_load_records = profile_model_registry['profile_load_records']
        profile_sam_processor = profile_model_registry['profile_sam_processor']
        profile_sam_model = profile_model_registry['profile_sam_model']
        profile_dino_processor = profile_model_registry['profile_dino_processor']
        profile_dino_model = profile_model_registry['profile_dino_model']
        profile_logreg_session = profile_model_registry['profile_logreg_session']
        profile_effad_session = profile_model_registry['profile_effad_session']
        profile_load_seconds = profile_model_registry['profile_load_seconds']
        profile_load_table = profile_model_registry['profile_load_table']
        print("Modelos reutilizados del registro; no se crean copias GPU.")

    return (
        profile_device,
        profile_dino_model,
        profile_dino_processor,
        profile_effad_session,
        profile_load_seconds,
        profile_load_table,
        profile_logreg_session,
        profile_sam_model,
        profile_sam_processor,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Rangos NVTX y límites de cada etapa

    NVTX etiqueta regiones de ejecución; no acelera el modelo ni produce por
    sí mismo un informe de tiempos GPU. Nsight Systems permite relacionar
    esas regiones con actividad CPU, kernels CUDA y transferencias.

    Plan de nombres, sujeto a confirmar la implementación de referencia:

    | Rango | Trabajo incluido |
    |---|---|
    | `inspect/frame` | Inspección completa de una imagen |
    | `inspect/load_decode` | Lectura del archivo y conversión a imagen |
    | `inspect/preprocess` | Transformaciones y preparación de entradas |
    | `inspect/h2d` | Transferencias CPU → GPU explícitas |
    | `inspect/sam` | Segmentación con SAM |
    | `inspect/mask_crop` | Aplicación de máscara y/o recorte |
    | `inspect/dinov2` | Extracción de características DINOv2 |
    | `inspect/logreg` | Clasificación de características |
    | `inspect/efficientad` | Scores de anomalía EfficientAD |
    | `inspect/fusion` | Combinación de scores y decisión final |

    No inventaremos etapas ausentes ni ejecutaremos una conversión dos veces
    para poder etiquetarla. Las copias implícitas se investigarán en la traza.
    Los rangos anidados y el trabajo solapado no deben sumarse como si fueran
    etapas exclusivas. Un rango CPU puede terminar antes que el trabajo CUDA
    que lanzó.
    """)
    return


@app.cell
def profile_inference(
    Image,
    PROFILE_CONFIG,
    contextmanager,
    np,
    nullcontext,
    profile_device,
    profile_dino_model,
    profile_dino_processor,
    profile_effad_session,
    profile_fusion,
    profile_logreg_session,
    profile_models_dir,
    profile_sam_model,
    profile_sam_processor,
    time,
    torch,
):
    # Inferencia §8, con rangos opcionales. collect=True es diagnóstico sincronizado.
    # El benchmark principal usa collect=False y nvtx=False, sin sincronizar cada etapa.
    @contextmanager
    def profile_stage(name, records, collect=False, nvtx=False):
        if collect:
            torch.cuda.synchronize(profile_device)
            _started = time.perf_counter()
        with (torch.cuda.nvtx.range("inspect/" + name) if nvtx else nullcontext()):
            yield
            if collect:
                torch.cuda.synchronize(profile_device)
        if collect:
            records[name] = (time.perf_counter() - _started) * 1000


    def profile_infer(path, collect=False, nvtx=False, effad_session=None, flush_denormal=False):
        _effad_session = profile_effad_session if effad_session is None else effad_session
        _stages = {}
        with (torch.cuda.nvtx.range("inspect/frame") if nvtx else nullcontext()), torch.inference_mode():
            with profile_stage("load_decode", _stages, collect, nvtx):
                with Image.open(path) as _opened:
                    _image = _opened.convert("RGB")
            with profile_stage("sam_preprocess", _stages, collect, nvtx):
                _sam_in = profile_sam_processor(images=_image, text=PROFILE_CONFIG["prompt"], return_tensors="pt")
            with profile_stage("sam_h2d", _stages, collect, nvtx):
                _sam_in = {k: (v.to(profile_device) if torch.is_tensor(v) else v) for k, v in _sam_in.items()}
            with profile_stage("sam", _stages, collect, nvtx):
                _sam_out = profile_sam_model(**_sam_in)
            with profile_stage("sam_postprocess", _stages, collect, nvtx):
                _query = int(_sam_out.pred_logits[0].argmax())
                _sam_score = float(_sam_out.pred_logits[0][_query])
                _small_mask = torch.sigmoid(_sam_out.pred_masks[0][_query]).cpu().numpy() > 0.5
                _mask = np.array(Image.fromarray((_small_mask * 255).astype("uint8")).resize(_image.size, Image.NEAREST)) > 0
            with profile_stage("mask_crop", _stages, collect, nvtx):
                # §8 calcula bbox pero no recorta ni aplica el fallback QC de batch.
                _ys, _xs = np.where(_mask)
                _bbox = (int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())) if len(_xs) else None
                _mask1024 = np.array(Image.fromarray((_mask * 255).astype("uint8")).resize((1024, 1024))) > 0
                _mean = np.load(profile_models_dir / "good_mean.npy", allow_pickle=False)
                _masked = Image.fromarray(np.where(_mask1024[:, :, None], np.asarray(_image.resize((1024, 1024))), _mean).astype("uint8"))
            with profile_stage("dino_preprocess", _stages, collect, nvtx):
                _dino_in = profile_dino_processor(images=_masked, size={"height": 224, "width": 224}, return_tensors="pt")
            with profile_stage("dino_h2d", _stages, collect, nvtx):
                _dino_in = {k: v.to(profile_device) for k, v in _dino_in.items()}
            with profile_stage("dinov2", _stages, collect, nvtx):
                _dino_out = profile_dino_model(**_dino_in).last_hidden_state[:, 0, :]
            with profile_stage("dino_d2h", _stages, collect, nvtx):
                _cls = _dino_out.cpu().numpy()
            with profile_stage("logreg", _stages, collect, nvtx):
                _pd = float(profile_logreg_session.run(None, {"input": _cls.astype("float32")})[1][0][1])
            with profile_stage("effad_preprocess", _stages, collect, nvtx):
                _xb = torch.from_numpy(np.asarray(_image.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0)
            with profile_stage("effad_gpu_roundtrip", _stages, collect, nvtx):
                # Se conserva la copia a GPU y regreso a CPU del ejemplo original.
                _effad_input = _xb.to(profile_device).cpu().numpy()
            with profile_stage("efficientad_cpu", _stages, collect, nvtx):
                if flush_denormal:
                    torch.set_flush_denormal(True)
                try:
                    _pe = float(torch.from_numpy(_effad_session.run(None, {"input": _effad_input})[0]).flatten()[0])
                finally:
                    if flush_denormal:
                        torch.set_flush_denormal(False)
            with profile_stage("fusion", _stages, collect, nvtx):
                _fz = profile_fusion
                _fused = float(_fz["alpha"] * (_pd - _fz["md"]) / _fz["id"] + (1 - _fz["alpha"]) * (_pe - _fz["me"]) / _fz["ie"])
                _bad = bool(_fused >= float(_fz["cut"]))
        return {"dino": _pd, "effad": _pe, "fused": _fused, "bad": _bad,
                "sam_score": _sam_score, "mask_area": float(_mask.mean()), "bbox": _bbox,
                "stages_ms": _stages}


    def profile_reference(path):
        # Ruta independiente, sin etapas ni rangos, transcrita de §1 y §8.
        # Usa exactamente los mismos modelos residentes para comprobar instrumentación.
        with torch.inference_mode():
            with Image.open(path) as _file:
                _im0 = _file.convert("RGB")
            _inp = profile_sam_processor(images=_im0, text=PROFILE_CONFIG["prompt"], return_tensors="pt")
            _inp = {k: (v.to(profile_device) if torch.is_tensor(v) else v) for k, v in _inp.items()}
            _out = profile_sam_model(**_inp)
            _qi = int(_out.pred_logits[0].argmax())
            _score = float(_out.pred_logits[0][_qi])
            _m = torch.sigmoid(_out.pred_masks[0][_qi]).cpu().numpy() > 0.5
            _mask = np.array(Image.fromarray((_m * 255).astype("uint8")).resize(_im0.size, Image.NEAREST)) > 0
            _ys, _xs = np.where(_mask)
            _bb = (int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())) if len(_xs) else None
            _m = np.array(Image.fromarray((_mask * 255).astype("uint8")).resize((1024, 1024))) > 0
            _mean = np.load(profile_models_dir / "good_mean.npy", allow_pickle=False)
            _masked = Image.fromarray(np.where(_m[:, :, None], np.asarray(_im0.resize((1024, 1024))), _mean).astype("uint8"))
            _in = profile_dino_processor(images=_masked, size={"height": 224, "width": 224}, return_tensors="pt")
            _cls = profile_dino_model(**{k: v.to(profile_device) for k, v in _in.items()}).last_hidden_state[:, 0, :].cpu().numpy()
            _pd = float(profile_logreg_session.run(None, {"input": _cls.astype("float32")})[1][0][1])
            _xb = torch.from_numpy(np.asarray(_im0.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(profile_device)
            _pe = float(torch.from_numpy(profile_effad_session.run(None, {"input": _xb.cpu().numpy()})[0]).flatten()[0])
            _fz = profile_fusion
            _f = float(_fz["alpha"] * (_pd - _fz["md"]) / _fz["id"] + (1 - _fz["alpha"]) * (_pe - _fz["me"]) / _fz["ie"])
        return {"dino": _pd, "effad": _pe, "fused": _f, "bad": bool(_f >= float(_fz["cut"])),
                "sam_score": _score, "mask_area": float(_mask.mean()), "bbox": _bb}

    print("Funciones listas: referencia §8 y ruta instrumentada opcional.")

    return profile_infer, profile_reference


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Protocolo de tiempos y memoria

    CUDA ejecuta operaciones de forma asíncrona. Para medir latencia completa
    con reloj CPU, sincronizaremos en los límites de la medición; los eventos
    CUDA servirán para tramos GPU con límites de stream bien definidos.
    No confundiremos tiempo de lanzamiento CPU con tiempo de ejecución GPU.

    Sincronizar después de cada etapa cambia el solapamiento del pipeline.
    Separaremos el diagnóstico por etapas del benchmark de extremo a extremo.
    La latencia por imagen se medirá sin confundirla con el rendimiento de un
    pipeline concurrente o con batching.

    Reportaremos mediana, p95, número de mediciones e imágenes/segundo sobre
    tiempo total medido. Indicaremos si lectura, decodificación, copias y
    posprocesamiento están incluidos. Un p95 con pocas observaciones será
    descriptivo; repetir la misma imagen no representa toda la variabilidad
    de adquisición de una cámara real.

    Para memoria distinguiremos memoria asignada y reservada por PyTorch de
    la memoria total del dispositivo. Reiniciaremos los contadores de pico
    en el límite del escenario que queramos medir; no vaciaremos la caché
    entre frames de una prueba que pretende representar régimen estable.
    """)
    return


@app.cell
def profile_warmup(
    PROFILE_CONFIG,
    mo,
    profile_allow_full,
    profile_capture_mode,
    profile_device,
    profile_infer,
    profile_manifest,
    time,
    torch,
):
    mo.stop(profile_capture_mode or not profile_allow_full.value, mo.md("Serie completa pausada. Actívala desde **Controles de ejecución** solo cuando quieras repetirla."))

    # Primer frame y calentamiento: fuera de las estadísticas de régimen estable.
    profile_warmup_log = []
    for _warmup_idx in range(PROFILE_CONFIG["warmup_frames"]):
        _warmup_row = profile_manifest.iloc[_warmup_idx % len(profile_manifest)]
        torch.cuda.synchronize(profile_device)
        _warmup_t0 = time.perf_counter()
        _warmup_result = profile_infer(_warmup_row.path)
        torch.cuda.synchronize(profile_device)
        _warmup_ms = (time.perf_counter() - _warmup_t0) * 1000
        profile_warmup_log.append({"frame": _warmup_idx + 1, "filename": _warmup_row.filename, "latency_ms": _warmup_ms})
        print(f"Calentamiento {_warmup_idx+1}/{PROFILE_CONFIG['warmup_frames']}: {_warmup_ms:.1f} ms; BAD={_warmup_result['bad']}", flush=True)
    profile_warmup_done = True

    return profile_warmup_done, profile_warmup_log


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Captura con Nsight Systems

    La captura se configura fuera de la libreta. Primero comprobaremos la
    instalación y las capacidades de Nsight Systems disponibles en Windows,
    su compatibilidad con driver/CUDA y los permisos necesarios.

    Marimo ejecuta el kernel en un proceso distinto del servidor: la captura
    deberá incluir el proceso que realmente hace inferencia. Ver rangos del
    servidor sin kernels CUDA no será evidencia de una captura correcta.
    El procedimiento concreto se fijará al verificar esa instalación.

    Capturaremos una ventana corta, después del calentamiento, con actividad
    CUDA y marcas NVTX. Buscaremos GPU ociosa, esperas CPU, transferencias,
    sincronizaciones y recargas de modelos. La primera captura servirá para
    localizar costes; no asumiremos de antemano qué modelo domina el tiempo.

    Las cifras de rendimiento se obtendrán también sin profiler. Guardaremos
    la traza de diagnóstico por separado, porque capturar puede alterar los
    tiempos. No incluiremos figuras, t-SNE, exportaciones ONNX ni escritura de
    reportes dentro de la región cronometrada de inferencia.
    """)
    return


@app.cell
def profile_nsys_check(
    mo,
    os,
    profile_active_nsys,
    profile_capture_mode,
    profile_tool_probe_mode,
    subprocess,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode,mo.md("Consultas de Nsight omitidas dentro del proceso capturado."))
    # Herramientas externas: solo consultas, sin lanzar otra inferencia.
    profile_nsys_info = {}
    if profile_active_nsys:
        for _args in (["--version"], ["profile", "--help"]):
            _result = subprocess.run([profile_active_nsys, *_args], capture_output=True, text=True, errors="replace", timeout=15)
            _key = "version" if _args == ["--version"] else "profile_help"
            profile_nsys_info[_key] = {"exit_code": _result.returncode, "text": _result.stdout + _result.stderr}
        print(profile_nsys_info["version"]["text"])
        print("Opciones de attach en CLI:")
        print("\n".join(_line for _line in profile_nsys_info["profile_help"]["text"].splitlines() if any(_word in _line.lower() for _word in ("attach", "admin", "nvtx", "capture", "trace"))))
    print("PID del kernel:", os.getpid())

    return


@app.cell
def profile_stage_benchmark(
    PROFILE_CONFIG,
    np,
    pd,
    profile_infer,
    profile_manifest,
    profile_timings,
    torch,
):
    # Diagnóstico por etapas: medición distinta, con sincronización y rangos NVTX.
    # No mezclar estos tiempos con la distribución de latencia principal.
    assert len(profile_timings) == len(profile_manifest) * PROFILE_CONFIG["repeats"]
    profile_stage_rows = []
    profile_stage_checks = []
    print("Diagnóstico NVTX por etapa:", len(profile_manifest), "imágenes", flush=True)
    with torch.cuda.nvtx.range("inspect/capture"):
        for _stage_idx, _row in enumerate(profile_manifest.itertuples()):
            _out = profile_infer(_row.path, collect=True, nvtx=True)
            for _stage_name, _milliseconds in _out["stages_ms"].items():
                profile_stage_rows.append({"filename": _row.filename, "label": _row.label, "stage": _stage_name, "ms": _milliseconds})
            _baseline = profile_timings[profile_timings.filename == _row.filename].iloc[0]
            _scores_match = all(np.isclose(float(_baseline[_key]), _out[_key], rtol=1e-5, atol=1e-6) for _key in ("dino", "effad", "fused", "sam_score", "mask_area"))
            profile_stage_checks.append({"filename": _row.filename, "same_decision": bool(_baseline.bad) == _out["bad"], "same_scores": bool(_scores_match)})
            print(f"Etapas {_stage_idx+1}/{len(profile_manifest)}: {_row.filename}", flush=True)
    profile_stage_timings = pd.DataFrame(profile_stage_rows)
    profile_stage_summary = profile_stage_timings.groupby("stage", as_index=False).agg(median_ms=("ms", "median"), mean_ms=("ms", "mean"), max_ms=("ms", "max"), n=("ms", "size")).sort_values("median_ms", ascending=False)
    profile_stage_equivalence = pd.DataFrame(profile_stage_checks)
    assert profile_stage_equivalence.same_decision.all() and profile_stage_equivalence.same_scores.all(), "Divergencia entre benchmark y diagnóstico NVTX."
    print(profile_stage_summary.to_string(index=False))

    return (
        profile_stage_equivalence,
        profile_stage_summary,
        profile_stage_timings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Resultados, trazabilidad y siguientes experimentos

    Guardaremos configuración, identidad de artefactos, manifest de imágenes,
    tiempos individuales y resumen bajo rutas específicas del proyecto.
    Los reportes de profiling tendrán su propia carpeta para no sobrescribir
    las métricas de entrenamiento ni las trazas anteriores.

    Cada comparación usará las mismas imágenes y condiciones. Los errores,
    frames omitidos o fallos de máscara se contabilizarán y explicarán;
    no se eliminarán silenciosamente para mejorar los tiempos.

    El logging y la escritura de artefactos quedarán fuera de los intervalos
    medidos. Las consultas a MLflow seguirán pasando por su MCP y las
    escrituras por el SDK desde celdas, conforme a `AGENTS.md`.

    Primero estableceremos la referencia. Después, el perfil decidirá si
    explorar TensorRT, preprocesamiento GPU, batching o cambios de residencia.
    Esta libreta no activa ninguna optimización todavía y no demuestra por
    sí sola capacidad de inspección en una línea industrial.
    """)
    return


@app.cell
def profile_results(
    PROFILE_CONFIG,
    hashlib,
    json,
    mo,
    pd,
    profile_benchmark_errors,
    profile_environment,
    profile_equivalence_rows,
    profile_equivalence_table,
    profile_load_seconds,
    profile_load_table,
    profile_manifest,
    profile_manifest_fingerprint,
    profile_memory_before,
    profile_memory_peak,
    profile_models_dir,
    profile_required,
    profile_root,
    profile_run_id,
    profile_stage_equivalence,
    profile_stage_summary,
    profile_stage_timings,
    profile_summary,
    profile_timings,
    profile_warmup_log,
):
    # Guardar fuera del intervalo medido, sin sobrescribir otras corridas.
    assert profile_stage_equivalence.same_decision.all()
    profile_report_dir = profile_root / "reports" / "profiling" / "transistor_binary" / profile_run_id
    profile_report_dir.mkdir(parents=True, exist_ok=False)
    profile_timings.to_csv(profile_report_dir / "latencies.csv", index=False)
    profile_stage_timings.to_csv(profile_report_dir / "stages.csv", index=False)
    profile_stage_summary.to_csv(profile_report_dir / "stage_summary.csv", index=False)
    profile_manifest.to_csv(profile_report_dir / "manifest.csv", index=False)
    profile_load_table.to_csv(profile_report_dir / "load_times.csv", index=False)
    profile_equivalence_table.to_csv(profile_report_dir / "reference_equivalence.csv", index=False)
    profile_stage_equivalence.to_csv(profile_report_dir / "nvtx_equivalence.csv", index=False)
    pd.DataFrame(profile_warmup_log).to_csv(profile_report_dir / "warmup.csv", index=False)
    profile_artifact_hashes = {
        _filename: hashlib.sha256((profile_models_dir / _filename).read_bytes()).hexdigest()
        for _filename in profile_required
    }
    profile_report = {
        "run_id": profile_run_id, "config": PROFILE_CONFIG, "environment": profile_environment,
        "load_seconds": profile_load_seconds, "summary": profile_summary,
        "memory_before": profile_memory_before, "memory_peak": profile_memory_peak,
        "manifest_sha256": profile_manifest_fingerprint, "artifacts_sha256": profile_artifact_hashes,
        "warmup": profile_warmup_log, "equivalence": profile_equivalence_rows,
        "errors": profile_benchmark_errors,
        "nsight_trace_collected": False,
        "nsight_note": "NVTX emitted in current kernel; no external Nsight collector attached. CLI version has no attach option. A .nsys-rep is not produced by timers or NVTX alone.",
        "limitations": ["10 balanced test images, not an accuracy study", "30 repeated timings, descriptive p95", "warm OS/model caches; startup excludes kernel process creation", "stage timings synchronize CUDA and are separate from primary latency", "CPU ONNX and EffAD GPU roundtrip retained from reference section 8", "SAM mask QC differs between reference batch path and section 8; this benchmark follows section 8"],
    }
    (profile_report_dir / "summary.json").write_text(json.dumps(profile_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Reportes guardados:", profile_report_dir)
    mo.vstack([
        mo.md(f"## Resultado — {profile_run_id}\n\nReferencia: **industrial_sam.py §8**; batch 1; SAM/DINO CUDA, ONNX CPU.\n\n**{profile_summary['median_ms']:.1f} ms mediana · {profile_summary['p95_ms']:.1f} ms p95 · {profile_summary['sequential_service_fps']:.3f} imágenes/s**\n\n{profile_summary['measurements']} mediciones de {profile_summary['unique_images']} imágenes únicas, después del calentamiento. VRAM pico asignada: {profile_summary['peak_allocated_gb']:.2f} GB.\n\nLas marcas NVTX se emitieron, pero **no se ha capturado una traza de Nsight**. Los tiempos por etapa siguientes son un diagnóstico sincronizado separado."),
        mo.ui.table(profile_stage_summary, selection=None),
        mo.md(f"Archivos: `{profile_report_dir.relative_to(profile_root)}`"),
    ])

    return


@app.cell
def profile_equivalence(
    np,
    pd,
    profile_infer,
    profile_manifest,
    profile_reference,
    profile_warmup_done,
):
    # Equivalencia: una GOOD y una BAD frente a la transcripción independiente de §8.
    assert profile_warmup_done
    profile_equivalence_rows = []
    for _validation_label in ("good", "bad"):
        _validation_row = profile_manifest[profile_manifest.label == _validation_label].iloc[0]
        _reference = profile_reference(_validation_row.path)
        _instrumented = profile_infer(_validation_row.path, collect=True, nvtx=True)
        _numeric_keys = ["dino", "effad", "fused", "sam_score", "mask_area"]
        _allclose = all(np.isclose(_reference[_key], _instrumented[_key], rtol=1e-5, atol=1e-6) for _key in _numeric_keys)
        _same = _allclose and _reference["bad"] == _instrumented["bad"] and _reference["bbox"] == _instrumented["bbox"]
        profile_equivalence_rows.append({"filename": _validation_row.filename, "label": _validation_label,
            "equivalent": bool(_same), "bad": _reference["bad"],
            "max_score_delta": max(abs(_reference[_key] - _instrumented[_key]) for _key in _numeric_keys)})
        print("Equivalencia:", profile_equivalence_rows[-1], flush=True)
    profile_equivalence_table = pd.DataFrame(profile_equivalence_rows)
    profile_equivalence_ok = bool(profile_equivalence_table.equivalent.all())
    assert profile_equivalence_ok, "La instrumentación cambia resultados: no continuar."

    return (
        profile_equivalence_ok,
        profile_equivalence_rows,
        profile_equivalence_table,
    )


@app.cell
def profile_benchmark(
    PROFILE_CONFIG,
    datetime,
    json,
    mo,
    np,
    pd,
    profile_allow_full,
    profile_capture_mode,
    profile_device,
    profile_equivalence_ok,
    profile_infer,
    profile_manifest,
    time,
    timezone,
    torch,
):
    mo.stop(profile_capture_mode or not profile_allow_full.value, mo.md("Serie completa pausada. Actívala desde **Controles de ejecución** solo cuando quieras repetirla."))

    # Benchmark principal: sin rangos NVTX, sin profiler, sin sincronización por etapa.
    assert profile_equivalence_ok
    profile_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    profile_benchmark_rows = []
    profile_benchmark_errors = []
    profile_memory_before = {"allocated_bytes": torch.cuda.memory_allocated(profile_device), "reserved_bytes": torch.cuda.memory_reserved(profile_device)}
    torch.cuda.reset_peak_memory_stats(profile_device)
    _profile_wall_start = time.perf_counter()
    _profile_total_frames = len(profile_manifest) * PROFILE_CONFIG["repeats"]
    print(f"Benchmark: {_profile_total_frames} mediciones, batch=1, modelos residentes.", flush=True)
    with mo.status.progress_bar(total=_profile_total_frames, title="Inferencia sin profiler", show_eta=True) as _bar:
        for _repeat in range(PROFILE_CONFIG["repeats"]):
            # Orden determinista mezclado en cada repetición: evitar agrupar todas las BAD al final.
            _order = np.random.default_rng(PROFILE_CONFIG["seed"] + _repeat).permutation(len(profile_manifest))
            for _manifest_idx in _order:
                _row = profile_manifest.iloc[int(_manifest_idx)]
                try:
                    torch.cuda.synchronize(profile_device)
                    _t0 = time.perf_counter()
                    _prediction = profile_infer(_row.path, collect=False, nvtx=False)
                    torch.cuda.synchronize(profile_device)
                    _elapsed = (time.perf_counter() - _t0) * 1000
                except Exception as _exc:
                    profile_benchmark_errors.append({"filename": _row.filename, "repeat": _repeat+1, "error": str(_exc)})
                    print("FALLO:", profile_benchmark_errors[-1], flush=True)
                    raise
                profile_benchmark_rows.append({"filename": _row.filename, "label": _row.label, "repeat": _repeat+1,
                    "latency_ms": _elapsed, **{k: v for k, v in _prediction.items() if k not in ("stages_ms", "bbox")}})
                _bar.update()
                if len(profile_benchmark_rows) % 5 == 0:
                    print(f"{len(profile_benchmark_rows)}/{_profile_total_frames}: {_row.filename}, {_elapsed:.1f} ms", flush=True)
    profile_benchmark_wall_seconds = time.perf_counter() - _profile_wall_start
    profile_memory_peak = {"allocated_bytes": torch.cuda.max_memory_allocated(profile_device), "reserved_bytes": torch.cuda.max_memory_reserved(profile_device)}
    profile_timings = pd.DataFrame(profile_benchmark_rows)
    profile_summary = {
        "unique_images": int(profile_timings.filename.nunique()), "measurements": len(profile_timings),
        "median_ms": float(profile_timings.latency_ms.median()),
        "p95_ms": float(profile_timings.latency_ms.quantile(0.95)),
        "mean_ms": float(profile_timings.latency_ms.mean()),
        "min_ms": float(profile_timings.latency_ms.min()), "max_ms": float(profile_timings.latency_ms.max()),
        "sequential_service_fps": float(1000 / profile_timings.latency_ms.mean()),
        "loop_wall_seconds": profile_benchmark_wall_seconds,
        "loop_fps_including_progress": len(profile_timings) / profile_benchmark_wall_seconds,
        "peak_allocated_gb": profile_memory_peak["allocated_bytes"] / 1e9,
        "peak_reserved_gb": profile_memory_peak["reserved_bytes"] / 1e9,
        "errors": len(profile_benchmark_errors),
    }
    print(json.dumps(profile_summary, indent=2))

    return (
        profile_benchmark_errors,
        profile_memory_before,
        profile_memory_peak,
        profile_run_id,
        profile_summary,
        profile_timings,
    )


@app.cell
def profile_pending_cpu_notes(mo):
    mo.md(r"""
    ## 8. Diagnóstico de EfficientAD completado

    La corrida histórica tuvo mediana de 142.95 s; no representa la capacidad nominal.
    Se detectaron 749,314 pesos subnormales en 19 tensores de EfficientAD. Se probó una
    variante explícita con FTZ/DAZ, ejecución secuencial, 4 hilos CPU y spinning desactivado.
    Los pesos, la fusión y los umbrales permanecen intactos.

    La matriz FTZ de 1/2/4 hilos terminó y eligió 4 hilos entre esas alternativas.
    Esa matriz compartió recursos con una prueba lenta: sirve para seleccionar una
    configuración candidata, no para prometer rendimiento. El informe final distingue
    esa exploración, la confirmación residente, las capturas y la comparación aislada.

    FTZ y el número de hilos cambian juntos en la comparación final. La mejora no
    puede atribuirse exclusivamente a uno de ellos. Los scores y decisiones se
    verifican con tolerancia rtol=1e-4, atol=1e-6 en las imágenes ensayadas.
    """)
    return


@app.cell
def profile_runtime_context(json, os, profile_effad_session, torch):
    # Contexto ligero, sin ejecutar inferencias ni cambiar afinidad/energía.
    import ctypes
    profile_runtime_context = {
        "cpu_logical_count": os.cpu_count(),
        "torch_intra_threads": torch.get_num_threads(),
        "torch_inter_threads": torch.get_num_interop_threads(),
        "ort_default_intra_threads": profile_effad_session.get_session_options().intra_op_num_threads,
        "ort_default_inter_threads": profile_effad_session.get_session_options().inter_op_num_threads,
        "ort_execution_mode": str(profile_effad_session.get_session_options().execution_mode),
        "ort_graph_optimization": str(profile_effad_session.get_session_options().graph_optimization_level),
        "onnx_providers": profile_effad_session.get_providers(),
        "kernel_pid": os.getpid(),
        "administrator": bool(ctypes.windll.shell32.IsUserAnAdmin()) if os.name == "nt" else None,
    }
    # Solo variables de rendimiento conocidas: no volcar el entorno ni credenciales.
    profile_runtime_context["thread_environment"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    print(json.dumps(profile_runtime_context, indent=2))

    return ctypes, profile_runtime_context


@app.cell
def profile_saved_results(json, mo, profile_root):
    # Mostrar la última corrida guardada, aunque las celdas largas estén pausadas.
    _profile_saved_candidates = sorted((profile_root / "reports" / "profiling" / "transistor_binary").glob("*/summary.json"))
    profile_saved_report_dir = _profile_saved_candidates[-1].parent if _profile_saved_candidates else None
    profile_saved_report = json.loads((profile_saved_report_dir / "summary.json").read_text(encoding="utf-8")) if profile_saved_report_dir else None
    if profile_saved_report:
        _s = profile_saved_report["summary"]
        mo.output.replace(mo.md(f"### Última corrida guardada\n\n`{profile_saved_report['run_id']}`: **{_s['median_ms']/1000:.2f} s mediana · {_s['p95_ms']/1000:.2f} s p95**, {_s['measurements']} mediciones.\n\nCorrida histórica anómala; el diagnóstico y la comparación aislada están al final de esta libreta. No usar como capacidad nominal del inspector."))
    else:
        print("Todavía no hay una corrida completa guardada.")

    return profile_saved_report, profile_saved_report_dir


@app.cell
def profile_short_cpu(
    Image,
    datetime,
    json,
    mo,
    np,
    ort,
    pd,
    profile_capture_mode,
    profile_manifest,
    profile_models_dir,
    profile_root,
    profile_runtime_context,
    profile_saved_report_dir,
    profile_short_button,
    time,
    timezone,
    torch,
):
    # Se ejecuta solo con el botón; deja intacta la sesión ONNX de referencia.
    mo.stop(profile_capture_mode or not profile_short_button.value, mo.md("Prueba CPU corta lista. Pulsa su botón en **Controles de ejecución** para ejecutarla."))
    profile_short_rows = []
    profile_short_context = dict(profile_runtime_context)
    _profile_short_t0 = time.perf_counter()
    _profile_short_stop = False
    _profile_short_expected = pd.read_csv(profile_saved_report_dir / "latencies.csv") if profile_saved_report_dir else None
    for _threads in (1, 2, 4):
        _options = ort.SessionOptions()
        _options.intra_op_num_threads = _threads
        _options.inter_op_num_threads = 1
        _options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(str(profile_models_dir / "effad.onnx"), sess_options=_options, providers=["CPUExecutionProvider"])
        for _label in ("good", "bad"):
            _row = profile_manifest[profile_manifest.label == _label].iloc[0]
            with Image.open(_row.path) as _opened:
                _image = _opened.convert("RGB")
            # Misma entrada de EffAD que §8; solo se cronometra Session.run.
            _input = torch.from_numpy(np.asarray(_image.resize((256, 256)), dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).numpy()
            _expected = None
            if _profile_short_expected is not None:
                _prior = _profile_short_expected[_profile_short_expected.filename == _row.filename]
                if len(_prior):
                    _expected = float(_prior.iloc[0].effad)
            for _trial in range(4):
                _start = time.perf_counter()
                _score = float(np.asarray(_session.run(None, {"input": _input})[0]).reshape(-1)[0])
                _ms = (time.perf_counter() - _start) * 1000
                _matches = bool(np.isclose(_score, _expected, rtol=1e-4, atol=1e-6)) if _expected is not None else None
                profile_short_rows.append({"filename": _row.filename, "label": _label, "threads": _threads,
                    "trial": _trial, "warmup": _trial == 0, "ms": _ms, "effad": _score,
                    "reference_effad": _expected, "score_match": _matches})
                print(f"CPU threads={_threads}, {_row.filename}, trial={_trial}: {_ms:.1f} ms; score_match={_matches}", flush=True)
                if _matches is False:
                    _profile_short_stop = True
                    break
                # Presupuesto entre llamadas; no interrumpe una llamada nativa ya iniciada.
                if time.perf_counter() - _profile_short_t0 > 120:
                    _profile_short_stop = True
                    break
            if _profile_short_stop:
                break
        del _session
        if _profile_short_stop:
            break
    profile_short_timings = pd.DataFrame(profile_short_rows)
    profile_short_summary = profile_short_timings[~profile_short_timings.warmup].groupby("threads", as_index=False).agg(median_ms=("ms", "median"), min_ms=("ms", "min"), max_ms=("ms", "max"), n=("ms", "size"))
    profile_short_report_dir = profile_root / "reports" / "profiling" / "transistor_binary" / ("cpu_threads_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    profile_short_report_dir.mkdir(parents=True, exist_ok=False)
    profile_short_timings.to_csv(profile_short_report_dir / "timings.csv", index=False)
    profile_short_summary.to_csv(profile_short_report_dir / "thread_summary.csv", index=False)
    (profile_short_report_dir / "context.json").write_text(json.dumps({"context": profile_short_context, "completed": not _profile_short_stop, "expected_calls": 24,
        "observed_calls": len(profile_short_rows), "score_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "note": "EfficientAD CPU only; excludes decode, transfers and other models. Explicit thread variants; not reference end-to-end latency."}, indent=2), encoding="utf-8")
    print("Prueba completa:", not _profile_short_stop, "guardado en", profile_short_report_dir)
    mo.ui.table(profile_short_summary, selection=None)

    return


@app.cell
def profile_capture_notes(mo):
    mo.md(r"""
    ## 9. Capturas reales completadas

    Se ejecutaron dos procesos nuevos de esta misma libreta: referencia y variante
    FTZ + 4 hilos. Nsight Systems 2025.1.3.140 portátil procede del distribuidor oficial
    de NVIDIA y su SHA-256 coincide con el manifiesto. La versión 2023.4.4 dejó
    intentos fallidos conservados; las trazas aceptadas usan 2025.1.

    Cada captura contiene dos frames, los 17 rangos NVTX esperados y 5,776 kernels CUDA.
    Se validaron el PID del proceso, las predicciones y el contenido de SQLite.
    Los modelos del kernel interactivo se descargan de CUDA antes de lanzar el hijo;
    el registro de modelos evita volver a cargarlos al reejecutar celdas.

    Los botones permiten repetir experimentos deliberadamente. La serie larga queda
    pausada al abrir la libreta. Los archivos pesados .nsys-rep y SQLite se conservan
    localmente y están excluidos de Git; los CSV/JSON/Markdown quedan disponibles.
    """)
    return


@app.cell
def profile_capture_setup(
    Path,
    datetime,
    mo,
    os,
    profile_active_nsys,
    profile_root,
    shutil,
    timezone,
):
    # Preparar el comando: no lanza Nsight ni otro kernel.
    profile_trace_reports_root = profile_root / "reports" / "profiling" / "transistor_binary"
    profile_capture_output_dir = Path(os.environ["QC_PROFILE_TRACE_DIR"]) if os.environ.get("QC_PROFILE_TRACE_DIR") else profile_trace_reports_root / ("nsight_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    profile_trace_base = profile_capture_output_dir / "capture"
    profile_uv_executable = shutil.which("uv")
    profile_capture_argv = ([profile_uv_executable, "run", profile_active_nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false",
        "--wait=primary", "--force-overwrite=false", "--output=" + str(profile_trace_base),
        str(profile_root / ".venv" / "Scripts" / "python.exe"), "notebooks/industrial_profile.py"]
        if profile_active_nsys and profile_uv_executable else [])

    def profile_powershell_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    profile_capture_command = ""
    if profile_capture_argv:
        profile_capture_command = "\n".join([
            "Set-Location -LiteralPath " + profile_powershell_quote(profile_root),
            "$qcPreviousCapture = $env:QC_PROFILE_CAPTURE",
            "$qcPreviousTraceDir = $env:QC_PROFILE_TRACE_DIR",
            "try {",
            "  New-Item -ItemType Directory -Force -Path " + profile_powershell_quote(profile_capture_output_dir) + " | Out-Null",
            "  $env:QC_PROFILE_CAPTURE = '1'",
            "  $env:QC_PROFILE_TRACE_DIR = " + profile_powershell_quote(profile_capture_output_dir),
            "  & " + " ".join(profile_powershell_quote(_arg) for _arg in profile_capture_argv),
            "} finally {",
            "  if ($null -eq $qcPreviousCapture) { Remove-Item Env:QC_PROFILE_CAPTURE -ErrorAction SilentlyContinue } else { $env:QC_PROFILE_CAPTURE = $qcPreviousCapture }",
            "  if ($null -eq $qcPreviousTraceDir) { Remove-Item Env:QC_PROFILE_TRACE_DIR -ErrorAction SilentlyContinue } else { $env:QC_PROFILE_TRACE_DIR = $qcPreviousTraceDir }",
            "}",
        ])
        mo.output.replace(mo.md("### Comando de captura PowerShell\n\nNo ejecutarlo mientras otro kernel mantenga los mismos modelos en GPU.\n\n```powershell\n" + profile_capture_command + "\n```"))
    else:
        print("No se puede preparar la captura: faltan nsys o uv en las rutas detectadas.")

    return (
        profile_capture_output_dir,
        profile_trace_reports_root,
        profile_uv_executable,
    )


@app.cell
def profile_capture_workload(
    PROFILE_CONFIG,
    json,
    mo,
    pd,
    profile_active_nsys,
    profile_active_nsys_version,
    profile_capture_ftz_mode,
    profile_capture_mode,
    profile_capture_output_dir,
    profile_device,
    profile_environment,
    profile_infer,
    profile_isolated_benchmark_mode,
    profile_manifest,
    profile_runtime_context,
    profile_windows_cpu,
    torch,
):
    mo.stop(profile_isolated_benchmark_mode)
    # Modo de captura de esta libreta: se activa solo al lanzarla bajo Nsight.
    mo.stop(not profile_capture_mode, mo.md("Carga de captura preparada; no ejecutada. Usar el comando PowerShell de §9."))
    profile_capture_output_dir.mkdir(parents=True, exist_ok=True)
    _profile_capture_frames = pd.concat([profile_manifest[profile_manifest.label == _label].head(1) for _label in ("good", "bad")], ignore_index=True)
    print("Modo Nsight: calentamiento fuera de captura, luego una GOOD y una BAD.", flush=True)
    profile_infer(_profile_capture_frames.iloc[0].path, collect=False, nvtx=False,flush_denormal=profile_capture_ftz_mode)
    torch.cuda.synchronize(profile_device)
    profile_capture_predictions = []
    try:
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range("inspect/capture"):
            for _row in _profile_capture_frames.itertuples():
                _prediction = profile_infer(_row.path, collect=False, nvtx=True,flush_denormal=profile_capture_ftz_mode)
                profile_capture_predictions.append({"filename": _row.filename, "label": _row.label,
                    **{k: v for k, v in _prediction.items() if k not in ("bbox", "stages_ms")}})
            torch.cuda.synchronize(profile_device)
    finally:
        torch.cuda.profiler.stop()
    # La existencia/validez del .nsys-rep se verifica DESPUÉS de salir del proceso.
    pd.DataFrame(profile_capture_predictions).to_csv(profile_capture_output_dir / "capture_predictions.csv", index=False)
    (profile_capture_output_dir / "capture_context.json").write_text(json.dumps({
        "source": PROFILE_CONFIG["source"], "environment": profile_environment,
        "runtime": profile_runtime_context, "windows_cpu": profile_windows_cpu, "capture_requested": True,
        "trace_validated": False, "nsys_executable": profile_active_nsys, "nsys_version": profile_active_nsys_version, "flush_denormal": profile_capture_ftz_mode, "effad_threads": 4 if profile_capture_ftz_mode else 0, "synchronized_stage_timing": False,
        "frames": len(profile_capture_predictions), "configuration": PROFILE_CONFIG,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Ventana CUDA/NVTX ejecutada. Comprobar la traza tras finalizar Nsight:", profile_capture_output_dir)

    return


@app.cell
def profile_trace_review_notes(mo):
    mo.md(r"""
    ## 10. Interpretación de las trazas

    Los rangos NVTX son duraciones de host y pueden incluir espera de CUDA. Las etapas
    anidadas no se suman; sus medianas no equivalen a tiempo puro de GPU.

    En las capturas, EfficientAD CPU pasó de 3.77 s a 0.78 s de mediana y el frame
    de 5.34 s a 2.30 s. Durante los rangos EfficientAD no hubo solapamiento con kernels
    CUDA del pipeline. Esto identifica una sección CPU que deja la GPU del pipeline
    sin trabajo; no describe la actividad global de otras aplicaciones.

    La referencia bajo Nsight es mucho más rápida que sin profiler: el entorno
    de captura perturba los tiempos. No usar esos frames como latencia nominal
    ni atribuir la diferencia exclusivamente a FTZ.

    La comparación sin profiler se ejecuta en procesos nuevos, con dos calentamientos
    y cuatro mediciones por escenario. Son dos imágenes únicas y resultados
    descriptivos: no constituyen p95 estable, evaluación de precisión completa ni
    medición de capacidad sostenida. El informe final conserva los intentos fallidos
    y excluye la confirmación contaminada por copias duplicadas de modelos en GPU.
    """)
    return


@app.cell
def profile_trace_review_control(mo):
    profile_trace_review_button = mo.ui.run_button(label="Analizar la última traza Nsight guardada")
    profile_trace_review_button

    return (profile_trace_review_button,)


@app.cell
def profile_trace_review(
    mo,
    np,
    pd,
    profile_active_nsys,
    profile_capture_mode,
    profile_saved_report_dir,
    profile_trace_reports_root,
    profile_trace_review_button,
    subprocess,
):
    # Análisis externo de una traza existente. No lanza inferencias ni modifica modelos.
    mo.stop(profile_capture_mode or not profile_trace_review_button.value, mo.md("Análisis de traza listo; requiere primero una captura .nsys-rep."))
    profile_existing_traces = sorted(profile_trace_reports_root.glob("nsight_*/*.nsys-rep"))
    mo.stop(not profile_existing_traces, mo.md("Todavía no hay una traza .nsys-rep en la carpeta de profiling."))
    profile_review_trace = profile_existing_traces[-1]
    assert profile_review_trace.stat().st_size > 0, "Traza vacía."
    assert profile_active_nsys, "No se encontró nsys."
    profile_trace_stats = {}
    for _report_name in ("nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum"):
        _stats = subprocess.run([profile_active_nsys, "stats", "--report=" + _report_name,
            "--format=csv", str(profile_review_trace)], capture_output=True,
            text=True, errors="replace", timeout=60)
        profile_trace_stats[_report_name] = {"exit_code": _stats.returncode, "stdout": _stats.stdout, "stderr": _stats.stderr}
        (profile_review_trace.parent / (_report_name + ".txt")).write_text(_stats.stdout + "\n" + _stats.stderr, encoding="utf-8")
        print("Reporte", _report_name, "exit_code", _stats.returncode)
        print((_stats.stdout + _stats.stderr)[:4000])
    _profile_predictions_path = profile_review_trace.parent / "capture_predictions.csv"
    if _profile_predictions_path.is_file() and profile_saved_report_dir:
        _captured = pd.read_csv(_profile_predictions_path)
        _baseline = pd.read_csv(profile_saved_report_dir / "latencies.csv").drop_duplicates("filename")
        _comparison = _captured.merge(_baseline, on="filename", suffixes=("_capture", "_reference"), validate="one_to_one")
        assert len(_comparison) == len(_captured), "Hay imágenes capturadas sin referencia."
        for _key in ("dino", "effad", "fused"):
            _comparison[_key + "_match"] = np.isclose(_comparison[_key + "_capture"], _comparison[_key + "_reference"], rtol=1e-4, atol=1e-6)
        _comparison["decision_match"] = _comparison.bad_capture == _comparison.bad_reference
        _comparison.to_csv(profile_review_trace.parent / "capture_equivalence.csv", index=False)
        print(_comparison[["filename", "dino_match", "effad_match", "fused_match", "decision_match"]].to_string(index=False))
    print("Revisar kernels y rangos reales en la salida: exit_code=0 por sí solo no valida la captura CUDA.")

    return


@app.cell
def profile_short_full_control(mo):
    profile_short_full_button = mo.ui.run_button(label="Confirmar pipeline completo: 2 imágenes × 2 repeticiones")
    profile_short_full_threads = mo.ui.dropdown(options={"1 hilo CPU (variante explícita)": 1, "2 hilos CPU": 2, "4 hilos CPU": 4}, value="1 hilo CPU (variante explícita)", label="Hilos internos ONNX para la confirmación")
    mo.vstack([mo.md("## 11. Confirmación corta sin profiler\n\nDespués del diagnóstico, elegir una configuración y comprobar el pipeline completo con una GOOD y una BAD. Esto prueba una variante explícita de runtime; no reemplaza ni modifica la sesión ONNX original."), profile_short_full_threads, profile_short_full_button])

    return profile_short_full_button, profile_short_full_threads


@app.cell
def profile_short_full_benchmark(
    PROFILE_CONFIG,
    datetime,
    json,
    mo,
    np,
    ort,
    pd,
    profile_capture_mode,
    profile_device,
    profile_infer,
    profile_manifest,
    profile_models_dir,
    profile_root,
    profile_saved_report_dir,
    profile_short_full_button,
    profile_short_full_threads,
    time,
    timezone,
    torch,
):
    mo.stop(profile_capture_mode or not profile_short_full_button.value, mo.md("Confirmación end-to-end lista; requiere pulsar su botón."))
    _profile_full_options = ort.SessionOptions()
    _profile_full_options.intra_op_num_threads = int(profile_short_full_threads.value)
    _profile_full_options.inter_op_num_threads = 1
    _profile_full_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _profile_full_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _profile_full_session = ort.InferenceSession(str(profile_models_dir / "effad.onnx"), sess_options=_profile_full_options, providers=["CPUExecutionProvider"])
    _profile_full_inputs = pd.concat([profile_manifest[profile_manifest.label == _label].head(1) for _label in ("good", "bad")], ignore_index=True)
    _profile_full_expected = pd.read_csv(profile_saved_report_dir / "latencies.csv").drop_duplicates("filename") if profile_saved_report_dir else None
    profile_short_full_rows = []
    _profile_full_started = time.perf_counter()
    _profile_full_abort = False
    for _repeat in range(3):
        for _row in _profile_full_inputs.itertuples():
            torch.cuda.synchronize(profile_device)
            _start = time.perf_counter()
            _prediction = profile_infer(_row.path, collect=False, nvtx=False, effad_session=_profile_full_session)
            torch.cuda.synchronize(profile_device)
            _ms = (time.perf_counter() - _start) * 1000
            _equivalent = None
            if _profile_full_expected is not None:
                _matches = _profile_full_expected[_profile_full_expected.filename == _row.filename]
                if len(_matches):
                    _prior = _matches.iloc[0]
                    _equivalent = bool(_prior.bad) == _prediction["bad"] and all(np.isclose(float(_prior[_key]), _prediction[_key], rtol=1e-4, atol=1e-6) for _key in ("dino", "effad", "fused"))
            profile_short_full_rows.append({"filename": _row.filename, "label": _row.label, "repeat": _repeat,
                "warmup": _repeat == 0, "latency_ms": _ms, "equivalent": _equivalent,
                "dino": _prediction["dino"], "effad": _prediction["effad"], "fused": _prediction["fused"], "bad": _prediction["bad"]})
            print(f"Confirmación {_repeat}/2, {_row.filename}: {_ms:.1f} ms, equivalente={_equivalent}", flush=True)
            if _equivalent is False or time.perf_counter() - _profile_full_started > 120:
                _profile_full_abort = True
                break
        if _profile_full_abort:
            break
    del _profile_full_session
    profile_short_full_timings = pd.DataFrame(profile_short_full_rows)
    profile_short_full_output = profile_root / "reports" / "profiling" / "transistor_binary" / ("confirm_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    profile_short_full_output.mkdir(parents=True, exist_ok=False)
    profile_short_full_timings.to_csv(profile_short_full_output / "timings.csv", index=False)
    (profile_short_full_output / "context.json").write_text(json.dumps({"completed": not _profile_full_abort,
        "threads": int(profile_short_full_threads.value), "inter_threads": 1,
        "source": PROFILE_CONFIG["source"], "runtime_variant": "EfficientAD CPU explicit threads; other stages unchanged",
        "profiler": False, "nvtx": False, "warmup_per_image": 1, "measured_repeats": 2,
        "note": "Four measured calls; descriptive timings, not a robust p95 or production accuracy estimate."}, indent=2), encoding="utf-8")
    print("Guardado:", profile_short_full_output)
    mo.ui.table(profile_short_full_timings, selection=None)

    return


@app.cell
def profile_windows_cpu(ctypes, json, os, subprocess):
    # Diagnóstico de Windows: consultar EcoQoS y plan energético; no cambia ajustes.
    from ctypes import wintypes
    class ProfilePowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    profile_windows_cpu = {}
    if os.name == "nt":
        _k32 = ctypes.windll.kernel32
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        _k32.GetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        _state = ProfilePowerThrottlingState(1, 0, 0)
        _ok = _k32.GetProcessInformation(_k32.GetCurrentProcess(), 4, ctypes.byref(_state), ctypes.sizeof(_state))
        profile_windows_cpu = {"query_success": bool(_ok), "control_mask": int(_state.ControlMask), "state_mask": int(_state.StateMask), "ecoqos_explicit": bool(_state.StateMask & 1)}
        for _name, _args in (("plan", ["powercfg", "/getactivescheme"]), ("cpu_max", ["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX"])):
            _r = subprocess.run(_args, capture_output=True, text=True, errors="replace", timeout=10)
            profile_windows_cpu[_name] = _r.stdout + _r.stderr
    print(json.dumps(profile_windows_cpu, indent=2))

    return (profile_windows_cpu,)


@app.cell
def profile_tool_probe_mode(os):
    # Validar la instrumentación con una operación pequeña antes de cargar modelos.
    profile_tool_probe_mode = os.environ.get("QC_PROFILE_TOOL_PROBE", "0") == "1"

    return (profile_tool_probe_mode,)


@app.cell
def profile_tool_probe_workload(
    Path,
    json,
    mo,
    os,
    profile_tool_probe_mode,
    torch,
):
    # Esta sonda valida CUDA/NVTX; NO es un benchmark del inspector.
    mo.stop(not profile_tool_probe_mode)
    profile_probe_dir = Path(os.environ["QC_PROFILE_TRACE_DIR"])
    profile_probe_dir.mkdir(parents=True, exist_ok=True)
    _profile_probe_tensor = torch.ones((32, 32), device="cuda")
    _profile_probe_tensor @ _profile_probe_tensor
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range("inspect/tool_probe"):
            _profile_probe_result = _profile_probe_tensor @ _profile_probe_tensor
            torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    (profile_probe_dir / "probe.json").write_text(json.dumps({"pid": os.getpid(), "torch": torch.__version__, "cuda": torch.version.cuda, "sum": float(_profile_probe_result.sum().cpu()), "probe_only": True}, indent=2), encoding="utf-8")
    print("Sonda CUDA/NVTX ejecutada.")

    return


@app.cell
def profile_tool_probe_control(mo):
    # Lanzamiento de sonda: run_cell explícito mediante MCP. Nunca se lanza al abrir.
    profile_probe_launch_button = mo.ui.run_button(label="Validar compatibilidad de Nsight con una sonda CUDA")
    profile_probe_launch_button

    return (profile_probe_launch_button,)


@app.cell
def profile_tool_probe_launch(
    datetime,
    mo,
    os,
    profile_active_nsys,
    profile_active_nsys_help,
    profile_active_nsys_version,
    profile_capture_mode,
    profile_probe_launch_button,
    profile_root,
    profile_tool_probe_mode,
    profile_trace_reports_root,
    profile_uv_executable,
    subprocess,
    timezone,
):
    mo.stop(profile_tool_probe_mode or profile_capture_mode or not profile_probe_launch_button.value)
    profile_probe_launch_dir = profile_trace_reports_root / ("tool_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    profile_probe_launch_dir.mkdir(parents=True, exist_ok=False)
    profile_probe_log = (profile_probe_launch_dir / "nsys.log").open("w", encoding="utf-8")
    _profile_probe_env = os.environ.copy()
    _profile_probe_env["QC_PROFILE_NSYS_VERSION"] = profile_active_nsys_version
    _profile_probe_env["QC_PROFILE_TOOL_PROBE"] = "1"
    _profile_probe_env["QC_PROFILE_CAPTURE"] = "0"
    _profile_probe_env["QC_PROFILE_TRACE_DIR"] = str(profile_probe_launch_dir)
    _profile_probe_argv = [profile_uv_executable, "run", profile_active_nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false", "--wait=primary", "--output=" + str(profile_probe_launch_dir / "probe"), str(profile_root / ".venv" / "Scripts" / "python.exe"), "notebooks/industrial_profile.py"]
    if "--cuda-event-trace" in profile_active_nsys_help:
        _profile_probe_argv.insert(4,"--cuda-event-trace=false")
    profile_probe_process = subprocess.Popen(_profile_probe_argv, cwd=profile_root, env=_profile_probe_env, stdout=profile_probe_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    print("Sonda en segundo plano, PID:",profile_probe_process.pid, "log:",profile_probe_launch_dir / "nsys.log")

    return profile_probe_launch_dir, profile_probe_log, profile_probe_process


@app.cell
def profile_thread_background_setup(
    Image,
    json,
    mo,
    np,
    ort,
    pd,
    profile_models_dir,
    time,
    torch,
):
    # Trabajo nativo en segundo plano: el MCP queda disponible para observar el avance.
    import concurrent.futures as profile_futures
    import threading
    profile_thread_jobs = {}
    profile_thread_executor = profile_futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ort-profile")

    def profile_thread_worker(output_dir, image_path, expected_score):
        _rows = []
        _context = {"status": "running", "protocol": "same GOOD image, one warmup + one timed call per configuration", "threads_order": [4, 2, 1], "spinning": False, "inter_threads": 1, "execution": "ORT_SEQUENTIAL", "background_worker": True, "expected_calls": 6, "source": "industrial_sam.py EffAD input recipe"}
        def _save():
            pd.DataFrame(_rows).to_csv(output_dir / "timings.csv", index=False)
            (output_dir / "context.json").write_text(json.dumps(_context, indent=2), encoding="utf-8")
            profile_thread_jobs["progress"] = dict(_context, calls=len(_rows), rows=list(_rows))
        try:
            with Image.open(image_path) as _opened:
                _im = _opened.convert("RGB")
            _input = torch.from_numpy(np.asarray(_im.resize((256, 256)), dtype="float32") / 255.0).permute(2,0,1).unsqueeze(0).numpy()
            for _threads in (4, 2, 1):
                _context["active_threads"] = _threads
                _save()
                _opt = ort.SessionOptions()
                _opt.intra_op_num_threads = _threads
                _opt.inter_op_num_threads = 1
                _opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                _opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
                _opt.add_session_config_entry("session.inter_op.allow_spinning", "0")
                _session = ort.InferenceSession(str(profile_models_dir / "effad.onnx"), sess_options=_opt, providers=["CPUExecutionProvider"])
                for _trial in (0,1):
                    _context["active_trial"] = _trial
                    _save()
                    _start = time.perf_counter()
                    _score = float(np.asarray(_session.run(None,{"input":_input})[0]).reshape(-1)[0])
                    _ms = (time.perf_counter()-_start)*1000
                    _matches = bool(np.isclose(_score, expected_score, rtol=1e-4, atol=1e-6))
                    _rows.append({"threads":_threads,"trial":_trial,"warmup":_trial==0,"ms":_ms,"effad":_score,"reference_effad":expected_score,"score_match":_matches})
                    _save()
                    if not _matches:
                        raise RuntimeError("La configuración cambia el score de referencia.")
                del _session
            _summary = pd.DataFrame(_rows).query("warmup == False").sort_values("ms")
            _summary.to_csv(output_dir / "thread_summary.csv", index=False)
            _context.update(status="complete", selected_threads=int(_summary.iloc[0].threads), measured_samples_per_threads=1,
                            note="Exploratory comparison, no p95; spinning disabled for every variant. Does not isolate thread count from reference default spinning.")
            _save()
            return {"summary":_summary.to_dict(orient="records"),"directory":str(output_dir),"selected_threads":_context["selected_threads"]}
        except Exception as _exc:
            _context.update(status="failed", error=repr(_exc))
            _save()
            raise

    profile_thread_background_button = mo.ui.run_button(label="Ejecutar comparación CPU observable (4, 2, 1 hilos)")
    mo.vstack([mo.md("### Comparación observable\n\nProtocolo reducido: una GOOD fija, calentamiento y una medición por configuración. Sin spinning en las tres variantes. Es exploratorio; la BAD se comprobará en el pipeline completo. El estado y los CSV se guardan después de cada llamada."),profile_thread_background_button])

    return (
        profile_futures,
        profile_thread_background_button,
        profile_thread_executor,
        profile_thread_jobs,
        profile_thread_worker,
        threading,
    )


@app.cell
def profile_thread_background_launch(
    datetime,
    mo,
    pd,
    profile_capture_mode,
    profile_manifest,
    profile_saved_report_dir,
    profile_thread_background_button,
    profile_thread_executor,
    profile_thread_jobs,
    profile_thread_worker,
    profile_tool_probe_mode,
    profile_trace_reports_root,
    timezone,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_thread_background_button.value)
    if "future" in profile_thread_jobs and not profile_thread_jobs["future"].done():
        print("La comparación ya está en ejecución.")
    else:
        _output = profile_trace_reports_root / ("cpu_observable_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        _output.mkdir(parents=True,exist_ok=False)
        _row = profile_manifest[profile_manifest.label=="good"].iloc[0]
        _prior = pd.read_csv(profile_saved_report_dir / "latencies.csv")
        _expected = float(_prior[_prior.filename==_row.filename].iloc[0].effad)
        profile_thread_jobs["directory"] = _output
        profile_thread_jobs["future"] = profile_thread_executor.submit(profile_thread_worker, _output, _row.path, _expected)
        print("Trabajo CPU iniciado en segundo plano:",_output)

    return


@app.cell
def profile_trace_validation_helpers(json, profile_active_nsys, subprocess):
    import sqlite3

    def profile_validate_nsys_trace(trace_path, expected_range):
        _stats = {}
        for _name in ("nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum"):
            _run = subprocess.run([profile_active_nsys,"stats","--report="+_name,"--format=csv",str(trace_path)],capture_output=True,text=True,errors="replace",timeout=120)
            _text = _run.stdout + "\n" + _run.stderr
            (trace_path.parent / (_name+".txt")).write_text(_text,encoding="utf-8")
            _stats[_name] = {"exit_code":_run.returncode,"text":_text[:12000]}
        _sql = trace_path.with_suffix(".sqlite")
        _validation = {"trace":str(trace_path),"sqlite_exists":_sql.is_file(),"stats":_stats}
        if _sql.is_file():
            with sqlite3.connect(_sql) as _db:
                _tables = {r[0] for r in _db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                _kt = next((t for t in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in _tables),None)
                _validation["cuda_kernel_count"] = int(_db.execute('SELECT COUNT(*) FROM "'+_kt+'"').fetchone()[0]) if _kt else 0
                _validation["cuda_kernels_ms"] = float(_db.execute('SELECT COALESCE(SUM(end-start),0)/1e6 FROM "'+_kt+'"').fetchone()[0]) if _kt else 0
                _messages = []
                if "NVTX_EVENTS" in _tables:
                    _cols = {r[1] for r in _db.execute('PRAGMA table_info("NVTX_EVENTS")')}
                    if "text" in _cols:
                        _messages.extend(r[0] for r in _db.execute("SELECT text FROM NVTX_EVENTS WHERE text IS NOT NULL"))
                    if "textId" in _cols and "StringIds" in _tables:
                        _messages.extend(r[0] for r in _db.execute("SELECT s.value FROM NVTX_EVENTS n JOIN StringIds s ON s.id=n.textId"))
                _validation["inspect_ranges"] = sorted(set(m for m in _messages if isinstance(m,str) and m.startswith("inspect/")))
                _validation["expected_range_found"] = expected_range in _validation["inspect_ranges"]
                _validation["trace_validated"] = _validation["cuda_kernel_count"] > 0 and _validation["expected_range_found"]
        else:
            _validation["trace_validated"] = False
        (trace_path.parent / "trace_validation.json").write_text(json.dumps(_validation,indent=2),encoding="utf-8")
        return _validation

    print("Validador listo: exige kernels CUDA y el rango NVTX esperado.")
    return profile_validate_nsys_trace, sqlite3


@app.cell
def profile_probe_validation(
    mo,
    profile_probe_launch_dir,
    profile_probe_process,
    profile_validate_nsys_trace,
):
    mo.stop(profile_probe_process.poll() is None,mo.md("Sonda de Nsight en ejecución; validar al terminar."))
    assert profile_probe_process.poll() == 0, "La sonda no ha terminado correctamente."
    profile_probe_validation = profile_validate_nsys_trace(profile_probe_launch_dir / "probe.nsys-rep", "inspect/tool_probe")
    print({k:v for k,v in profile_probe_validation.items() if k != "stats"})
    assert profile_probe_validation["trace_validated"], "La sonda no contiene CUDA/NVTX válidos."
    return


@app.cell
def profile_confirm_background_setup(
    PROFILE_CONFIG,
    datetime,
    json,
    np,
    ort,
    pd,
    profile_device,
    profile_infer,
    profile_manifest,
    profile_models_dir,
    profile_saved_report_dir,
    profile_trace_reports_root,
    time,
    timezone,
    torch,
):
    # Confirmación completa observable: mantiene la sesión original y usa una variante aparte.
    profile_confirm_jobs = {}

    def profile_confirm_worker(threads):
        _gpu_before = torch.cuda.memory_allocated(profile_device)
        assert _gpu_before < 0.75 * torch.cuda.get_device_properties(profile_device).total_memory, "Demasiadas asignaciones GPU antes de medir."
        torch.cuda.reset_peak_memory_stats(profile_device)
        _output = profile_trace_reports_root / ("confirm_observable_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        _output.mkdir(parents=True,exist_ok=False)
        _options = ort.SessionOptions()
        _options.intra_op_num_threads = int(threads)
        _options.inter_op_num_threads = 1
        _options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _options.add_session_config_entry("session.set_denormal_as_zero", "1")
        _options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        _options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        _session = ort.InferenceSession(str(profile_models_dir / "effad.onnx"),sess_options=_options,providers=["CPUExecutionProvider"])
        _inputs = pd.concat([profile_manifest[profile_manifest.label==label].head(1) for label in ("good","bad")],ignore_index=True)
        _baseline = pd.read_csv(profile_saved_report_dir / "latencies.csv").drop_duplicates("filename").set_index("filename")
        _rows = []
        _context = {"status":"running","threads":int(threads),"spinning":False,"flush_denormal":True,"source":PROFILE_CONFIG["source"],"scenario":"resident_models_threads_and_ftz_variant","profiler":False,"gpu_allocated_before_gb":_gpu_before/1e9,"expected_calls":6,"warmups":2,"timed_calls":4,"note":"Small confirmation, no robust p95."}
        try:
            for _repeat in range(3):
                for _row in _inputs.itertuples():
                    profile_confirm_jobs["progress"] = dict(_context,completed=len(_rows),filename=_row.filename,repeat=_repeat,directory=str(_output))
                    torch.cuda.synchronize(profile_device)
                    _start = time.perf_counter()
                    _prediction = profile_infer(_row.path,collect=False,nvtx=False,effad_session=_session,flush_denormal=True)
                    torch.cuda.synchronize(profile_device)
                    _ms = (time.perf_counter()-_start)*1000
                    _reference = _baseline.loc[_row.filename]
                    _equiv = bool(_reference.bad)==_prediction["bad"] and all(np.isclose(float(_reference[k]),_prediction[k],rtol=1e-4,atol=1e-6) for k in ("dino","effad","fused"))
                    _rows.append({"filename":_row.filename,"label":_row.label,"repeat":_repeat,"warmup":_repeat==0,"latency_ms":_ms,"equivalent":bool(_equiv),**{k:_prediction[k] for k in ("dino","effad","fused","bad")}})
                    pd.DataFrame(_rows).to_csv(_output/"timings.csv",index=False)
                    if not _equiv:
                        raise RuntimeError("La confirmación cambia predicciones/scores.")
            _timed = pd.DataFrame(_rows).query("warmup == False")
            _context.update(status="complete",latency_valid=True,peak_allocated_gb=torch.cuda.max_memory_allocated(profile_device)/1e9,median_ms=float(_timed.latency_ms.median()),mean_ms=float(_timed.latency_ms.mean()),min_ms=float(_timed.latency_ms.min()),max_ms=float(_timed.latency_ms.max()),equivalent=True)
            (_output/"context.json").write_text(json.dumps(_context,indent=2),encoding="utf-8")
            profile_confirm_jobs["progress"] = dict(_context,completed=len(_rows),directory=str(_output))
            return dict(_context,directory=str(_output))
        except Exception as _exc:
            _context.update(status="failed",error=repr(_exc))
            (_output/"context.json").write_text(json.dumps(_context,indent=2),encoding="utf-8")
            profile_confirm_jobs["progress"] = dict(_context,completed=len(_rows),directory=str(_output))
            raise

    print("Confirmación en segundo plano preparada, con guardado por imagen.")

    return profile_confirm_jobs, profile_confirm_worker


@app.cell
def profile_job_monitor(
    json,
    mo,
    pd,
    profile_confirm_jobs,
    profile_thread_jobs,
):
    # Estado visible: volver a ejecutar esta celda refresca los resultados, sin inferencias.
    profile_job_states = []
    for _name,_registry in (("CPU threads",profile_thread_jobs),("Confirmación",profile_confirm_jobs)):
        _state = _registry.get("progress",{"status":"not_started"})
        profile_job_states.append({"job":_name,**{k:v for k,v in _state.items() if k not in ("rows",)}})
    print(json.dumps(profile_job_states,indent=2))
    mo.ui.table(pd.DataFrame(profile_job_states),selection=None)

    return


@app.cell
def profile_capture_launch_control(mo):
    profile_capture_jobs = {}
    profile_capture_launch_button = mo.ui.run_button(label="Capturar inspector con Nsight (libera primero los modelos GPU de este kernel)")
    profile_capture_variant = mo.ui.dropdown(options={"Referencia CPU sin FTZ":False,"Variante FTZ + 4 hilos":True},value="Referencia CPU sin FTZ",label="Escenario de captura")
    mo.vstack([profile_capture_variant,profile_capture_launch_button])

    return (
        profile_capture_jobs,
        profile_capture_launch_button,
        profile_capture_variant,
    )


@app.cell
def profile_capture_launch(
    datetime,
    mo,
    os,
    profile_active_nsys,
    profile_active_nsys_help,
    profile_active_nsys_version,
    profile_capture_jobs,
    profile_capture_launch_button,
    profile_capture_mode,
    profile_capture_variant,
    profile_device,
    profile_dino_model,
    profile_execution_ledger,
    profile_root,
    profile_sam_model,
    profile_thread_jobs,
    profile_tool_probe_mode,
    profile_trace_reports_root,
    profile_uv_executable,
    subprocess,
    timezone,
    torch,
):
    # Se permite solo después de cerrar las pruebas CPU/sin profiler.
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_capture_launch_button.value)
    assert not profile_thread_jobs.get("future") or profile_thread_jobs["future"].done(), "Esperar el estudio CPU."
    assert profile_execution_ledger.get("confirmation",{}).get("latency_valid"), "Falta confirmación válida sin profiler en la bitácora."
    if profile_capture_jobs.get("process") is not None and profile_capture_jobs["process"].poll() is None:
        print("La captura ya está en marcha.")
    else:
        print("Descargando de GPU los modelos residentes antes de iniciar el proceso capturado.",flush=True)
        profile_sam_model.to("cpu")
        profile_dino_model.to("cpu")
        torch.cuda.synchronize(profile_device)
        torch.cuda.empty_cache()
        profile_capture_jobs["parent_gpu_allocated_bytes"] = torch.cuda.memory_allocated(profile_device)
        _capture_ftz = bool(profile_capture_variant.value)
        _outdir = profile_trace_reports_root / (("nsight_ftz_" if _capture_ftz else "nsight_reference_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        _outdir.mkdir(parents=True,exist_ok=False)
        _log = (_outdir/"nsys.log").open("w",encoding="utf-8")
        _env = os.environ.copy()
        _env["QC_PROFILE_NSYS_VERSION"] = profile_active_nsys_version
        _env["QC_PROFILE_CAPTURE"] = "1"
        _env["QC_PROFILE_TOOL_PROBE"] = "0"
        _env["QC_PROFILE_CAPTURE_FTZ"] = "1" if _capture_ftz else "0"
        _env["QC_PROFILE_TRACE_DIR"] = str(_outdir)
        _argv = [profile_uv_executable,"run",profile_active_nsys,"profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none","--capture-range=cudaProfilerApi","--capture-range-end=stop","--kill=false","--wait=primary","--force-overwrite=false","--output="+str(_outdir/"capture"),str(profile_root/".venv"/"Scripts"/"python.exe"),"notebooks/industrial_profile.py"]
        if profile_capture_jobs.get("log") and not profile_capture_jobs["log"].closed:
            profile_capture_jobs["log"].close()
        if "--cuda-event-trace" in profile_active_nsys_help:
            _argv.insert(4,"--cuda-event-trace=false")
        profile_capture_jobs.setdefault("captures",[]).append({"directory":str(_outdir),"flush_denormal":_capture_ftz,"cpu_sampling":False,"cpu_context_switches":False})
        profile_capture_jobs.update(directory=_outdir,log=_log,argv=_argv,status="running",flush_denormal=_capture_ftz)
        try:
            profile_capture_jobs["process"] = subprocess.Popen(_argv,cwd=profile_root,env=_env,stdout=_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        except Exception:
            _log.close()
            profile_sam_model.to(profile_device)
            profile_dino_model.to(profile_device)
            profile_capture_jobs["status"] = "launch_failed"
            raise
        print("Captura lanzada en segundo plano:",_outdir, "PID",profile_capture_jobs["process"].pid)

    return


@app.cell
def profile_subnormal_inspection(np, pd, profile_models_dir):
    # Hipótesis adicional: subnormales en los pesos pueden penalizar CPU.
    # Solo inspección de pesos; no los modifica ni activa flush-to-zero.
    import onnx
    from onnx import numpy_helper as profile_onnx_numpy
    _profile_onnx_graph = onnx.load(str(profile_models_dir / "effad.onnx"),load_external_data=True)
    profile_subnormal_rows = []
    for _tensor in _profile_onnx_graph.graph.initializer:
        _array = profile_onnx_numpy.to_array(_tensor)
        if np.issubdtype(_array.dtype,np.floating):
            _abs = np.abs(_array)
            _count = int(np.count_nonzero((_abs > 0) & (_abs < np.finfo(_array.dtype).tiny)))
            if _count:
                profile_subnormal_rows.append({"tensor":_tensor.name,"elements":int(_array.size),"subnormal":_count})
    profile_subnormal_weights = pd.DataFrame(profile_subnormal_rows)
    print("Tensores con subnormales:",len(profile_subnormal_rows),"valores:",sum(r["subnormal"] for r in profile_subnormal_rows))
    if len(profile_subnormal_rows):
        print(profile_subnormal_weights.sort_values("subnormal",ascending=False).head(10).to_string(index=False))

    return (profile_subnormal_weights,)


@app.cell
def profile_ftz_setup(
    Image,
    PROFILE_CONFIG,
    datetime,
    json,
    mo,
    np,
    ort,
    pd,
    profile_futures,
    profile_manifest,
    profile_models_dir,
    profile_saved_report_dir,
    profile_thread_jobs,
    profile_trace_reports_root,
    time,
    timezone,
    torch,
):
    # Hipótesis documentada por ORT: subnormales -> coste CPU; FTZ/DAZ es variante numérica explícita.
    profile_ftz_jobs = {}
    profile_ftz_executor = profile_futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix="ort-ftz")

    def profile_ftz_worker():
        _out = profile_trace_reports_root / ("cpu_ftz_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        _out.mkdir(parents=True,exist_ok=False)
        _baseline = pd.read_csv(profile_saved_report_dir/"latencies.csv").drop_duplicates("filename").set_index("filename")
        _rows = []
        _start_all = time.perf_counter()
        _context = {"status":"running","flush_denormal":True,"spinning":False,"source":PROFILE_CONFIG["source"],"reference_path":str(profile_saved_report_dir),"weights_modified":False,"concurrent_unoptimized_baseline":not profile_thread_jobs["future"].done(),"note":"FTZ changes treatment of subnormal arithmetic; verify scores/decisions before adopting."}
        torch.set_flush_denormal(True)  # Solo este hilo de trabajo; pools ORT se configuran abajo.
        try:
            for _threads in (1,2,4):
                _opt=ort.SessionOptions()
                _opt.intra_op_num_threads=_threads
                _opt.inter_op_num_threads=1
                _opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                _opt.add_session_config_entry("session.set_denormal_as_zero","1")
                _opt.add_session_config_entry("session.intra_op.allow_spinning","0")
                _opt.add_session_config_entry("session.inter_op.allow_spinning","0")
                _sess=ort.InferenceSession(str(profile_models_dir/"effad.onnx"),sess_options=_opt,providers=["CPUExecutionProvider"])
                for _index,(_label,_trial) in enumerate((("good",0),("good",1),("good",2),("bad",1))):
                    _row=profile_manifest[profile_manifest.label==_label].iloc[0]
                    with Image.open(_row.path) as _opened:
                        _im=_opened.convert("RGB")
                    _input=torch.from_numpy(np.asarray(_im.resize((256,256)),dtype="float32")/255.0).permute(2,0,1).unsqueeze(0).numpy()
                    profile_ftz_jobs["progress"] = dict(_context,active_threads=_threads,active_image=_row.filename,calls=len(_rows),directory=str(_out))
                    _t0=time.perf_counter()
                    _score=float(np.asarray(_sess.run(None,{"input":_input})[0]).reshape(-1)[0])
                    _ms=(time.perf_counter()-_t0)*1000
                    _expected=float(_baseline.loc[_row.filename].effad)
                    _match=bool(np.isclose(_score,_expected,rtol=1e-4,atol=1e-6))
                    _rows.append({"threads":_threads,"filename":_row.filename,"label":_label,"trial":_trial,"warmup":_index==0,"ms":_ms,"effad":_score,"reference_effad":_expected,"score_match":_match,"flush_denormal":True})
                    pd.DataFrame(_rows).to_csv(_out/"timings.csv",index=False)
                    if not _match:
                        raise RuntimeError("FTZ cambia el score fuera de tolerancia.")
                    if _ms>30000:
                        raise RuntimeError("FTZ no resuelve la lentitud en esta llamada; revisar antes de ampliar.")
                del _sess
            _summary=pd.DataFrame(_rows).query("warmup == False").groupby("threads",as_index=False).agg(median_ms=("ms","median"),min_ms=("ms","min"),max_ms=("ms","max"),n=("ms","size")).sort_values("median_ms")
            _summary.to_csv(_out/"thread_summary.csv",index=False)
            _context.update(status="complete",selected_threads=int(_summary.iloc[0].threads),calls=len(_rows),seconds=time.perf_counter()-_start_all)
            (_out/"context.json").write_text(json.dumps(_context,indent=2),encoding="utf-8")
            profile_ftz_jobs["progress"]=dict(_context,directory=str(_out),summary=_summary.to_dict(orient="records"))
            return profile_ftz_jobs["progress"]
        except Exception as _exc:
            _context.update(status="failed",error=repr(_exc))
            (_out/"context.json").write_text(json.dumps(_context,indent=2),encoding="utf-8")
            profile_ftz_jobs["progress"]=dict(_context,directory=str(_out),calls=len(_rows))
            raise
        finally:
            torch.set_flush_denormal(False)

    profile_ftz_button = mo.ui.run_button(label="Probar subnormales: FTZ/DAZ en 1, 2 y 4 hilos")
    mo.vstack([mo.md("### Subnormales: hipótesis adicional\n\nSe encontraron pesos subnormales. FTZ/DAZ se prueba como variante numérica explícita; no se editan los pesos. Scores se comparan con la referencia (rtol=1e-4, atol=1e-6). La prueba registra si la línea base lenta seguía en segundo plano. [Opción oficial de ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)."),profile_ftz_button])

    return (
        profile_ftz_button,
        profile_ftz_executor,
        profile_ftz_jobs,
        profile_ftz_worker,
    )


@app.cell
def profile_ftz_launch(
    mo,
    profile_capture_mode,
    profile_ftz_button,
    profile_ftz_executor,
    profile_ftz_jobs,
    profile_ftz_worker,
    profile_tool_probe_mode,
):
    mo.stop(profile_tool_probe_mode or profile_capture_mode or not profile_ftz_button.value)
    if profile_ftz_jobs.get("future") and not profile_ftz_jobs["future"].done():
        print("La prueba FTZ ya está ejecutándose.")
    else:
        profile_ftz_jobs["future"]=profile_ftz_executor.submit(profile_ftz_worker)
        print("Prueba FTZ iniciada en segundo plano.")

    return


@app.cell
def profile_cancel_raw_control(mo):
    # Cancelación cooperativa de la ampliación lenta creada aquí.
    # La llamada nativa en curso se deja terminar. Se impide solo su siguiente llamada.
    # El guard temporal se limita al hilo de este trabajo y se restaura al finalizar.
    profile_cancel_raw_button = mo.ui.run_button(label="No ampliar la serie sin FTZ después de su llamada actual")
    profile_cancel_raw_button

    return (profile_cancel_raw_button,)


@app.cell
def profile_cancel_raw_expansion(
    json,
    mo,
    ort,
    profile_cancel_raw_button,
    profile_capture_mode,
    profile_futures,
    profile_thread_jobs,
    profile_tool_probe_mode,
    threading,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_cancel_raw_button.value)
    profile_raw_cancel_state = {"requested":True,"guard_restored":False,"reason":"Original 30-frame baseline already saved; stop redundant slow expansion after subnormal diagnosis."}
    _profile_raw_future = profile_thread_jobs.get("future")
    if _profile_raw_future is None or _profile_raw_future.done():
        profile_raw_cancel_state["guard_restored"] = True
        print("La serie lenta ya terminó.")
    else:
        profile_original_ort_run = ort.InferenceSession.run
        def profile_raw_cancel_guard(session, *args, **kwargs):
            if threading.current_thread().name.startswith("ort-profile"):
                raise profile_futures.CancelledError("Ampliación sin FTZ cancelada antes de una nueva llamada nativa.")
            return profile_original_ort_run(session,*args,**kwargs)
        def profile_restore_ort_guard(future):
            if ort.InferenceSession.run is profile_raw_cancel_guard:
                ort.InferenceSession.run = profile_original_ort_run
            profile_raw_cancel_state["guard_restored"] = True
            _directory = profile_thread_jobs.get("directory")
            if _directory and (_directory/"context.json").is_file():
                _context = json.loads((_directory/"context.json").read_text(encoding="utf-8"))
                _context.update(status="cancelled",cancel_reason=profile_raw_cancel_state["reason"],guard_restored=True)
                (_directory/"context.json").write_text(json.dumps(_context,indent=2),encoding="utf-8")
                profile_thread_jobs["progress"] = dict(_context,directory=str(_directory))
        ort.InferenceSession.run = profile_raw_cancel_guard
        _profile_raw_future.add_done_callback(profile_restore_ort_guard)
        print("Cancelación solicitada. No se interrumpe ni se altera la llamada en curso.")

    return (profile_raw_cancel_state,)


@app.cell
def profile_confirm_final_control(mo):
    profile_confirm_final_button = mo.ui.run_button(label="Confirmar pipeline completo con la variante FTZ elegida")
    profile_confirm_final_button

    return (profile_confirm_final_button,)


@app.cell
def profile_confirm_final_launch(
    mo,
    profile_capture_mode,
    profile_confirm_final_button,
    profile_confirm_jobs,
    profile_confirm_worker,
    profile_ftz_jobs,
    profile_raw_cancel_state,
    profile_thread_executor,
    profile_thread_jobs,
    profile_tool_probe_mode,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_confirm_final_button.value)
    assert profile_ftz_jobs["future"].done(), "Esperar la comparación FTZ."
    _profile_ftz_result=profile_ftz_jobs["future"].result()
    assert _profile_ftz_result["status"]=="complete"
    assert profile_thread_jobs["future"].done(), "Esperar que termine la llamada lenta y se restaure el guard de cancelación."
    assert profile_raw_cancel_state["guard_restored"]
    if profile_confirm_jobs.get("future") and not profile_confirm_jobs["future"].done():
        print("Confirmación ya en ejecución.")
    else:
        profile_confirm_jobs["future"]=profile_thread_executor.submit(profile_confirm_worker,_profile_ftz_result["selected_threads"])
        print("Confirmación iniciada en segundo plano, threads=",_profile_ftz_result["selected_threads"])

    return


@app.cell
def profile_capture_ftz_mode(os):
    profile_capture_ftz_mode = os.environ.get("QC_PROFILE_CAPTURE_FTZ","0")=="1"

    return (profile_capture_ftz_mode,)


@app.cell
def profile_gpu_cleanup_control(mo):
    profile_gpu_cleanup_button = mo.ui.run_button(label="Liberar copias GPU inactivas del profiling")
    profile_gpu_cleanup_button

    return (profile_gpu_cleanup_button,)


@app.cell
def profile_gpu_cleanup(
    json,
    mo,
    profile_capture_mode,
    profile_confirm_worker,
    profile_gpu_cleanup_button,
    profile_tool_probe_mode,
    threading,
    torch,
):
    # Solo copias inactivas. No mover modelos usados por la inferencia en curso.
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_gpu_cleanup_button.value)
    import gc as profile_gc
    import sys as profile_sys
    _profile_active_models = set()
    profile_canonical_runtime = {}
    for _thread in threading.enumerate():
        if _thread.name.startswith("ort-profile"):
            _frame = profile_sys._current_frames().get(_thread.ident)
            while _frame:
                if _frame.f_code.co_name == "profile_infer":
                    profile_canonical_runtime.update({k:_frame.f_globals[k] for k in ("profile_device","profile_sam_model","profile_sam_processor","profile_dino_model","profile_dino_processor","profile_logreg_session","profile_effad_session","profile_load_records","profile_load_seconds","profile_load_table") if k in _frame.f_globals})
                    for _name in ("profile_sam_model","profile_dino_model"):
                        _model = _frame.f_globals.get(_name)
                        if _model is not None:
                            _profile_active_models.add(id(_model))
                _frame = _frame.f_back
    if not _profile_active_models:
        _namespace = profile_confirm_worker.__globals__["profile_infer"].__globals__
        profile_canonical_runtime.update({k:_namespace[k] for k in ("profile_device","profile_sam_model","profile_sam_processor","profile_dino_model","profile_dino_processor","profile_logreg_session","profile_effad_session","profile_load_records","profile_load_seconds","profile_load_table") if k in _namespace})
        _profile_active_models = {id(profile_canonical_runtime[k]) for k in ("profile_sam_model","profile_dino_model")}
    assert len(_profile_active_models)==2, "No se identificaron exactamente los dos modelos de la inferencia activa."
    profile_gpu_cleanup_report = {"before_allocated_gb":torch.cuda.memory_allocated()/1e9,"before_reserved_gb":torch.cuda.memory_reserved()/1e9,"active_model_ids":sorted(_profile_active_models),"models":[]}
    _profile_models = [o for o in profile_gc.get_objects() if type(o).__name__ in ("Sam3Model","Dinov2Model")]
    for _model in _profile_models:
        _device=str(next(_model.parameters()).device)
        _move=_device.startswith("cuda") and id(_model) not in _profile_active_models
        profile_gpu_cleanup_report["models"].append({"id":id(_model),"type":type(_model).__name__,"device_before":_device,"offloaded":_move})
        if _move:
            _model.to("cpu")
    torch.cuda.empty_cache()
    profile_gc.collect()
    profile_gpu_cleanup_report.update(after_allocated_gb=torch.cuda.memory_allocated()/1e9,after_reserved_gb=torch.cuda.memory_reserved()/1e9)
    del _profile_models, _model
    print(json.dumps(profile_gpu_cleanup_report,indent=2))

    return profile_canonical_runtime, profile_gpu_cleanup_report


@app.cell
def profile_model_registry():
    profile_model_registry = {}
    return (profile_model_registry,)


@app.cell(disabled=True)
def profile_adopt_runtime(
    Path,
    json,
    mo,
    profile_canonical_runtime,
    profile_capture_ftz_mode,
    profile_capture_mode,
    profile_confirm_jobs,
    profile_gpu_cleanup_report,
    profile_model_registry,
    profile_tool_probe_mode,
):
    # Reparar el registro del kernel y conservar la confirmación contaminada para auditoría.
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    assert profile_confirm_jobs["future"].done()
    profile_discarded_confirm_report = profile_confirm_jobs["future"].result()
    _profile_bad_dir = Path(profile_discarded_confirm_report["directory"])
    _profile_bad_context = json.loads((_profile_bad_dir/"context.json").read_text(encoding="utf-8"))
    _profile_bad_context.update(latency_valid=False,invalid_latency_reason="Duplicate GPU model instances exceeded physical VRAM; cleanup occurred during this confirmation.",gpu_cleanup=profile_gpu_cleanup_report,equivalence_valid=True)
    (_profile_bad_dir/"context.json").write_text(json.dumps(_profile_bad_context,indent=2),encoding="utf-8")
    profile_model_registry.update(profile_canonical_runtime)
    profile_model_registry.update(_initialized=True,_capture_ftz=profile_capture_ftz_mode)
    print("Runtime único adoptado; confirmación previa marcada no válida para latencia:",_profile_bad_dir)

    return


@app.cell
def profile_memory_stability(
    mo,
    profile_capture_mode,
    profile_confirm_jobs,
    profile_device,
    profile_tool_probe_mode,
    torch,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    assert not profile_confirm_jobs.get("future") or profile_confirm_jobs["future"].done(), "Esperar la inferencia antes de liberar caché."
    torch.cuda.empty_cache()
    profile_memory_stability = {"allocated_gb":torch.cuda.memory_allocated(profile_device)/1e9,"reserved_gb":torch.cuda.memory_reserved(profile_device)/1e9,"device_gb":torch.cuda.get_device_properties(profile_device).total_memory/1e9}
    print(profile_memory_stability)

    return


@app.cell
def profile_execution_ledger():
    # Conservar resultados y procesos independientemente de las celdas de carga.
    profile_execution_ledger = {}

    return (profile_execution_ledger,)


@app.cell
def profile_raw_debug_control(mo):
    profile_raw_debug_button=mo.ui.run_button(label="Diagnosticar la ejecución del notebook capturado sin el lanzador Nsight")
    profile_raw_debug_button

    return (profile_raw_debug_button,)


@app.cell
def profile_raw_debug_launch(
    datetime,
    mo,
    os,
    profile_capture_mode,
    profile_execution_ledger,
    profile_raw_debug_button,
    profile_root,
    profile_tool_probe_mode,
    profile_trace_reports_root,
    profile_uv_executable,
    subprocess,
    timezone,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_raw_debug_button.value)
    _profile_out=profile_trace_reports_root/("capture_debug_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    _profile_out.mkdir(parents=True,exist_ok=False)
    _profile_env=os.environ.copy()
    _profile_env["QC_PROFILE_CAPTURE"]="1"
    _profile_env["QC_PROFILE_TOOL_PROBE"]="0"
    _profile_env["QC_PROFILE_CAPTURE_FTZ"]="1"
    _profile_env["QC_PROFILE_TRACE_DIR"]=str(_profile_out)
    _profile_log=(_profile_out/"python.log").open("w",encoding="utf-8")
    profile_execution_ledger["raw_debug"]={"directory":_profile_out,"log":_profile_log,
        "process":subprocess.Popen([profile_uv_executable,"run","python","notebooks/industrial_profile.py"],cwd=profile_root,env=_profile_env,stdout=_profile_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)}
    print("Diagnóstico lanzado:",_profile_out)

    return


@app.cell
def profile_record_completed_work(
    profile_confirm_jobs,
    profile_execution_ledger,
    profile_ftz_jobs,
):
    # La bitácora no se borra al redefinir funciones o volver a ejecutar la carga.
    if profile_confirm_jobs.get("future") and profile_confirm_jobs["future"].done():
        _confirmation=profile_confirm_jobs["future"].result()
        if _confirmation.get("latency_valid"):
            profile_execution_ledger["confirmation"]=_confirmation
    if profile_ftz_jobs.get("future") and profile_ftz_jobs["future"].done():
        profile_execution_ledger["ftz_study"]=profile_ftz_jobs["future"].result()
    print("Resultados conservados:",[k for k in profile_execution_ledger if k!='raw_debug'])

    return


@app.cell
def profile_tool_download_setup(Path, hashlib, json, mo, profile_futures):
    # Herramienta portable oficial: no instala drivers ni modifica Program Files.
    import urllib.request
    import zipfile
    profile_tool_download_jobs = {}
    profile_tool_download_executor = profile_futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix="nsys-download")

    def profile_download_nsight_portable():
        _base="https://developer.download.nvidia.com/compute/cuda/redist/"
        _manifest_url=_base+"redistrib_12.9.0.json"
        with urllib.request.urlopen(_manifest_url,timeout=60) as _response:
            _manifest=json.load(_response)
        _component=_manifest["nsight_systems"]
        _entry=_component["windows-x86_64"]
        _relative=_entry["relative_path"]
        assert _relative.startswith("nsight_systems/windows-x86_64/")
        _expected=_entry["sha256"]
        _version=_component["version"]
        _root=Path.home()/".cache"/"qc-vision-tools"/("nsight-systems-"+_version)
        _root.mkdir(parents=True,exist_ok=True)
        _archive=_root/Path(_relative).name
        _state={"status":"downloading","version":_version,"url":_base+_relative,"manifest":_manifest_url,"expected_sha256":_expected,"bytes":0,"directory":str(_root)}
        profile_tool_download_jobs["progress"]=dict(_state)
        if not _archive.is_file() or hashlib.sha256(_archive.read_bytes()).hexdigest()!=_expected:
            with urllib.request.urlopen(_base+_relative,timeout=60) as _response, (_archive.with_suffix(".partial")).open("wb") as _out:
                while True:
                    _chunk=_response.read(4*1024*1024)
                    if not _chunk:
                        break
                    _out.write(_chunk)
                    _state["bytes"]+=len(_chunk)
                    profile_tool_download_jobs["progress"]=dict(_state)
            _partial=_archive.with_suffix(".partial")
            _actual=hashlib.sha256(_partial.read_bytes()).hexdigest()
            assert _actual==_expected,"Checksum oficial no coincide."
            _partial.replace(_archive)
        _state.update(status="extracting",sha256=_expected)
        profile_tool_download_jobs["progress"]=dict(_state)
        _extract=_root/"extracted"
        _extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(_archive) as _zip:
            for _info in _zip.infolist():
                _target=(_extract/_info.filename).resolve()
                assert _target.is_relative_to(_extract.resolve()),"Ruta fuera del directorio de herramientas."
            _zip.extractall(_extract)
        _executables=list(_extract.rglob("nsys.exe"))
        assert _executables,"No hay ejecutable nsys en el paquete oficial."
        _executable=next((p for p in _executables if "target-windows-x64" in str(p)),_executables[0])
        _state.update(status="complete",executable=str(_executable))
        (_root/"verified_download.json").write_text(json.dumps(_state,indent=2),encoding="utf-8")
        profile_tool_download_jobs["progress"]=dict(_state)
        return _state

    profile_tool_download_button=mo.ui.run_button(label="Preparar Nsight portable oficial con checksum verificado")
    profile_tool_download_button

    return (
        profile_download_nsight_portable,
        profile_tool_download_button,
        profile_tool_download_executor,
        profile_tool_download_jobs,
    )


@app.cell
def profile_tool_download_launch(
    mo,
    profile_capture_mode,
    profile_download_nsight_portable,
    profile_tool_download_button,
    profile_tool_download_executor,
    profile_tool_download_jobs,
    profile_tool_probe_mode,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_tool_download_button.value)
    if profile_tool_download_jobs.get("future") and not profile_tool_download_jobs["future"].done():
        print("Descarga ya en marcha.")
    else:
        profile_tool_download_jobs["future"]=profile_tool_download_executor.submit(profile_download_nsight_portable)
        print("Descarga oficial en segundo plano.")

    return


@app.cell
def profile_active_nsight(
    Path,
    json,
    os,
    profile_capture_mode,
    profile_nsys,
    profile_tool_probe_mode,
    subprocess,
):
    # Elegir una instalación portable verificada; no cambia los drivers del sistema.
    profile_active_nsys = profile_nsys
    _profile_tool_root = Path.home()/".cache"/"qc-vision-tools"
    for _verified in sorted(_profile_tool_root.glob("nsight-systems-*/verified_download.json")):
        _tool=json.loads(_verified.read_text(encoding="utf-8"))
        _exe=Path(_tool.get("executable",""))
        if _tool.get("status")=="complete" and _exe.is_file() and _exe.resolve().is_relative_to(_profile_tool_root.resolve()):
            profile_active_nsys=str(_exe)
    assert profile_active_nsys
    if profile_capture_mode or profile_tool_probe_mode:
        profile_active_nsys_version=os.environ.get("QC_PROFILE_NSYS_VERSION","Portable verificado; versión no consultada dentro de la captura")
        profile_active_nsys_help=""
    else:
        _profile_version=subprocess.run([profile_active_nsys,"--version"],capture_output=True,text=True,errors="replace",timeout=20)
        profile_active_nsys_version=_profile_version.stdout+_profile_version.stderr
        _profile_help=subprocess.run([profile_active_nsys,"profile","--help"],capture_output=True,text=True,errors="replace",timeout=20)
        profile_active_nsys_help=_profile_help.stdout+_profile_help.stderr
    print(profile_active_nsys_version)
    print("Ejecutable:",profile_active_nsys)
    _profile_kill_section=profile_active_nsys_help.split("--kill=",1)
    if len(_profile_kill_section)>1:
        print("kill option:",_profile_kill_section[1][:600])

    return (
        profile_active_nsys,
        profile_active_nsys_help,
        profile_active_nsys_version,
    )


@app.cell
def profile_archive_probe_job(
    profile_execution_ledger,
    profile_probe_launch_dir,
    profile_probe_log,
    profile_probe_process,
):
    # Mantener el proceso de sonda registrado aunque se redefina su celda de lanzamiento.
    if "profile_probe_process" in globals():
        profile_execution_ledger["probe_job"]={"process":profile_probe_process,"directory":profile_probe_launch_dir,"log":profile_probe_log}
    print("Sonda registrada:",profile_execution_ledger.get("probe_job",{}).get("directory"))

    return


@app.cell
def profile_validate_current_capture(
    json,
    mo,
    np,
    pd,
    profile_capture_jobs,
    profile_capture_mode,
    profile_execution_ledger,
    profile_saved_report_dir,
    profile_tool_probe_mode,
    profile_validate_nsys_trace,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or "directory" not in profile_capture_jobs)
    # Validación de la captura real, más equivalencia de las dos predicciones.
    _profile_trace_job=profile_capture_jobs
    _profile_trace_path=_profile_trace_job["directory"]/"capture.nsys-rep"
    mo.stop(not _profile_trace_path.is_file() or _profile_trace_path.stat().st_size==0,mo.md("Esperando el archivo de captura Nsight."))
    profile_current_trace_validation=profile_validate_nsys_trace(_profile_trace_path,"inspect/capture")
    assert profile_current_trace_validation["trace_validated"],"No se validaron CUDA/NVTX."
    _profile_cap=pd.read_csv(_profile_trace_path.parent/"capture_predictions.csv")
    _profile_ref=pd.read_csv(profile_saved_report_dir/"latencies.csv").drop_duplicates("filename")
    profile_current_trace_equivalence=_profile_cap.merge(_profile_ref,on="filename",suffixes=("_capture","_reference"),validate="one_to_one")
    assert len(profile_current_trace_equivalence)==2
    for _key in ("dino","effad","fused"):
        profile_current_trace_equivalence[_key+"_match"]=np.isclose(profile_current_trace_equivalence[_key+"_capture"],profile_current_trace_equivalence[_key+"_reference"],rtol=1e-4,atol=1e-6)
    profile_current_trace_equivalence["decision_match"]=profile_current_trace_equivalence.bad_capture==profile_current_trace_equivalence.bad_reference
    assert profile_current_trace_equivalence[["dino_match","effad_match","fused_match","decision_match"]].all().all()
    profile_current_trace_equivalence.to_csv(_profile_trace_path.parent/"capture_equivalence.csv",index=False)
    _profile_context=json.loads((_profile_trace_path.parent/"capture_context.json").read_text(encoding="utf-8"))
    _profile_context.update(trace_validated=True,predictions_equivalent=True)
    (_profile_trace_path.parent/"capture_context.json").write_text(json.dumps(_profile_context,indent=2),encoding="utf-8")
    profile_execution_ledger.setdefault("validated_traces",{})["ftz" if _profile_context["flush_denormal"] else "reference"]={"directory":str(_profile_trace_path.parent),"validation":{k:v for k,v in profile_current_trace_validation.items() if k!="stats"},"context":_profile_context}
    print({k:v for k,v in profile_current_trace_validation.items() if k!="stats"})
    print(profile_current_trace_equivalence[["filename","dino_match","effad_match","fused_match","decision_match"]].to_string(index=False))

    return


@app.cell
def profile_trace_analysis_helpers(Path, json, pd, sqlite3):
    # Lectura del SQLite exportado por Nsight; no consulta la base de MLflow.
    def profile_read_trace_ranges(directory):
        _directory=Path(directory)
        _sql=_directory/"capture.sqlite"
        assert _sql.is_file()
        with sqlite3.connect(_sql) as _db:
            _tables={r[0] for r in _db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            _cols={r[1] for r in _db.execute('PRAGMA table_info("NVTX_EVENTS")')}
            if "textId" in _cols and "StringIds" in _tables:
                _label="COALESCE(NULLIF(n.text,''),s.value)" if "text" in _cols else "s.value"
                _query="SELECT n.start,n.end,"+_label+",n.globalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
            else:
                _query="SELECT start,end,text,globalTid FROM NVTX_EVENTS"
            _rows=[r for r in _db.execute(_query) if r[1] is not None and r[1]>r[0] and isinstance(r[2],str) and r[2].startswith("inspect/")]
            _ranges=pd.DataFrame(_rows,columns=["start_ns","end_ns","stage","global_tid"])
            _ranges["ms"]=(_ranges.end_ns-_ranges.start_ns)/1e6
            _kt=next(t for t in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in _tables)
            _kernels=pd.read_sql_query('SELECT start,end,globalPid FROM "'+_kt+'"',_db)
        _context=json.loads((_directory/"capture_context.json").read_text(encoding="utf-8"))
        _expected_pid=int(_context["environment"]["kernel_pid"])
        _raw_pids={int(v) for v in _kernels.globalPid.dropna()}
        _decoded_pids={(v>>24)&0xFFFFFF for v in _raw_pids}
        assert _expected_pid in (_raw_pids|_decoded_pids),"Los kernels no corresponden al PID de inferencia."
        assert int((_ranges.stage=="inspect/frame").sum())==2,"La ventana no contiene exactamente dos frames."
        _summary=_ranges.groupby("stage",as_index=False).agg(median_ms=("ms","median"),mean_ms=("ms","mean"),n=("ms","size")).sort_values("median_ms",ascending=False)
        _overlap=[]
        for _interval in _ranges[_ranges.stage=="inspect/efficientad_cpu"].itertuples():
            _overlapping=_kernels[(_kernels.end>_interval.start_ns)&(_kernels.start<_interval.end_ns)]
            _gpu_ms=sum(max(0,min(int(r.end),int(_interval.end_ns))-max(int(r.start),int(_interval.start_ns)))/1e6 for r in _overlapping.itertuples())
            _overlap.append({"cpu_range_ms":float(_interval.ms),"pipeline_gpu_kernel_overlap_ms":float(_gpu_ms)})
        _ranges.to_csv(_directory/"nvtx_ranges.csv",index=False)
        _summary.to_csv(_directory/"nvtx_stage_summary.csv",index=False)
        _scope={"expected_pid":_expected_pid,"decoded_kernel_pids":sorted(_decoded_pids),"pid_matches":True,"frames":2,"effad_gpu_overlap":_overlap}
        (_directory/"scope_analysis.json").write_text(json.dumps(_scope,indent=2),encoding="utf-8")
        return _ranges,_summary,_scope

    print("Análisis preparado: rangos host NVTX, PID real y actividad GPU durante EfficientAD.")
    return (profile_read_trace_ranges,)


@app.cell
def profile_compare_captures(
    json,
    mo,
    pd,
    profile_capture_mode,
    profile_execution_ledger,
    profile_read_trace_ranges,
    profile_tool_probe_mode,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not {"reference","ftz"}.issubset(profile_execution_ledger.get("validated_traces",{})),mo.md("Esperando ambas capturas validadas."))
    profile_trace_comparison_rows = []
    profile_trace_scope_results = {}
    for _scenario in ("reference","ftz"):
        _capture = profile_execution_ledger["validated_traces"][_scenario]
        _ranges,_summary,_scope = profile_read_trace_ranges(_capture["directory"])
        profile_trace_scope_results[_scenario] = _scope
        for _row in _summary.itertuples():
            profile_trace_comparison_rows.append({"scenario":_scenario,"stage":_row.stage,"median_ms":float(_row.median_ms),"mean_ms":float(_row.mean_ms),"n":int(_row.n)})
    profile_trace_comparison = pd.DataFrame(profile_trace_comparison_rows)
    profile_trace_pivot = profile_trace_comparison.pivot(index="stage",columns="scenario",values="median_ms").reset_index()
    print(profile_trace_pivot.to_string(index=False))
    print("Ámbito/PID y GPU durante EfficientAD:",json.dumps(profile_trace_scope_results,indent=2))
    profile_execution_ledger["trace_comparison"] = profile_trace_comparison.to_dict(orient="records")
    profile_execution_ledger["trace_scopes"] = profile_trace_scope_results
    mo.ui.table(profile_trace_pivot,selection=None)

    return


@app.cell
def profile_isolated_benchmark_mode(os):
    profile_isolated_benchmark_mode=os.environ.get("QC_PROFILE_NO_NSYS","0")=="1"

    return (profile_isolated_benchmark_mode,)


@app.cell
def profile_isolated_workload(
    PROFILE_CONFIG,
    Path,
    json,
    mo,
    np,
    os,
    pd,
    profile_capture_ftz_mode,
    profile_device,
    profile_effad_session,
    profile_environment,
    profile_infer,
    profile_isolated_benchmark_mode,
    profile_manifest,
    profile_saved_report_dir,
    profile_windows_cpu,
    time,
    torch,
):
    # Misma libreta en un proceso limpio; sin profiler y sin NVTX dentro de los frames.
    mo.stop(not profile_isolated_benchmark_mode)
    _profile_iso_dir=Path(os.environ["QC_PROFILE_TRACE_DIR"])
    _profile_iso_dir.mkdir(parents=True,exist_ok=True)
    _profile_iso_inputs=pd.concat([profile_manifest[profile_manifest.label==label].head(1) for label in ("good","bad")],ignore_index=True)
    _profile_iso_reference=pd.read_csv(profile_saved_report_dir/"latencies.csv").drop_duplicates("filename").set_index("filename")
    _profile_iso_rows=[]
    torch.cuda.reset_peak_memory_stats(profile_device)
    for _repeat in range(3):
        for _row in _profile_iso_inputs.itertuples():
            torch.cuda.synchronize(profile_device)
            _start=time.perf_counter()
            _prediction=profile_infer(_row.path,collect=False,nvtx=False,flush_denormal=profile_capture_ftz_mode)
            torch.cuda.synchronize(profile_device)
            _milliseconds=(time.perf_counter()-_start)*1000
            _expected=_profile_iso_reference.loc[_row.filename]
            _equiv=bool(_expected.bad)==_prediction["bad"] and all(np.isclose(float(_expected[key]),_prediction[key],rtol=1e-4,atol=1e-6) for key in ("dino","effad","fused"))
            _profile_iso_rows.append({"filename":_row.filename,"label":_row.label,"repeat":_repeat,"warmup":_repeat==0,"latency_ms":_milliseconds,"equivalent":bool(_equiv),**{key:_prediction[key] for key in ("dino","effad","fused","bad")}})
            pd.DataFrame(_profile_iso_rows).to_csv(_profile_iso_dir/"isolated_timings.csv",index=False)
            print(f"Aislado FTZ={profile_capture_ftz_mode}, repeat={_repeat}, {_row.filename}: {_milliseconds:.1f} ms, equiv={_equiv}",flush=True)
            assert _equiv,"Los scores/decisiones no coinciden con la referencia."
    _profile_iso_timed=pd.DataFrame(_profile_iso_rows).query("warmup == False")
    _profile_iso_summary={"status":"complete","profiler":False,"nvtx_during_measurement":False,"flush_denormal":profile_capture_ftz_mode,"effad_threads":profile_effad_session.get_session_options().intra_op_num_threads,"unique_images":2,"measurements":4,"warmup_calls":2,"median_ms":float(_profile_iso_timed.latency_ms.median()),"mean_ms":float(_profile_iso_timed.latency_ms.mean()),"min_ms":float(_profile_iso_timed.latency_ms.min()),"max_ms":float(_profile_iso_timed.latency_ms.max()),"peak_allocated_gb":torch.cuda.max_memory_allocated(profile_device)/1e9,"predictions_equivalent":True,"environment":profile_environment,"windows_cpu":profile_windows_cpu,"source":PROFILE_CONFIG["source"],"note":"Two unique test images, two measured repetitions each; descriptive timings, no robust p95."}
    (_profile_iso_dir/"isolated_summary.json").write_text(json.dumps(_profile_iso_summary,indent=2),encoding="utf-8")
    print(json.dumps(_profile_iso_summary,indent=2),flush=True)

    return


@app.cell
def profile_isolated_pair_setup(
    datetime,
    json,
    mo,
    os,
    profile_active_nsys_version,
    profile_execution_ledger,
    profile_futures,
    profile_root,
    profile_trace_reports_root,
    profile_uv_executable,
    subprocess,
    timezone,
):
    profile_isolated_jobs={}
    profile_isolated_executor=profile_futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix="isolated-profile")

    def profile_isolated_pair_worker():
        _results={}
        for _scenario,_ftz in (("reference",False),("ftz",True)):
            _directory=profile_trace_reports_root/("isolated_"+_scenario+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
            _directory.mkdir(parents=True,exist_ok=False)
            _env=os.environ.copy()
            _env.update(QC_PROFILE_CAPTURE="1",QC_PROFILE_TOOL_PROBE="0",QC_PROFILE_NO_NSYS="1",QC_PROFILE_CAPTURE_FTZ="1" if _ftz else "0",QC_PROFILE_TRACE_DIR=str(_directory),QC_PROFILE_NSYS_VERSION=profile_active_nsys_version,PYTHONUNBUFFERED="1")
            with (_directory/"python.log").open("w",encoding="utf-8") as _log:
                _process=subprocess.Popen([profile_uv_executable,"run","python","notebooks/industrial_profile.py"],cwd=profile_root,env=_env,stdout=_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                profile_isolated_jobs["progress"]={"status":"running","scenario":_scenario,"directory":str(_directory),"pid":_process.pid}
                _returncode=_process.wait()
            assert _returncode==0,"La ejecución aislada falló: "+str(_directory/"python.log")
            _summary=json.loads((_directory/"isolated_summary.json").read_text(encoding="utf-8"))
            assert _summary["predictions_equivalent"]
            _results[_scenario]=dict(_summary,directory=str(_directory))
            profile_isolated_jobs["results"]=dict(_results)
        profile_execution_ledger["isolated_comparison"]=_results
        profile_isolated_jobs["progress"]={"status":"complete","results":_results}
        return _results

    profile_isolated_pair_button=mo.ui.run_button(label="Comparar referencia y FTZ sin profiler, en procesos limpios")
    profile_isolated_pair_button

    return (
        profile_isolated_executor,
        profile_isolated_jobs,
        profile_isolated_pair_button,
        profile_isolated_pair_worker,
    )


@app.cell
def profile_isolated_pair_launch(
    mo,
    profile_capture_jobs,
    profile_capture_mode,
    profile_isolated_executor,
    profile_isolated_jobs,
    profile_isolated_pair_button,
    profile_isolated_pair_worker,
    profile_tool_probe_mode,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_isolated_pair_button.value)
    assert profile_capture_jobs["process"].poll()==0,"Esperar que termine la captura."
    if profile_isolated_jobs.get("future") and not profile_isolated_jobs["future"].done():
        print("Comparación aislada ya en ejecución.")
    else:
        profile_isolated_jobs["future"]=profile_isolated_executor.submit(profile_isolated_pair_worker)
        print("Dos ejecuciones aisladas y secuenciales iniciadas.")

    return


@app.cell
def profile_final_report(
    PROFILE_CONFIG,
    Path,
    datetime,
    hashlib,
    json,
    mo,
    pd,
    profile_active_nsys,
    profile_capture_mode,
    profile_execution_ledger,
    profile_models_dir,
    profile_saved_report,
    profile_saved_report_dir,
    profile_subnormal_weights,
    profile_tool_probe_mode,
    profile_trace_reports_root,
    timezone,
):
    # Informe reproducible: solo consume resultados terminados, nunca dispara inferencias.
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    mo.stop("isolated_comparison" not in profile_execution_ledger,mo.md("Informe final: esperando comparación aislada."))
    profile_final_directory=profile_trace_reports_root/("final_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    profile_final_directory.mkdir(parents=True,exist_ok=False)
    _profile_final_pairs=profile_execution_ledger["isolated_comparison"]
    assert set(_profile_final_pairs)=={"reference","ftz"}
    assert all(v["predictions_equivalent"] and v["status"]=="complete" for v in _profile_final_pairs.values())
    _profile_hashes={name:hashlib.sha256((profile_models_dir/name).read_bytes()).hexdigest() for name in profile_saved_report["artifacts_sha256"]}
    assert _profile_hashes==profile_saved_report["artifacts_sha256"],"Cambió un artefacto de referencia."
    _profile_traces=profile_execution_ledger["validated_traces"]
    assert all(_profile_traces[k]["validation"]["trace_validated"] and _profile_traces[k]["context"]["predictions_equivalent"] and profile_execution_ledger["trace_scopes"][k]["pid_matches"] for k in ("reference","ftz"))
    profile_final_comparison=pd.DataFrame([dict(scenario=k,**v) for k,v in _profile_final_pairs.items()])
    profile_final_comparison.to_csv(profile_final_directory/"unprofiled_comparison.csv",index=False)
    pd.DataFrame(profile_execution_ledger["trace_comparison"]).to_csv(profile_final_directory/"stage_comparison.csv",index=False)
    profile_subnormal_weights.to_csv(profile_final_directory/"subnormal_weights.csv",index=False)
    _profile_ratio=_profile_final_pairs["reference"]["median_ms"]/_profile_final_pairs["ftz"]["median_ms"]
    profile_final_analysis={
     "status":"complete","source":PROFILE_CONFIG["source"],"weights_modified":False,
     "artifacts_sha256":_profile_hashes,"artifacts_unchanged":True,
     "isolated_unprofiled":_profile_final_pairs,"median_ratio_reference_over_variant":_profile_ratio,
     "validated_traces":_profile_traces,"trace_scopes":profile_execution_ledger["trace_scopes"],
     "trace_stages":profile_execution_ledger["trace_comparison"],
     "ftz_thread_exploration":profile_execution_ledger["ftz_study"],
     "resident_confirmation":profile_execution_ledger["confirmation"],
     "subnormal_weights":{"count":int(profile_subnormal_weights.subnormal.sum()),"tensors":int((profile_subnormal_weights.subnormal>0).sum())},
     "historical_baseline":{"directory":str(profile_saved_report_dir),"median_ms":profile_saved_report["summary"]["median_ms"],"paired_with_new_variant":False},
     "cancelled_explorations":{"cpu_threads_20261031T230732_629807Z":"Incomplete: one slow warmup then time budget stop; not a thread matrix.","cpu_observable_20261031T231120_479390Z":"Cancelled cooperatively before next native call after denormal finding; original ONNX run method restored."},
     "excluded_latency_run":{"directory":"confirm_observable_20261031T232002_958436Z","reason":"Duplicated GPU model copies exceeded physical VRAM; equivalence only."},
     "limitations":["Two unique frozen-test images; 2 warmup + 4 timed calls per isolated scenario; no robust p95 or full accuracy validation.",
     "FTZ/DAZ, thread count and spinning change together; cannot isolate their individual effects.",
     "Historical 142.95 s median agrees approximately with fresh unprofiled reference; Nsight reference host frames were much shorter, so profiler timings are not nominal latency.",
     "Host NVTX intervals are asynchronous and nested; do not sum medians or call them GPU execution time.",
     "CPU thread exploration overlapped an unoptimized job; only final isolated runs are sequential.",
     "Parent notebook retained CPU-resident model objects during both children; GPU isolation, not exclusive hardware reservation.",
     "No continuous power/thermal control, no CPU instruction sampling; observed parent Windows power plan was Silent.",
     "FTZ may change numerical behavior outside tested cases; original production reference remains unchanged."],
     "sources":{"ort_ftz":"https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h",
     "nsight_manifest":"https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.0.json"}}
    (profile_final_directory/"analysis.json").write_text(json.dumps(profile_final_analysis,indent=2),encoding="utf-8")
    _profile_ref=_profile_final_pairs["reference"]["median_ms"]/1000
    _profile_ftz=_profile_final_pairs["ftz"]["median_ms"]/1000
    _profile_gui=Path(profile_active_nsys).parent.parent/"host-windows-x64"/"nsys-ui.exe"
    assert _profile_gui.is_file()
    _profile_report=f"""# Profiling de industrial_sam.py — cierre ejecutado

    Pipeline de referencia: industrial_sam.py §1 + §8; SAM3 y DINOv2 en CUDA,
    EfficientAD y regresión ONNX en CPU. Artefactos, pesos y umbral originales intactos
    (SHA-256 verificados en analysis.json). Sin entrenamiento ni recalibración.

    ## Comparación principal sin profiler
    Procesos nuevos ejecutados secuencialmente desde esta misma libreta, sin NVTX en
    los frames. Dos imágenes únicas del test congelado (una GOOD y una BAD), dos
    calentamientos y cuatro llamadas medidas por escenario.

    | Escenario | Mediana por imagen |
    |---|---:|
    | Referencia, ONNX CPU por defecto, sin FTZ | {_profile_ref:.3f} s |
    | Variante FTZ/DAZ, 4 hilos, secuencial, sin spinning | {_profile_ftz:.3f} s |

    Razón entre medianas: {_profile_ratio:.2f}×. Estadística descriptiva de esta muestra
    pequeña: no es p95 estable ni throughput de producción. Las predicciones y scores
    DINO/EffAD/fusión coinciden con la referencia, rtol=1e-4 y atol=1e-6. FTZ es una
    variante numérica explícita; no se ha aplicado al notebook de producción.

    ## Evidencia CUDA + NVTX
    Dos capturas reales validadas por contenido, PID, dos frames y 5,776 kernels CUDA
    por captura. Los 17 rangos inspect/* están presentes y las decisiones coinciden.

    | Mediana del rango host | Referencia | FTZ + 4 hilos |
    |---|---:|---:|
    | inspect/frame | 5.337 s | 2.296 s |
    | inspect/efficientad_cpu | 3.771 s | 0.783 s |
    | inspect/sam | 1.375 s | 1.356 s |

    Estos son intervalos de host bajo profiler, con esperas y etapas anidadas;
    no se suman ni representan tiempo puro de GPU. Durante EfficientAD no hubo kernels
    CUDA del pipeline solapados; no implica que otras aplicaciones de la GPU estuvieran
    inactivas.

    Referencia: [{Path(_profile_traces['reference']['directory']).name}](../{Path(_profile_traces['reference']['directory']).name}/capture.nsys-rep)
    Variante: [{Path(_profile_traces['ftz']['directory']).name}](../{Path(_profile_traces['ftz']['directory']).name}/capture.nsys-rep)

    ## Diagnóstico y trazabilidad
    EfficientAD contiene 749,314 valores subnormales repartidos entre 19 tensores.
    La matriz exploratoria FTZ de 1/2/4 hilos terminó; eligió 4 entre esas opciones,
    pero compartió recursos con una prueba lenta. La comparación principal de arriba
    se ejecutó después de terminar los demás trabajos.

    Las exploraciones CPU sin FTZ que se detuvieron por presupuesto o cancelación
    se conservan como incompletas; no forman una matriz terminada ni un resultado de
    throughput. No se mantuvo el parche temporal de cancelación de ONNX.

    La mediana histórica interactiva de 142.95 s es coherente con la referencia
    aisladamente repetida sin profiler ({_profile_ref:.3f} s). Bajo Nsight, la referencia
    tuvo frames host mucho más cortos (5.337 s): el entorno de captura perturbó
    fuertemente el comportamiento temporal. No se ha aislado la causa de esa
    perturbación; las trazas sirven para inspeccionar actividad y etapas, no para
    estimar latencia nominal. La comparación principal usa exclusivamente los
    procesos sin profiler. FTZ, hilos y spinning cambian conjuntamente; no se
    identifican sus contribuciones individuales.

    La confirmación confirm_observable_20261031T232002_958436Z se excluye de latencias:
    acumuló copias de modelos que excedieron la VRAM física. Se corrigió el registro
    para reutilizar un único SAM y DINO; una confirmación residente posterior válida
    dio 5.038 s de mediana (4 mediciones). Los intentos fallidos se preservan para auditoría.

    ## Herramientas y reproducción
    Nsight Systems portátil 2025.1.3.140, descarga oficial NVIDIA con SHA-256
    8d61f266a8a1bc1ababd44de3ee38f7b85c8cdfcdaebace14c8f065d7624d0b3.
    Los intentos con 2023.4.4 fallaron; las capturas aceptadas usan 2025.1.
    No se instalaron drivers. Se capturó CUDA + NVTX sin muestreo de instrucciones CPU.
    Para revisar visualmente los archivos, abrir con la GUI de la misma versión:
    `{_profile_gui}` (File → Open y seleccionar capture.nsys-rep).

    La libreta incluye controles separados para serie completa, prueba CPU, captura
    y comparación aislada. Cada trabajo deja CSV/JSON/log; la serie larga está pausada
    por defecto. Los .nsys-rep y SQLite se mantienen localmente e ignoran en Git.

    ## Límites y adopción
    El kernel padre conservó objetos de modelos en CPU durante ambos procesos hijos;
    se aisló la VRAM, sin reservar el hardware de manera exclusiva.
    No se afirma precisión global, estabilidad térmica continua ni capacidad nominal.
    FTZ puede cambiar scores fuera de estas imágenes; adoptar la variante en producción
    requiere una validación del conjunto pertinente. Ese despliegue queda fuera del
    experimento de profiling y el pipeline original se conserva.

    Archivos: analysis.json, unprofiled_comparison.csv, stage_comparison.csv,
    subnormal_weights.csv; los directorios de cada corrida están en analysis.json.
    Documentación: [ORT FTZ/DAZ](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
    y [manifiesto oficial NVIDIA](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.0.json).
    """
    (profile_final_directory/"REPORT.md").write_text(_profile_report,encoding="utf-8")
    print("Informe final:",profile_final_directory)
    mo.output.replace(mo.vstack([mo.md(f"## Profiling completado\n\nReferencia **{_profile_ref:.3f} s** → FTZ + 4 hilos **{_profile_ftz:.3f} s**, medianas sin profiler. Razón **{_profile_ratio:.2f}×** en dos imágenes únicas; cuatro mediciones por escenario.\n\nDos trazas CUDA/NVTX validadas, scores equivalentes y artefactos intactos. Informe: `{profile_final_directory}/REPORT.md`"),mo.ui.table(profile_final_comparison[["scenario","median_ms","mean_ms","min_ms","max_ms","measurements","unique_images"]])]))

    return profile_final_analysis, profile_final_directory


@app.cell
def profile_finalize_kernel(
    json,
    mo,
    ort,
    profile_capture_jobs,
    profile_capture_mode,
    profile_device,
    profile_dino_model,
    profile_execution_ledger,
    profile_final_analysis,
    profile_final_directory,
    profile_isolated_jobs,
    profile_model_registry,
    profile_sam_model,
    profile_tool_probe_mode,
    torch,
):
    # Cierre: esperar resultados, restaurar las mismas instancias y cerrar recursos propios.
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    mo.stop("isolated_comparison" not in profile_execution_ledger)
    assert profile_final_analysis["status"]=="complete"
    assert ort.InferenceSession.run.__module__=="onnxruntime.capi.onnxruntime_inference_collection"
    assert profile_isolated_jobs["future"].done()
    assert profile_isolated_jobs["future"].exception() is None
    _profile_closed_logs=[]
    for _job_name,_job in [("captures",profile_capture_jobs),("debug",profile_execution_ledger["raw_debug"]),("probe",profile_execution_ledger["probe_job"])]:
        assert _job["process"].poll() is not None,"Todavía hay un proceso externo: "+_job_name
        if _job.get("log") and not _job["log"].closed:
            _job["log"].close()
            _profile_closed_logs.append(_job_name)
    # Registro único: no se llama a from_pretrained ni se crean copias.
    profile_model_registry["profile_sam_model"].to(profile_device)
    profile_model_registry["profile_dino_model"].to(profile_device)
    torch.cuda.synchronize(profile_device)
    torch.cuda.empty_cache()
    _profile_model_ids=[id(profile_model_registry[k]) for k in ("profile_sam_model","profile_dino_model")]
    assert id(profile_sam_model)==_profile_model_ids[0] and id(profile_dino_model)==_profile_model_ids[1]
    # Los pools quedan inactivos y reutilizables por sus botones; no quedan tareas pendientes.
    profile_final_kernel_state={"status":"ready","external_jobs_complete":True,"original_models_restored":"cuda","sam_device":str(next(profile_sam_model.parameters()).device),"dino_device":str(next(profile_dino_model.parameters()).device),"gpu_allocated_gb":torch.cuda.memory_allocated(profile_device)/1e9,"gpu_reserved_gb":torch.cuda.memory_reserved(profile_device)/1e9,"closed_logs":_profile_closed_logs,"reused_model_instances":True}
    assert profile_final_kernel_state["gpu_allocated_gb"]<5.5,"Revisar duplicaciones GPU antes de cerrar."
    (profile_final_directory/"kernel_final_state.json").write_text(json.dumps(profile_final_kernel_state,indent=2),encoding="utf-8")
    print(json.dumps(profile_final_kernel_state,indent=2))

    return


@app.cell
def profile_progress_control(
    mo,
    profile_capture_mode,
    profile_tool_probe_mode,
):
    # Observación sin bloquear el kernel ni iniciar inferencias adicionales.
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    profile_progress_refresh=mo.ui.run_button(label="Actualizar avance de comparación aislada")
    profile_progress_refresh

    return (profile_progress_refresh,)


@app.cell
def profile_progress_status(
    Path,
    json,
    mo,
    pd,
    profile_capture_mode,
    profile_isolated_jobs,
    profile_progress_refresh,
    profile_tool_probe_mode,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    _profile_refresh=profile_progress_refresh.value
    _profile_progress=profile_isolated_jobs.get("progress",{"status":"not_started"})
    if _profile_progress["status"]=="complete":
        print("Comparación aislada completada: referencia y FTZ + 4 hilos. No hay inferencias pendientes.")
        mo.output.append(mo.ui.table(pd.DataFrame([{"scenario":key,"median_ms":value["median_ms"],"measurements":value["measurements"],"predictions_equivalent":value["predictions_equivalent"]} for key,value in _profile_progress["results"].items()])))
    else:
        print(json.dumps(_profile_progress,indent=2))
        if "directory" in _profile_progress:
            _progress_dir=Path(_profile_progress["directory"])
            if (_progress_dir/"isolated_timings.csv").exists():
                mo.output.append(mo.ui.table(pd.read_csv(_progress_dir/"isolated_timings.csv")))
            else:
                print("Inicializando modelos o calculando la primera imagen; se guarda después de cada inferencia.")
    if profile_isolated_jobs.get("future") and profile_isolated_jobs["future"].done() and profile_isolated_jobs["future"].exception():
        print("ERROR:",repr(profile_isolated_jobs["future"].exception()))

    return


@app.cell
def profile_completed_results(
    json,
    mo,
    profile_capture_mode,
    profile_root,
    profile_tool_probe_mode,
):
    # Resultados persistidos: visibles también después de reiniciar el kernel.
    mo.stop(profile_capture_mode or profile_tool_probe_mode)
    _profile_final_files=sorted((profile_root/"reports"/"profiling"/"transistor_binary").glob("final_*/analysis.json"))
    if _profile_final_files:
        _latest_final=_profile_final_files[-1]
        _latest_analysis=json.loads(_latest_final.read_text(encoding="utf-8"))
        print("Informe persistido:",_latest_final.parent/"REPORT.md")
        mo.output.append(mo.md((_latest_final.parent/"REPORT.md").read_text(encoding="utf-8")))
    else:
        print("La comparación aislada está en curso. El informe persistido aparecerá al cerrar sus dos escenarios.")

    return


@app.cell
def profile_release_gpu_control(mo):
    # Este botón libera VRAM sin cerrar la libreta ni perder los resultados.
    profile_release_gpu_button=mo.ui.run_button(label="Liberar memoria GPU del profiling")
    profile_release_gpu_button

    return (profile_release_gpu_button,)


@app.cell
def profile_release_gpu_action(
    json,
    mo,
    profile_capture_jobs,
    profile_capture_mode,
    profile_confirm_jobs,
    profile_device,
    profile_ftz_jobs,
    profile_isolated_jobs,
    profile_model_registry,
    profile_release_gpu_button,
    profile_thread_jobs,
    profile_tool_probe_mode,
    torch,
):
    mo.stop(profile_capture_mode or profile_tool_probe_mode or not profile_release_gpu_button.value)
    # Conserva los modelos en RAM; para inferir de nuevo, restaurar estas instancias en CUDA.
    import gc as profile_release_gc
    assert not any(job.get("future") and not job["future"].done() for job in (profile_thread_jobs,profile_ftz_jobs,profile_confirm_jobs,profile_isolated_jobs)),"Hay un benchmark en curso."
    assert not profile_capture_jobs.get("process") or profile_capture_jobs["process"].poll() is not None,"Hay una captura en curso."
    torch.cuda.synchronize(profile_device)
    profile_gpu_release_result={"before_allocated_gb":torch.cuda.memory_allocated()/1e9,"before_reserved_gb":torch.cuda.memory_reserved()/1e9}
    for _key in ("profile_sam_model","profile_dino_model"):
        profile_model_registry[_key].to("cpu")
    profile_release_gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(profile_device)
    profile_gpu_release_result.update(after_allocated_gb=torch.cuda.memory_allocated()/1e9,after_reserved_gb=torch.cuda.memory_reserved()/1e9,models_device="cpu")
    print(json.dumps(profile_gpu_release_result,indent=2))
    print("Modelos fuera de GPU. Los resultados guardados se conservan. Restaurar los modelos a CUDA antes de otra inferencia.")

    return


if __name__ == "__main__":
    app.run()
