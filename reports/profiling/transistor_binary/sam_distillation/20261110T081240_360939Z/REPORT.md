# Destilación de SAM 3.1 — resultados medidos

Campaña: 20261110T081240_360939Z
Dataset fingerprint: c1743b64b515a6eeb6ec1605f1d384beb1b76d975188f8ed9bbf3a11a3a0dfc5
Teacher checkpoint SHA-256: 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
Todos los targets son pseudomáscaras SAM 3.1; el acuerdo no equivale a exactitud humana.
Train/val se separaron dentro del train congelado. El test no participa en la selección.

| Modelo | Backend | Entrada | IoU teacher | Dice | Borde F1 | Red CUDA mediana ms | Segmentación mediana ms | Segmentación p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26s-seg | pytorch_fp32 | 640 | 0.9706 | 0.9851 | 0.8723 | 36.700 | 50.585 | 65.208 |
| efficientvit-sam-l0 | pytorch_fp32 | 512 | 0.9618 | 0.9805 | 0.7628 | 49.469 | 52.889 | 66.504 |
| yolo26m-seg | pytorch_fp32 | 640 | 0.9698 | 0.9846 | 0.8659 | 41.272 | 53.204 | 66.630 |
| yolo26n-seg | pytorch_fp32 | 640 | 0.9671 | 0.9833 | 0.8368 | 35.273 | 51.399 | 68.400 |
| efficientvit-sam-l1 | pytorch_fp32 | 512 | 0.9548 | 0.9768 | 0.6901 | 57.459 | 62.162 | 75.077 |
| yolo26n-seg | tensorrt_fp16 | 640 | 0.9670 | 0.9832 | 0.8342 | 8.060 | 24.225 | 36.013 |
| yolo26s-seg | tensorrt_fp16 | 640 | 0.9706 | 0.9851 | 0.8718 | 13.567 | 29.527 | 38.033 |
| yolo26m-seg | tensorrt_fp16 | 640 | 0.9698 | 0.9846 | 0.8680 | 17.366 | 31.383 | 48.283 |
| efficientvit-sam-l0 | tensorrt_fp16 INVALIDO (NaN/Inf) | 512 | — | — | — | — | — | — |
| efficientvit-sam-l1 | tensorrt_fp16 INVALIDO (NaN/Inf) | 512 | — | — | — | — | — | — |

## Interpretación
Red CUDA usa entrada residente y eventos CUDA; segmentación incluye H2D, preprocesamiento, predicción, posprocesamiento y máscara CPU.
Disk pipeline agrega lectura/decodificación; cámara y clasificación GOOD/BAD quedan fuera.
Memoria es el máximo de asignaciones PyTorch; TensorRT puede reservar memoria adicional.
Recetas de entrenamiento: YOLO AMP FP16; EfficientViT FP32 y lr=1e-5 por inestabilidad FP16/BF16. Los tiempos de entrenamiento corresponden a recetas diferentes.
Resoluciones: YOLO 640, EfficientViT 512. Se comparan configuraciones, no arquitecturas a idénticos píxeles.
Los backends FP32/FP16 se mantienen separados. Las trazas Nsight no estiman latencia nominal.
Entrenamiento usa 30 épocas salvo la configuración registrada; tiempos registrados incluyen setup y logging del tramo ejecutado. En nano reanudado, no incluyen el tramo anterior al checkpoint ni los intentos descartados.
Las métricas por GOOD/BAD, mediciones individuales, cantidad de imágenes únicas y runs están en all_results.csv.

## Selección
null

Si no hay selección, ningún candidato completado cumplió los requisitos fijados; no se inventa un ganador.
La selección es provisional hasta revisar la calidad real de las máscaras y su uso posterior.

## Test del candidato
Test no abierto: ningún candidato cumple los requisitos fijados en validación.

## Trazabilidad
Experimento MLflow: transistor-binary-sam-distillation
SAM liberó VRAM al terminar las pseudomáscaras: {"before_allocated_gb": 0.0, "after_allocated_gb": 0.01048576, "after_reserved_gb": 0.023068672, "teacher_retained": false}
Trazas: {
  "yolo26n-seg_pytorch_fp32": {
    "candidate": "yolo26n-seg",
    "backend": "pytorch_fp32",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26n-seg_pytorch_fp32_20261111T043333_865206Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 19228,
    "frames": 4,
    "cuda_kernel_count": 2164,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 0.9998772710337723,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "yolo26s-seg_pytorch_fp32": {
    "candidate": "yolo26s-seg",
    "backend": "pytorch_fp32",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26s-seg_pytorch_fp32_20261111T043430_355738Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 2288,
    "frames": 4,
    "cuda_kernel_count": 2272,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 0.9992328782046627,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "yolo26m-seg_pytorch_fp32": {
    "candidate": "yolo26m-seg",
    "backend": "pytorch_fp32",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26m-seg_pytorch_fp32_20261111T043458_058194Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 38668,
    "frames": 4,
    "cuda_kernel_count": 2432,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 0.9999608964489087,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "efficientvit-sam-l0_pytorch_fp32": {
    "candidate": "efficientvit-sam-l0",
    "backend": "pytorch_fp32",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\efficientvit-sam-l0_pytorch_fp32_20261111T043525_429330Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 30944,
    "frames": 4,
    "cuda_kernel_count": 1784,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 0.9997374355896681,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "efficientvit-sam-l1_pytorch_fp32": {
    "candidate": "efficientvit-sam-l1",
    "backend": "pytorch_fp32",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\efficientvit-sam-l1_pytorch_fp32_20261111T043601_606751Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 12824,
    "frames": 4,
    "cuda_kernel_count": 2192,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 0.9995496648558454,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "yolo26n-seg_tensorrt_fp16": {
    "candidate": "yolo26n-seg",
    "backend": "tensorrt_fp16",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26n-seg_tensorrt_fp16_20261111T055450_787078Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 15588,
    "frames": 4,
    "cuda_kernel_count": 312,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 1.0,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "yolo26s-seg_tensorrt_fp16": {
    "candidate": "yolo26s-seg",
    "backend": "tensorrt_fp16",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26s-seg_tensorrt_fp16_20261111T055520_542891Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 14308,
    "frames": 4,
    "cuda_kernel_count": 312,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 1.0,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "yolo26m-seg_tensorrt_fp16": {
    "candidate": "yolo26m-seg",
    "backend": "tensorrt_fp16",
    "directory": "C:\\Users\\marti\\Documents\\BetterClasificator\\reports\\profiling\\transistor_binary\\sam_distillation\\20261110T081240_360939Z\\yolo26m-seg_tensorrt_fp16_20261111T055549_138359Z",
    "returncode": 0,
    "validated": true,
    "kernel_pid": 38056,
    "frames": 4,
    "cuda_kernel_count": 312,
    "ranges": [
      "distill/capture",
      "distill/frame",
      "distill/h2d",
      "distill/load_decode",
      "distill/postprocess_d2h",
      "distill/preprocess",
      "distill/student_predict"
    ],
    "min_mask_iou_vs_unprofiled": 1.0,
    "note": "Host NVTX intervals can nest; not nominal latency."
  },
  "efficientvit-sam-l0_tensorrt_fp16": {
    "candidate": "efficientvit-sam-l0",
    "backend": "tensorrt_fp16",
    "skipped": true,
    "validated": false,
    "reason": "Backend numerically invalid; FP32 trace remains available."
  },
  "efficientvit-sam-l1_tensorrt_fp16": {
    "candidate": "efficientvit-sam-l1",
    "backend": "tensorrt_fp16",
    "skipped": true,
    "validated": false,
    "reason": "Backend numerically invalid; FP32 trace remains available."
  }
}
Fallos/limitaciones de ejecución: {}
Backends inválidos: {
  "efficientvit-sam-l0_tensorrt_fp16_val": "TensorRT FP16 produces nonfinite logits; latency is not a usable speed result.",
  "efficientvit-sam-l1_tensorrt_fp16_val": "TensorRT FP16 produces nonfinite logits; latency is not a usable speed result."
}
Los tiempos de backends con logits NaN/Inf no se consideran aceleraciones utilizables.
Los checkpoints de origen y dataset original no se modificaron.