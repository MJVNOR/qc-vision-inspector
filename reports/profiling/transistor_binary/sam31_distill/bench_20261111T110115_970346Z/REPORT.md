# Profiling de industrial_sam31_distill.py

Pipeline: EfficientViT-SAM-L1 FP32 destilado de SAM3.1 + DINO-masked + EffAD + fusion.
Dataset fingerprint `943ae352501916c0`. Segmentador `efficientvit-sam-l1`.

## Segmentador destilado (acuerdo con el teacher, no exactitud humana)
IoU 0.9490 / Dice 0.9715 / Borde F1 0.6872 / mascaras vacias 0.000
Red 22.6 ms (p95 29.4), segmentacion 24.0 ms (p95 30.9), teacher 1560 ms -> 65.0x (sin 1a foto SAM)

## Latencia sin profiler por camino de EffAD
| EffAD | n | mediana ms | p95 ms | fps secuencial |
|---|---:|---:|---:|---:|
| gpu | 30 | 119.3 | 134.2 | 8.47 |
| onnx_cpu_ftz | 8 | 4136.8 | 4378.6 | 0.26 |
| onnx_cpu | 4 | 2494.3 | 3352.0 | 0.37 |

Muestra pequena y balanceada del test congelado: estadistica descriptiva, no p95 estable ni
throughput de produccion. Incluye lectura, decodificacion, H2D/D2H y postproceso.

## Acuerdo entre caminos
```json
[
  {
    "camino": "gpu",
    "fotos_comparadas": 10,
    "max_delta_fused": 0.0,
    "max_delta_effad": 0.0,
    "decisiones_iguales": true
  },
  {
    "camino": "onnx_cpu_ftz",
    "fotos_comparadas": 8,
    "max_delta_fused": 0.00133551139517607,
    "max_delta_effad": 9.906291961669922e-05,
    "decisiones_iguales": true
  },
  {
    "camino": "onnx_cpu",
    "fotos_comparadas": 4,
    "max_delta_fused": 0.0008525737606988049,
    "max_delta_effad": 6.324052810668945e-05,
    "decisiones_iguales": true
  }
]
```

## Equivalencia de la instrumentacion
```json
[
  {
    "filename": "good_228.png",
    "label": "good",
    "equivalente": true,
    "bad": false,
    "max_delta": 0.0
  },
  {
    "filename": "bad_024.png",
    "label": "bad",
    "equivalente": true,
    "bad": true,
    "max_delta": 0.0
  }
]
```

## Traza Nsight validada
```json
{
  "trace": "reports\\profiling\\transistor_binary\\sam31_distill\\nsight_20261111T110229_952818Z\\capture.nsys-rep",
  "kernel_pid": 44532,
  "frames": 2,
  "cuda_kernels": 1794,
  "rangos": [
    "qc31d/capture",
    "qc31d/dino_d2h",
    "qc31d/dino_h2d",
    "qc31d/dino_preprocess",
    "qc31d/dinov2",
    "qc31d/effad_gpu",
    "qc31d/effad_preprocess",
    "qc31d/frame",
    "qc31d/fusion",
    "qc31d/h2d",
    "qc31d/load_decode",
    "qc31d/logreg",
    "qc31d/mask_crop",
    "qc31d/mask_restore",
    "qc31d/preprocess",
    "qc31d/student_predict"
  ],
  "scores_iguales_sin_profiler": true,
  "nota": "intervalos NVTX de host; la latencia nominal viene del §10"
}
```

Los rangos NVTX son intervalos de host y se anidan: no se suman ni equivalen a tiempo de GPU.
El entorno de captura perturba los tiempos; la latencia nominal es la del benchmark sin profiler.

## Subnormales en los pesos
EffAD: 729729 valores. Estudiante: 0 valores.
FTZ/DAZ solo afecta a la aritmetica CPU x86; el estudiante corre en GPU, donde los subnormales
no tienen penalizacion. Si alguna vez se mueve a ONNX-CPU, revisar este censo primero.

## Siguiente candidato (no medido aqui)
TF32 en el encoder del estudiante (`torch.backends.cuda.matmul.allow_tf32`). Baja la mantisa a
10 bits: adoptarlo solo si las mascaras y las decisiones siguen coincidiendo (IoU >= 0.999).