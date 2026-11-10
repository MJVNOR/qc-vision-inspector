# Profiling de industrial_sam.py — cierre ejecutado

Pipeline de referencia: industrial_sam.py §1 + §8; SAM3 y DINOv2 en CUDA,
EfficientAD y regresión ONNX en CPU. Artefactos, pesos y umbral originales intactos
(SHA-256 verificados en analysis.json). Sin entrenamiento ni recalibración.

## Comparación principal sin profiler
Procesos nuevos ejecutados secuencialmente desde esta misma libreta, sin NVTX en
los frames. Dos imágenes únicas del test congelado (una GOOD y una BAD), dos
calentamientos y cuatro llamadas medidas por escenario.

| Escenario | Mediana por imagen |
|---|---:|
| Referencia, ONNX CPU por defecto, sin FTZ | 143.092 s |
| Variante FTZ/DAZ, 4 hilos, secuencial, sin spinning | 4.338 s |

Razón entre medianas: 32.99×. Estadística descriptiva de esta muestra
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

Referencia: [nsight_reference_20261031T235032_944311Z](../nsight_reference_20261031T235032_944311Z/capture.nsys-rep)
Variante: [nsight_ftz_20261031T234733_579588Z](../nsight_ftz_20261031T234733_579588Z/capture.nsys-rep)

## Diagnóstico y trazabilidad
EfficientAD contiene 749,314 valores subnormales repartidos entre 19 tensores.
La matriz exploratoria FTZ de 1/2/4 hilos terminó; eligió 4 entre esas opciones,
pero compartió recursos con una prueba lenta. La comparación principal de arriba
se ejecutó después de terminar los demás trabajos.

Las exploraciones CPU sin FTZ que se detuvieron por presupuesto o cancelación
se conservan como incompletas; no forman una matriz terminada ni un resultado de
throughput. No se mantuvo el parche temporal de cancelación de ONNX.

La mediana histórica interactiva de 142.95 s es coherente con la referencia
aisladamente repetida sin profiler (143.092 s). Bajo Nsight, la referencia
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
`C:\Users\marti\.cache\qc-vision-tools\nsight-systems-2025.1.3.140\extracted\nsight_systems-windows-x86_64-2025.1.3.140-archive\nsight-systems\2025.1.3\host-windows-x64\nsys-ui.exe` (File → Open y seleccionar capture.nsys-rep).

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
