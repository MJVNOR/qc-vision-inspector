"""Central paths and constants. Import this instead of hardcoding relatives.

Marimo kernels run with CWD = server launch dir, so every path here is
anchored at the repo root and works no matter where notebooks live.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "transistor_binary"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim" / "transistor_norm"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

LABELS_CSV = DATA_PROCESSED / "labels.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "transistor-binary"

SEED = 42
N_SPLITS = 5
IMG_SIZE = 256

DINO_BACKBONE = "facebook/dinov2-base"
PATCH_LAYERS = ("layer2", "layer3")
