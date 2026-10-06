"""Execute notebooks/industrial_master.ipynb with repo root as kernel CWD."""

import nbformat
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks" / "industrial_master.ipynb"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    timeout=-1,
    kernel_name="betterclasificator",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbformat.write(nb, NB)
print("executed ok")
