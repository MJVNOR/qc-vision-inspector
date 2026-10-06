.PHONY: sync notebooks mlflow predict train-prod em-prod

sync:
	uv sync

notebooks:
	uv run marimo edit --no-token --no-skew-protection --port 2718 --host localhost --mcp code-mode --mcp-allow-remote notebooks/eda.py notebooks/fase_a.py notebooks/fase_b.py notebooks/fase_c.py notebooks/fase_d.py notebooks/fase_e.py notebooks/fase_f.py notebooks/fase_g.py notebooks/fase_h.py

mlflow:
	uv run mlflow ui

train-prod:
	uv run python -m betterclasificator.modeling.train --logreg --bank

em-prod:
	uv run python -m betterclasificator.modeling.train --em

predict:
	uv run python -m betterclasificator.modeling.predict $(IMAGES)
