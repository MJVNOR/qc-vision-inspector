# BetterClasificator — binary good/bad classification of MVTec transistor
(CCDS layout: `data/`, `notebooks/`, `models/`, `reports/`, `betterclasificator/`)

Data-science repo (uv + marimo + torch CUDA + MLflow). Dataset is tiny and
imbalanced: `data/processed/transistor_binary/{good:273, bad:40}` + `labels.csv`
(filename,label,defect,source,original_path). Originals under
`data/raw/mvtec_anomaly_detection/` are never modified.

## Env (uv only)

- `uv sync` / `uv run ...`. Python pinned 3.12 (`.python-version`).
- torch CUDA comes from the explicit `pytorch-cu128` index + `[tool.uv.sources]`
  in `pyproject.toml`. Do NOT remove. `UV_TORCH_BACKEND` does nothing for
  `uv sync`. Verify: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → must show `+cu128 True`.
- After `uv add`, test-import in the marimo kernel before restarting the server;
  new packages are often importable without restart.

## Secrets

- `HF_TOKEN` lives in `.env` (gitignored). Never print it.
- PowerShell gotcha: `(Get-Content .env) -match '^HF_TOKEN='` on a one-line
  file returns a Boolean, not the line. Use:
  `((Get-Content .env | Select-String '^HF_TOKEN=').Line -replace '^HF_TOKEN=','').Trim()`
- No `<` stdin redirect in pwsh. Long jobs run with `background=true`.

## Marimo notebooks (`notebooks/eda.py`, `notebooks/fase_a.py`, …)

- Server (always from repo root — kernels resolve relative paths against CWD):
  `uv run marimo edit --no-token --no-skew-protection --port 2718 --host localhost --mcp code-mode --mcp-allow-remote notebooks/<notebook.py>`
  or `make notebooks` (all nine). (with `HF_TOKEN` set in env). `--no-skew-protection` is required for the
  OpenCode `marimo` MCP (`Missing server token` otherwise); `--mcp code-mode`
  mounts `/mcp/server` (off by default). Needs `uv add "marimo[mcp]"` for
  MCP SDK >=2.0. Kill stale server via
  `Get-NetTCPConnection -LocalPort 2718` → `Stop-Process`.
- NEVER edit notebook `.py` files directly while the server runs; drive the
  live kernel instead (bridge helpers live outside the repo).
  MANDATORY: all notebook reads/writes go through `marimo._code_mode`
  (the marimo MCP): no direct file edits, no shell heredocs into notebooks,
  no Python scripts that touch notebook state outside the kernel.
- Reach the live kernel ONLY via the marimo MCP through the `execute`
  Code Mode bridge (`search({query:"marimo"})`, then
  `tools.marimo.list_sessions()` / `tools.marimo.execute_code(...)`).
  NEVER use direct HTTP (`Invoke-RestMethod`/`fetch` against
  `/api/sessions`, `/api/kernel/execute`), shell scripts
  (`execute-code.sh`, `discover-servers.sh`), or `mx.py`-style bridge
  scripts / per-task driver files. User mandate: el MCP siempre.
- Program ALWAYS in the notebook: every Python step must live in a
  notebook cell (`cm.create/edit/run_cell`) so progress is visible in
  the UI. Scratchpad `execute_code` calls are for probing/status only,
  never for developing logic off-screen. User mandate: todo en la libreta.
- Sessions churn on browser reconnect: re-resolve the session on every call,
  target by `--file`/filename, never cache session IDs.
- `create_cell` applies on context exit: create and `run_cell` in SEPARATE
  calls. `run_cell` is async: poll cell `status` until no `running`/`queued`.
- NEVER read-edit the same cell twice in one context: the in-context view
  is stale, so the second `edit_cell` silently wipes the first
  (lost update). One edit per cell per context.
- Graph rules: `_private` names are cell-local (invisible elsewhere);
  top-level loop vars ARE definitions (never reuse e.g. `r` in two cells);
  imports count as definitions (no double `import X as Y`).
- Re-running cells cascades downstream, including `mlflow_log` cells
  (re-logs runs → delete duplicates, keep latest per run name).

## MLflow

- MLflow 3 rejects file store: use `sqlite:///mlflow.db` (stays at repo root;
  artifact URIs are relative to it, so never move it). Experiment
  `transistor-binary`. View with `uv run mlflow ui` (or `make mlflow`).
- `opencode.jsonc` wires `mlflow-mcp` to that sqlite URI. MANDATORY split:
  every MLflow read/query goes through mlflow-mcp via the `execute` Code
  Mode bridge (`search({namespace:"mlflow-mcp"})`, then
  `tools["mlflow-mcp"].list_runs(...)`), even when not listed natively —
  never re-derive via ad-hoc SDK queries when an MCP tool exists.
  Writes go through the SDK in notebook cells ONLY because the MCP has no
  `log_metric`/`log_artifact` tools.

## Models

- DINOv3 (`facebook/dinov3-vits16-*`) is gated: needs license acceptance +
  author approval + valid `HF_TOKEN`. 401 = bad/missing token, 403 = pending
  approval. Current fallback: `facebook/dinov2-base`
  (public). DINOv2 has no pooler: use CLS token `last_hidden_state[:, 0]`.
  transformers>=4.56 is required for DINOv3 support.
- `torch.load` defaults `weights_only=True` (torch≥2.6) and chokes on numpy
  payloads: own cache files (e.g. `models/fase_a_emb.pt`) load with
  `weights_only=False`.
