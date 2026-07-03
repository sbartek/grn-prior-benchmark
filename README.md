# GRN-Prior Expression Embedding Benchmark

Does a **DoRothEA GRN-prior graph-aware encoder** produce better **pseudobulk expression
embeddings** than a matched **non-graph baseline** — especially under low-data, noisy, and
graph-corrupted conditions? "Better" means the embedding captures **useful biological state**
(cell type / disease), not that it reconstructs expression.

Take-home for Tolemy Bio (Founding ML Engineer). Public data only. See `PLAN.md` for the full
approach and `memo/memo.md` for results and interpretation.

## Data
- **Expression:** CELLxGENE RA PBMC `d18736c3-6292-4379-919a-d6d973204c87`
  (~108k cells, 36 donors — 18 RA / 18 healthy, 15 immune cell types, single 10x 3′ v3 assay).
- **Prior:** DoRothEA TF→target network via `decoupler` (`dc.op.dorothea(organism="human")`).

## Setup

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

## Run

```bash
# 1. fetch + cache the CELLxGENE subset  (writes to data/, gitignored)
uv run python scripts/00_fetch_data.py
# 2. build pseudobulk (donor x cell_type)
uv run python scripts/01_build_pseudobulk.py
# 3. build DoRothEA adjacency + graph controls
uv run python scripts/02_build_graph.py
# 4. run baseline vs GRN encoders + stress sweeps -> results/
uv run python scripts/03_run_experiments.py
```

## Layout
`src/grn_bench/` core lib · `scripts/` pipeline · `notebooks/` EDA · `results/` figures+tables ·
`memo/` write-up. Full design rationale in `PLAN.md`.

## Status
Scaffold + plan. Implementation in progress — see `PLAN.md` §7 for the timeline.
