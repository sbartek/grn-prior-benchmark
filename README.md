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

Primary dataset pipeline (each script writes to `data/`, gitignored):

```bash
uv run python scripts/00_fetch_data.py            # fetch + cache the CELLxGENE RA PBMC subset
uv run python scripts/01_build_pseudobulk.py      # pseudobulk (donor x cell_type)
uv run python scripts/02_build_graph.py           # DoRothEA adjacency + graph controls
uv run python scripts/11_final.py                 # DEFINITIVE sweep -> results/tables/final.csv
uv run python scripts/04_make_figures.py          # all figures -> results/figures/
```

Supplementary sweeps (optional, reproduce specific figures/tables):

```bash
uv run python scripts/05_density_ablation.py      # DoRothEA confidence A/AB/ABC + rewired-A
uv run python scripts/10_bottleneck.py            # bottleneck-dim sensitivity (z=32/64/128)
```

Second dataset (external validity — COVID PBMC; re-run 00/01/02 with the covid args first):

```bash
uv run python scripts/00_fetch_data.py 2a498ace-872a-4935-984b-1afa70fd9886 data/covid_raw.h5ad
uv run python scripts/01_build_pseudobulk.py data/covid_raw.h5ad data/covid_pseudobulk.h5ad
uv run python scripts/02_build_graph.py data/covid_pseudobulk.h5ad data/covid_graph.npz
uv run python scripts/08_second_dataset.py        # cell type + 3-class disease -> results/tables/covid.csv
```

> Note: `dc.op.dorothea/collectri` are fetched live at runtime (not version-pinned); the CELLxGENE
> census is pinned to `2025-11-08`. Numbers are near-deterministic but not bit-reproducible (MPS).

## Notebook
`notebooks/01_eda_suitability.ipynb` — narrative EDA + dataset-suitability check (composition,
balance, library sizes, PCA/UMAP showing cell type separates but disease/donor don't, DoRothEA
graph stats, headline result). **Interactive Plotly (white theme).** Committed with outputs;
re-run with (needs `plotly`, `umap-learn`, `jupyterlab` in the env):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_suitability.ipynb
```

For reviewing without running, open the self-contained
[`notebooks/01_eda_suitability.html`](notebooks/01_eda_suitability.html) (Plotly.js embedded,
renders offline in any browser).

`notebooks/02_toy_example.ipynb` — a tiny illustrative example (12 genes, 3 TFs, 3 cell types)
comparing **graph-aware vs not-graph-aware** representations as noise grows: shows *why* a GRN
prior denoises (aggregating co-regulated genes) and *why the right graph matters* (a rewired graph
destroys the signal). Standalone view: [`02_toy_example.html`](notebooks/02_toy_example.html).

## Layout
`src/grn_bench/` core lib · `scripts/` pipeline · `notebooks/` EDA · `results/` figures+tables ·
`memo/` write-up. Full design rationale in `PLAN.md`; findings in `memo/memo.md`.

## Status
Complete. Runnable pipeline + baseline/GRN encoders + TF-activity arms, full sweeps, EDA
notebook, and a ~2-page memo (`memo/memo.md`). Post-submission critical review and fixes logged in
`PLAN.md` §14; literature context in `references.md`.
