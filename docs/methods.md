# Methods & steps

The end-to-end pipeline. Full design rationale and fairness controls are in
[`PLAN.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/PLAN.md).

## Train on reconstruction, evaluate on biology

| | What | Role |
|---|---|---|
| **Training objective** | reconstruct the pseudobulk profile (MSE on log-norm, no labels) | how the embedding is *learned* |
| **Evaluation** | frozen probes for cell type / disease on held-out donors | how the embedding is *judged* |

Both encoders share **the same objective, loss, bottleneck size, training budget, and gene
set**. The *only* difference is architecture (dense vs graph-masked), so any advantage is
attributable to the graph.

## Steps

- **Step -1 — Repo + docs.** Private repo `sbartek/grn-prior-benchmark`; MkDocs Material docs →
  GitHub Pages; living-docs protocol (update plan/docs/memory/commit after every step).
- **Step 0 — Environment.** ✅ `uv venv --python 3.11` (local Python 3.14 lacks scanpy/torch
  wheels). Verified: scanpy 1.11.5, torch 2.12.1 (MPS/Apple-GPU available), decoupler 2.1.6,
  cellxgene-census 1.18.0. Exact pins in `requirements.lock.txt`.
- **Step 1 — Data.** Fetch CELLxGENE RA PBMC subset (`d18736c3-…`), cache `.h5ad` to `data/`.
- **Step 2 — Pseudobulk.** Aggregate (donor × cell_type), drop small groups (<~10 cells),
  normalize + log-transform.
- **Step 3 — Graph + controls.** DoRothEA via `decoupler`; adjacency on shared genes; build
  controls: degree-preserving **rewire**, **sign-shuffle**, **random**, confidence A/AB/ABC.
  Restrict pseudobulk to the shared gene set (baseline and GRN see identical features).
- **Step 4 — Encoders.** PCA + dense MLP autoencoder (baseline); graph-masked MLP autoencoder
  (GRN). GNN optional stretch (avoids torch_geometric friction on Mac).
- **Step 5 — Eval harness.** Frozen embedding → linear + kNN probes; grouped-by-donor CV;
  donor-predictability check; macro-F1 / accuracy, mean ± std over folds and seeds.
- **Step 6 — Experiments.** Full-data · low-data (subsample donors) · noise (count
  downsampling / dropout) · **graph corruption — the decisive real-vs-rewired test**.
- **Step 7 — Suitability.** RA vs healthy is between-subjects → disease ≈ donor. Cell type is
  the trustworthy readout; disease is reported as suggestive only.
- **Step 8 — Write-up.** Ablation table + 3–4 figures → ~2-page memo (reserve the last ~4h).
- **Step 9 — Package + submit.** One-command reproduce; push; email repo link.

**Critical path if short on time:** -1 → 0 → 1 → 2 → 3 → 4 → 5 → (Step 6 corruption test) → 8.

## Reproduce

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv run python scripts/00_fetch_data.py
uv run python scripts/01_build_pseudobulk.py
uv run python scripts/02_build_graph.py
uv run python scripts/03_run_experiments.py
```
