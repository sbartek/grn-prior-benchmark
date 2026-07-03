# PLAN — GRN-Prior Expression Embedding Benchmark (Tolemy Bio take-home)

**Role:** Founding ML Engineer (Tolemy Bio, biotech, Barcelona — Ian Phipps lead)
**Window:** starts Fri 2026-07-03 · **due Sun 2026-07-06 12:00 CET**
**Submit:** GitHub repo or zip, by email to Caelan, with run notes.

> They explicitly do **not** expect the full 48h. Scope tightly. A clean, well-argued
> negative result beats an impressive positive one built on an unfair comparison.

---

## 1. The question (restated in my words)

Does a **graph-aware encoder** that uses **DoRothEA TF→target regulatory edges** produce
**better pseudobulk expression embeddings** than a matched **non-graph baseline** —
*especially* under **low-data, noisy, and graph-corrupted** conditions?

"Better" = the embedding captures **useful biological state**, not that it reconstructs
expression well. So evaluation is on *downstream biological readouts*, not reconstruction MSE.

**The real crux** (what they will probe in the follow-up): if a *corrupted/rewired* graph
helps just as much as the *real* DoRothEA graph, then any "benefit" is just **structured
sparsity / regularization**, not biology. The whole design must be able to tell those apart.

## 2. Key decisions (mine to make and defend)

| Decision | Choice | Why |
|---|---|---|
| Data granularity | **Pseudobulk** (donor × cell_type), per constraints | Required; also stabilizes noise, small N |
| Primary dataset | RA PBMC `d18736c3-6292-4379-919a-d6d973204c87` | Balanced 18 RA / 18 healthy, single assay → no assay–disease confound |
| Biological state to predict | **Cell type** (strong signal, within-donor) + **disease** (RA vs healthy, weak/confounded) | Cell type = sanity ceiling; disease = the hard, interesting readout |
| Split | **Group split by DONOR** (held-out donors) | The only way to test genuine generalization; disease is between-subjects so donor leakage would inflate everything |
| Baseline encoder | PCA + **capacity-matched MLP autoencoder** (dense) | PCA = honest floor; MLP AE = fair same-family baseline |
| GRN encoder | **Masked MLP** (weights restricted to DoRothEA TF–target adjacency) and/or a small **GNN** | Masked-MLP isolates *graph structure* holding capacity ~constant; GNN is the richer variant if time |
| Graph controls | degree-preserving **rewire**, **sign-shuffle**, **confidence A/AB/ABC** density, **fully random** | This is the core rigor: does the *specific biology* matter, or just sparsity? |
| Stress axes | **low-data** (subsample donors), **noise** (count downsampling / dropout / Gaussian) | Where a prior is theorized to help most |

## 3. Confounds to control (the part they grade hardest)

- **Donor = disease** (between-subjects). A "disease classifier" can just be a donor
  classifier. Mitigate: group-split by donor, report disease results as *suggestive only*,
  and lean on **cell type** as the primary trustworthy readout.
- **Batch/donor signal vs biology.** Probe embeddings for **donor predictability** and
  batch mixing (kNN donor-accuracy, silhouette). Good embedding = high cell-type signal,
  low donor signal.
- **Capacity confound.** GRN model must not win just by having more/fewer params. Match
  parameter count and training budget; report both.
- **Gene-set leakage.** DoRothEA restricts to ~9.3k genes / 429 TFs. The baseline must see
  the **same gene set** — otherwise the graph model wins by feature selection, not structure.

## 4. Evaluation protocol

- **Probes** on frozen embeddings: linear (logreg) + kNN. Frozen = tests the *embedding*, not a
  new supervised head.
- **Metrics:** macro-F1 / accuracy for cell type & disease; donor-prediction accuracy (lower
  better); optional scIB-style batch/bio metrics if time.
- **CV:** grouped k-fold over donors; report mean ± std across folds/seeds. Multiple seeds
  because N is small and variance will be large.
- **Primary comparison curves:** metric vs (a) # training donors, (b) noise level, (c) graph
  corruption level. The *story* lives in these curves, not a single number.

**Falsifiable hypotheses:**
- H1: GRN ≈ baseline at full data; GRN > baseline under low-data/noise (prior as regularizer).
- H2: Real graph > rewired graph — *if this fails, the prior is just sparsity.* (Most important test.)

## 5. Scope — what I will deliberately NOT do (state in memo)

- No raw single-cell modeling (pseudobulk only — per constraints).
- No STATE/State integration, no metabolic/GEM edges, no curated pathway sets (forbidden).
- No large hyperparameter search — fixed sensible configs, note it.
- Backup datasets only if the primary proves unsuitable or time permits a validation pass.
- No attempt at SOTA; a fair small comparison is the deliverable.

## 6. Deliverables (priority order)

1. **Runnable repo** + README (setup + one-command reproduce). *[essential]*
2. **Baseline encoder + GRN-aware encoder**, each → pseudobulk embedding. *[essential]*
3. **~2-page technical memo** interpreting results, incl. suitability, confounds, limitations. *[essential]*
4. Preprocessing + DoRothEA graph-construction scripts. *[if time]*
5. Ablation table + a few key figures (low-data, noise, corruption curves). *[if time]*

Memo must answer: data/task & why · dataset suitable? · pseudobulk + graph build · what was
compared · did GRN help & where · graph-control results · biology vs batch/donor · limitations ·
what would make it biologically stronger.

## 7. Timeline (target ~1.5 focused days)

- **Block A (setup + data):** uv env (Py 3.11), pull census subset, build pseudobulk, EDA + suitability check.
- **Block B (graph + models):** DoRothEA via decoupler, adjacency + controls, baseline AE, masked-MLP GRN encoder.
- **Block C (eval + stress):** probe harness, donor-split CV, low-data + noise + corruption sweeps.
- **Block D (write-up):** ablation table, 3–4 figures, memo, README, repo cleanup, submit.

Hard stop for writing: leave the **last ~4h for the memo** regardless of experiment state.
The memo is graded above coverage.

## 8. Environment / tech risks

- Local Python is **3.14** (homebrew) — scanpy/torch/torch_geometric lack 3.14 wheels.
  → Use **`uv` with pinned Python 3.11** in a project venv. (uv 0.9.26 available.)
- `cellxgene_census` download can be large; cache the fetched subset to `data/` (gitignored).
- torch_geometric adds install friction on Mac → prefer **masked-MLP** as primary GRN model
  (pure torch), treat GNN as optional stretch.

## 9. Stack

`uv` · Python 3.11 · scanpy/anndata · cellxgene_census · decoupler · numpy/pandas/scipy ·
scikit-learn (probes/PCA) · torch (encoders) · matplotlib. torch_geometric optional.

## 10. Repo layout

```
src/grn_bench/   data.py · pseudobulk.py · graph.py · models.py · eval.py · experiments.py
scripts/         00_fetch_data.py · 01_build_pseudobulk.py · 02_build_graph.py · 03_run_experiments.py
notebooks/       01_eda_suitability.ipynb
docs/            index.md · concepts.md · methods.md · results.md  (MkDocs Material → GitHub Pages)
results/         figures/ · tables/
memo/            memo.md (→ 2-page PDF)
data/            (gitignored cache)
```

## 11. Execution steps

- **Step -1 — Repo + documentation.** Private GitHub repo `sbartek/grn-prior-benchmark`
  (SSH remote). Human-readable docs as **MkDocs Material** in `docs/`, deployed to **GitHub
  Pages** via `.github/workflows/docs.yml`. All docs are plain markdown so they also render
  natively on github.com (fallback if Pages is unavailable on a private repo — see below).
- **Step 0 — Environment.** ✅ `uv venv --python 3.11` (CPython 3.11.14). Installed + smoke-tested:
  scanpy 1.11.5, anndata 0.12.19, torch 2.12.1 (**MPS available**), decoupler 2.1.6
  (`dc.op.dorothea` present), cellxgene-census 1.18.0, scikit-learn 1.9.0. Pins in
  `requirements.txt`; exact freeze in `requirements.lock.txt`.
- **Step 1 — Data.** ✅ Fetched RA PBMC subset → `data/raw.h5ad` (108,717 cells × 61,497 genes,
  302 MB, **raw integer counts**). Suitability confirmed: disease balanced (18 RA / 18 normal),
  **no sex confound** (12F/6M in both arms), **single assay** (10x 3′ v3), 15 cell types present
  (rare: CD4 595, γδ-T 1424 cells). Main confound = donor (between-subjects), as expected.
- **Step 2 — Pseudobulk.** ✅ Summed raw counts per (donor × cell_type) via a sparse indicator
  matmul → `data/pseudobulk.h5ad`: **500 samples × 21,572 genes** (536 groups → 500 kept at
  ≥10 cells; median 104 cells/group). Raw counts kept in `layers["counts"]`; X = CP10K + log1p.
  Gene symbols stored in `var["gene_symbol"]` for DoRothEA matching.
- **Step 3 — Graph + controls.** ✅ `data/graph.npz`. Each hidden unit = a TF aggregating its
  regulon (+ self-loop) → mask is the gene×TF adjacency. **8,376 input genes × 411 TFs**,
  30,609 real edges (0.89% density). Controls at matched density: rewired 28,841 (degree-
  preserving target-permutation; ~6% dropped as dup collisions — note), sign_shuffled 30,609,
  random 30,486. Density ablations: A 5,664, AB 14,312. Baseline + all controls share this gene set.
- **Step 4 — Encoders.** ✅ `models.py` + `data.py`. Shared AE (gene→hidden→z=64→…→gene); the
  **only** structural difference is the first encoder layer: dense (baseline) vs `MaskedLinear`
  (GRN: effective weight = mask·sign·softplus(raw), each hidden unit = a signed TF regulon).
  Corrupted graphs feed the same class → controls at identical nominal params (6.95M; GRN's
  *effective* free params = #edges). Objective = MSE reconstruction, no labels. Smoke test:
  both train on MPS, finite 64-d embeddings; baseline recon val_mse 0.052 < GRN 0.056 (expected).
- **Step 5 — Eval harness.** ✅ `eval.py` (donor-grouped probes, donor-leakage) + `experiments.py`
  (nested donor-CV: encoder retrained per fold on train donors, then frozen probe).
- **Step 6 — Experiments.** ✅ `03_run_experiments.py` → `results/tables/*.csv`, figures via
  `04_make_figures.py`. **NEGATIVE result:** GRN prior trails PCA + baseline on cell type at full/
  low-data/noise; real graph does NOT beat same-density controls at full data (random 0.731 >
  real 0.699); faint real>rewired signal under stress but never beats baseline; disease not
  decodable from held-out donors (donor confound); prior doesn't reduce donor leakage.
- **Step 7 — Suitability.** Report donor≈disease confound; lean on cell type as trustworthy readout.
- **Step 8 — Write-up.** ✅ Memo (`memo/memo.md`) + 4 figures + **executed EDA notebook**
  (`notebooks/01_eda_suitability.ipynb`: composition/balance, PCA+UMAP confound visual, graph
  stats, headline). Ablation table = `results/tables/results.csv`.
- **Step 9 — Package + submit.** README one-command reproduce; push; email repo link to Caelan.

**Critical path if short on time:** -1 → 0 → 1 → 2 → 3 → 4 → 5 → (Step 6 corruption test only) → 8.

### GitHub Pages — status
Repo made **public** 2026-07-03; Pages enabled (source = GitHub Actions). **Live site:
https://sbartek.github.io/grn-prior-benchmark/** (verified HTTP 200). Docs also render as
markdown on github.com, and `mkdocs serve` works locally.
*(Historical caveat: private repos need GitHub Pro/Team for Pages — moot now that it's public.)*

## 12. Living-documentation protocol (update after EVERY step)

After completing each execution step above, before moving on:
1. **PLAN.md** — tick the step / note what actually happened, deviations, new decisions.
2. **Docs** — update the relevant `docs/*.md` (methods/results/concepts) + `memo/memo.md`.
3. **Memory** — update `project_grn_benchmark.md` with durable status/decisions.
4. **Commit** — one focused commit per step (`Step N: …`), push to the remote.

Definition of done for a step = code works **and** all four of the above are updated.
