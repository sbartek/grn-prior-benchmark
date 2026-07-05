# Wiki — evaluation techniques

How we decide whether an embedding is *good*. The guiding principle (from the brief): **"better"
means the embedding captures biological state — not that it reconstructs expression**. So we never
score models by reconstruction error; we score whether **biology is recoverable** from the frozen
embedding.

Code: [`eval.py`](https://github.com/sbartek/grn-prior-benchmark/blob/main/src/grn_bench/eval.py) ·
[`experiments.py`](https://github.com/sbartek/grn-prior-benchmark/blob/main/src/grn_bench/experiments.py).
Everything below is demonstrated in the [evaluation notebook](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/03_evaluation.ipynb).

---

## 1. Pseudobulk — the unit of analysis
Single cells are extremely noisy (random dropout). We **average all cells of the same
`(donor × cell type)`** into one clean profile → ~500 pseudobulk samples. All modelling happens at
this level. (See the [toy notebook](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/02_toy_example.ipynb)
for a visual of noise washing out.)

## 2. Donor-grouped cross-validation (the generalization test)
We split by **whole donors**, never cells — a form of **grouped k-fold CV** (a.k.a.
leave-donors-out). The 36 donors are split into 5 folds; each fold's donors are held out once.

Inside every fold: **train the encoder on the training donors only → freeze it → fit the probe on
training-donor embeddings → score on the held-out donors.** The encoder is retrained per fold, so a
held-out donor is unseen by *both* the encoder and the probe — genuine generalization, no leakage.
Folds are **re-shuffled per seed** so error bars reflect the donor-split variance.

Why donors, not cells: a random cell split lets a model "recognize the donor" and memorize
batch/genetics rather than learn transferable biology.

## 3. The probe — the main metric
On the frozen embedding we train a simple **logistic-regression probe** to predict the label, and
measure **macro-F1**.

- **Macro-F1** = per-class F1 (harmonic mean of precision & recall), averaged with **equal weight
  across all 15 cell types**. Chosen over accuracy because the cell types are highly imbalanced —
  macro-F1 makes rare subtypes count as much as common ones.
- A **fixed 15-class label set** is used so every fold is scored on the same classes and the
  numbers are comparable.
- "Linear probe" logic: a *good* embedding makes biology **linearly readable** — if a simple linear
  model recovers cell type, the embedding captured it. (kNN is available as a non-linear check.)

## 4. Readouts (what we predict)
| readout | role |
|---|---|
| **cell type** (15 classes) | **primary** — the trustworthy signal |
| **disease** (RA 2-class / COVID 3-class) | **secondary, suggestive** — confounded with donor |
| **donor identity** | **negative control** — see §7 |

## 5. Stress conditions
Each comparison is run under several regimes to map *where* a prior helps:
- **full** — all training donors.
- **low-data** — subsample to k ∈ {4, 8, 16} training donors (where priors are theorized to help).
- **noise** — binomial **count-thinning** to fraction p ∈ {0.3, 0.1} (simulate lower sequencing
  depth), renormalized so p=1 recovers the clean input.

## 6. The controls — defending against "it's not really biology"
The core of the rigor. For any apparent benefit of the real graph, we rule out the boring
explanations with matched-shape nulls (see [Models](models.md)): **rewired** / **random** /
**sign-shuffled** graphs, a **matched-dimension random projection**, and a **dimension-matched**
TF-activity. A benefit only counts if the real graph beats these.

## 7. Donor leakage (biology vs batch)
A negative control: train a kNN to predict **donor identity** from the embedding. **Lower is
better** — a good embedding encodes biology, not which person the sample came from.

## 8. Bottleneck-dimension sensitivity
Sweep the embedding size z ∈ {32, 64, 128} to confirm the conclusions aren't an artifact of the
default z=64.

## 9. Unsupervised clustering (a stricter, complementary metric)
The probe is *supervised*. A label-free complement (the **scIB** standard) is to **cluster** the
embedding and compare the clusters to true cell types with:
- **ARI** (Adjusted Rand Index) — agreement of two partitions, corrected for chance.
- **NMI** (Normalized Mutual Information) — shared information between clusters and labels.

We use **KMeans**; note **DBSCAN is a poor fit** for single-cell embeddings (imbalanced,
variable-density, high-dimensional → it collapses to one blob or labels a third of points as
noise); the field uses graph-based **Leiden**. Key finding: **the metric flips the winner** — PCA
wins the probe but is *worst* by clustering; the GRN-informed representations cluster tightest.

## 10. Statistics
Reported as mean ± std over seeds; paired deltas vs baseline computed at matched (seed, fold).
Robustness to the training rule was checked by re-running with a **fixed epoch budget** instead of
early stopping (verdict unchanged, mean |ΔF1| = 0.004).

## What we deliberately do **not** use
- **Reconstruction MSE as a metric** — it's the autoencoders' *training* objective, not an
  evaluation (an AE can reconstruct perfectly while its embedding encodes junk). Not all models can
  even reconstruct (`dc_tfact` is one-way).
- **Same-label supervised embeddings** — training the encoder to predict cell type and then testing
  cell type would be circular and unfair to the unsupervised baselines.

Full numbers: [Results](results.md) · the models themselves: [Models](models.md).
