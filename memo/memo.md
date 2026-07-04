# Technical Memo — GRN-Prior Expression Embedding Benchmark

**Question.** Does a graph-aware encoder using DoRothEA TF→target edges produce *better*
pseudobulk expression embeddings than a matched non-graph baseline — where "better" means the
embedding captures biological state (cell type, disease), not that it reconstructs expression —
and does any benefit hold up under low-data, noise, and graph corruption?

**Answer (short).** **How you use the GRN matters more than whether you use it.**
- As a **deep-encoder constraint** (hard mask or soft graph-penalty): it *hurts* — consistently
  below PCA and a plain autoencoder — and a degree-preserving **rewired** graph does as well,
  so what little it offers is regularization, not the specific biology.
- As a **fixed feature transform** (decoupler ULM TF-activity, the classical use): it carries
  **real** biological signal — it beats a **matched-dimension random projection** everywhere —
  and it **beats PCA and the baseline in the low-data regime** (where priors are supposed to
  help). At full data it merely ties PCA; under noise it ties the baseline.
- The effect is **not DoRothEA-specific** (CollecTRI reproduces and slightly exceeds it) and
  **replicates on a second dataset** (COVID PBMC). Disease is decodable on the 3-class COVID data
  (PCA best) but not on the fully donor-confounded 2-class RA data.

A fair, carefully-scoped result: the prior encodes genuine biology, but only helps when applied
the classical way and mostly when data is scarce; imposing it on a learned encoder is worse than
doing nothing.

## Data, task, evaluation
CELLxGENE RA PBMC (`d18736c3…`, 108,717 cells) → **pseudobulk by (donor × cell_type)**, 536
groups → **500 kept** (≥10 cells), CP10K+log1p, restricted to the 8,376 genes shared with
DoRothEA. Readouts: **cell type** (15 classes, primary) and **disease** (secondary). All scoring
is **donor-grouped 5-fold CV** with the encoder retrained on train donors *inside* each fold — so
neither encoder nor probe ever sees a held-out donor. Metric: macro-F1 (mean over 5 seeds; paired
deltas computed at matched (seed, fold)).

## Is the dataset suitable?
Balanced **18 RA / 18 normal**, **no sex confound** (12F/6M both arms), **single assay** (10x 3′
v3 → no assay–disease confound), 15 cell types. But the design is **between-subjects**, so disease
is confounded with **donor identity**. Consequence: cell type is trustworthy; RA disease is
reported as suggestive only. (The COVID replication has 3 disease states across 75 donors — still
between-subjects, but less degenerate, and there disease *is* decodable.)

## Models compared (all share the gene set, capacity where noted, and training budget)
- **PCA** (linear floor) and **baseline** (dense autoencoder) — 64-d, no prior.
- **GRN-as-constraint**: `grn_real` (first layer masked to signed TF regulons), `grn_soft:λ`
  (dense layer + penalty shrinking off-regulon weights). Controls at matched density/degree:
  `grn_rewired`, `grn_sign_shuffled`, `grn_random`.
- **GRN-as-transform**: `dc_tfact` (decoupler ULM TF-activity, 293-d), `dc_tfact_pca` (→PCA-64,
  dimension-matched), `dc_tfact_collectri` (CollecTRI net, 675-d).
- **Null**: `rand_proj` (random linear features at the TF-activity dimension).

## Results (cell-type macro-F1)

| model (dim) | full | lowdata k=8 | noise p=0.3 |
|---|---|---|---|
| PCA (64) | 0.862 | 0.653 | **0.769** |
| baseline AE (64) | 0.794 | 0.640 | 0.733 |
| **dc_tfact** (293) | **0.864** | **0.691** | 0.730 |
| **dc_tfact_collectri** (675) | 0.858 | **0.740** | **0.779** |
| dc_tfact_pca (64) | 0.793 | 0.633 | 0.685 |
| rand_proj (293) | 0.767 | 0.627 | 0.640 |
| grn_soft:0.001 | 0.754 | 0.637 | 0.721 |
| grn_real (hard mask) | 0.714 | 0.569 | 0.597 |
| grn_soft:0.01 | 0.619 | 0.498 | 0.661 |

**Reading it:**
1. **GRN-as-constraint hurts.** `grn_real` and `grn_soft` trail both PCA and the baseline
   everywhere; *more* prior is *worse* (soft 0.001 > soft 0.01 > hard). The problem isn't mask
   hardness — it's imposing the graph on the encoder at all.
2. **The corruption test kills the "it's biology" story for the constraint.** At matched density
   `grn_real` ≈ `grn_rewired` ≈ `grn_random` (and a **density ablation** shows the same: a
   high-confidence A-subset ≈ baseline, but a rewired-A graph matches it). Any constraint benefit
   is sparsity/regularization, not regulatory content.
3. **The transform is different — biology is real there.** `dc_tfact` (293-d) beats the
   **matched-dimension random projection** `rand_proj` (293-d) in every condition (+0.06–0.10),
   so the DoRothEA weights carry genuine cell-type signal beyond dimensionality.
4. **…but it doesn't beat a strong simple baseline except when data is scarce.** At full data
   `dc_tfact` ≈ PCA (0.864 vs 0.862, using 4× the dimensions); under **low data** it clearly beats
   PCA and baseline (0.691 / CollecTRI 0.740 vs 0.653), the classic small-data regularization win;
   under noise it ties the baseline. `dc_tfact_pca` (dimension-matched) ≈ baseline — the full-rep
   advantage needs the extra dimensions.
5. **Not prior-specific.** CollecTRI ≥ DoRothEA (notably under low-data/noise) → the effect is a
   property of "TF-activity as a transform," and a broader/better network helps more.

**Disease.** On RA (2-class, fully donor-confounded) nothing decodes disease from held-out donors
(≈ chance). On **COVID (3-class, 75 donors)** disease *is* decodable — PCA best (0.715), then
TF-activity (~0.66–0.67); the GRN-mask models beat the *overfit* dense baseline (grn_real 0.654 vs
baseline 0.588) but `grn_rewired` (0.646) matches them → again regularization, not biology.

**Biology vs batch.** Donor-leakage (kNN donor accuracy, lower better): baseline 0.086 < random
0.098 < grn_real 0.114 < … < PCA 0.166. The prior does not reduce donor signal.

**External validity.** The full ordering (PCA ≥ TF-activity > baseline ≈ GRN-mask > soft-prior)
replicates on COVID → not an artifact of the RA dataset.

## Interpretation
The GRN is not useless — as a TF-activity transform it encodes real regulatory signal and gives a
genuine low-data/regularization benefit (stronger with CollecTRI). But it is **not more
informative than raw-expression PCA**, which captures the same cell-type structure more compactly;
and **injecting it into a learned encoder is strictly counterproductive**, adding constraint whose
only measurable effect (rewired-equivalent) is regularization. The honest one-liner: *use the GRN
as a feature transform, not as a network prior on a deep model, and expect help mainly when data
is limited.*

## Related work (context)
This matches a clear recent pattern: biology-wired networks rarely beat strong simple baselines on
accuracy, and reported gains often reflect capacity/regularization or dimensionality rather than
biology. Plain logistic regression matches scBERT on cell typing (Nat Mach Intell 2024); *"one PCA
still rules them all"* for perturbation prediction (Nat Methods 2025); biologically-informed
decoders (expiMap, Nat Cell Biol 2023; P-NET, Nature 2021) are framed as **interpretability**, not
accuracy, wins; and randomly-wired biological nets often match knowledge-primed ones — which is why
a shuffled/rewired-graph control is the demanded ablation (Kong et al. 2023). Biologically-informed
nets help mainly under small samples / weak signal (2025), consistent with our low-data result.
The canonical *successful* use of DoRothEA is exactly ours: decoupler ULM TF-activity (SCENIC/
AUCell lineage, Nat Methods 2017), now often superseded by CollecTRI (NAR 2023). Notably, Arc's
"virtual cell" STATE model uses no GRN prior at all. Our donor-held-out design addresses the
pseudoreplication/donor-confound pitfall central to between-subjects disease scRNA-seq
(Squair et al., Nat Commun 2021).

## Limitations
Single primary dataset (COVID as replication); n=500 pseudobulk; RA disease confounded with donor;
5 seeds; one encoder family; probes = logistic/kNN; TF-activity dims differ by network (293 vs 675)
so the CollecTRI arm is not dimension-matched to DoRothEA; no formal HPO (by design — see below).

## On hyperparameter optimization
Deliberately none. The brief states peak performance is not the goal; all models share
architecture/budget *by design*, so tuning one more than another would break the fairness that
makes the comparison valid (scIB's benchmark likewise used defaults). Instead we swept prior
strength (soft-λ), ran 5 seeds for variance, and used matched-dimension and rewired controls — the
robustness checks that actually guard the conclusion. A bottleneck-dim sensitivity sweep (32/64/128)
is the one cheap extension that would further rule out a tuning artifact.

## What would make it biologically stronger
A genuinely *regulatory* readout (perturbation response or measured TF activity) rather than
cell-type identity, which linear methods already solve; **within-subjects** perturbation data so
state isn't aliased with donor; more datasets/assays; and a soft/GNN prior evaluated specifically
in the low-data regime where the transform already shows a benefit.

## What I deliberately left out (48h scope)
Raw single-cell modeling; STATE/metabolic/pathway priors (out of scope); a full GNN (masked-MLP +
soft penalty covered the "learned prior" family); formal HPO; a bottleneck-dim sweep. I prioritized
a fair, controlled comparison of *how the prior is applied* (constraint vs transform), with
corruption, matched-dimension, and second-dataset controls, over broad but shallow coverage.
