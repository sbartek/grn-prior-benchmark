# Technical Memo — GRN-Prior Expression Embedding Benchmark

*(~2 pages. Draft skeleton — fill as experiments complete. Keep claims scoped to evidence.)*

## Question & framing
_What we tested and why "better" means biological state, not reconstruction._

## Data & task chosen
_RA PBMC dataset; predicting cell type (primary) and disease (suggestive). Why these labels._

## Is the dataset suitable?
Checked directly on the fetched data (108,717 cells × 61,497 genes, raw counts):

- **Disease is balanced:** 18 rheumatoid-arthritis vs 18 normal donors (48,637 / 60,080 cells).
- **No sex confound:** 12 female / 6 male donors in *both* arms.
- **No assay confound:** a single assay (10x 3′ v3) across all cells — so disease cannot be
  explained by chemistry/platform differences.
- **15 immune cell types** present; some are rare (CD4 α-β T 595 cells, γδ-T 1,424).

The design is **between-subjects**, so disease is inevitably confounded with **donor identity**:
any RA-vs-normal signal could be donor idiosyncrasy. We therefore treat **cell type** as the
trustworthy readout and report **disease** as suggestive only, always splitting by donor.

## Pseudobulk & graph construction
**Pseudobulk:** summed raw counts within each (donor × cell_type) group, dropped groups < 10
cells (536 possible → **500 kept**, median 104 cells/group), then CP10K + log1p. Result:
**500 samples × 21,572 expressed genes**, spanning 36 donors × 15 cell types. Summation (not
mean) keeps the aggregation statistically principled; raw counts retained for the noise
experiments.

**Graph:** DoRothEA (A+B+C) filtered to measured genes. Each **hidden unit = a TF**, connected
only to its target genes (+ the TF's own gene as a self-loop); the encoder's input→hidden mask
*is* the gene×TF adjacency. Result: **8,376 input genes × 411 TF units, 30,609 edges** (0.89%
density). Controls share the exact same gene set and (where noted) edge count: degree-preserving
**rewire** (28,841 edges), **sign-shuffle** (30,609), **random** Erdős–Rényi (30,486); plus
density ablations at confidence A (5,664) and A+B (14,312). Because baseline and controls see the
identical gene set, any advantage is attributable to graph *structure*, not feature selection.

## What we compared
_Baseline (PCA / capacity-matched MLP AE) vs GRN-aware (masked-MLP restricted to TF–target edges);
matched gene set, capacity, training budget._

## Did the GRN help — and where?
_Full-data vs low-data vs noise. Curves, not single numbers._

## Graph controls (the decisive test)
_Real graph vs degree-preserving rewire vs sign-shuffle vs random. Is the benefit biology or sparsity?_

## Biology vs batch/donor
_Donor-predictability of embeddings; batch mixing. Does the model learn state or donor identity?_

## Limitations
_Small N, one dataset, donor confound, no hyperparameter search, pseudobulk aggregation choices._

## What would make this biologically stronger
_Within-subjects perturbation design; more datasets; matched assays; richer biological readouts._
