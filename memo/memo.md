# Technical Memo — GRN-Prior Expression Embedding Benchmark

*(~2 pages. Draft skeleton — fill as experiments complete. Keep claims scoped to evidence.)*

## Question & framing
_What we tested and why "better" means biological state, not reconstruction._

## Data & task chosen
_RA PBMC dataset; predicting cell type (primary) and disease (suggestive). Why these labels._

## Is the dataset suitable?
_Between-subjects RA/healthy → disease is confounded with donor. What this dataset can and cannot answer._

## Pseudobulk & graph construction
_How donor×cell_type pseudobulk was built; gene alignment to DoRothEA; confidence filtering._

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
