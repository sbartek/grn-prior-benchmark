# Results

*Placeholder — filled as experiments complete (Step 6+). Figures land in
[`results/figures/`](https://github.com/sbartek/grn-prior-benchmark/tree/main/results), the
narrative in [`memo/memo.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/memo/memo.md).*

## Dataset suitability (Step 1, confirmed on data)
108,717 cells × 61,497 genes, **raw integer counts**. Balanced **18 RA / 18 normal** donors;
**no sex confound** (12F/6M in both arms); **single assay** (10x 3′ v3, so no assay–disease
confound); **15 cell types** present (some rare: CD4 α-β T 595 cells, γδ-T 1,424). Design is
between-subjects → disease is confounded with donor. Consequence: **cell type** = trustworthy
readout, **disease** = suggestive only, always splitting by donor.

## Headline (to fill)
_Did the GRN prior help, and where?_

## Full-data comparison
_Baseline vs GRN at full data — expected to roughly tie._

## Low-data
_Metric vs number of training donors._

## Noise robustness
_Metric vs injected technical noise level._

## Graph corruption (decisive)
_Real graph vs degree-preserving rewire / sign-shuffle / random. Is the benefit biology or sparsity?_

## Biology vs batch/donor
_Donor-predictability of the embedding; batch mixing._

## Ablation table
_(to fill)_
