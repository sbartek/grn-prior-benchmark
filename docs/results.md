# Results

Full interpretation in [`memo/memo.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/memo/memo.md).
All scores: donor-grouped 5-fold CV, macro-F1, mean over 2 seeds.

## Dataset suitability (Step 1, confirmed on data)
108,717 cells × 61,497 genes, **raw integer counts**. Balanced **18 RA / 18 normal** donors;
**no sex confound** (12F/6M in both arms); **single assay** (10x 3′ v3, so no assay–disease
confound); **15 cell types** present (some rare: CD4 α-β T 595 cells, γδ-T 1,424). Design is
between-subjects → disease is confounded with donor. Consequence: **cell type** = trustworthy
readout, **disease** = suggestive only, always splitting by donor.

## Headline
**The DoRothEA prior did not improve embeddings.** On cell type it trails both PCA and the
unconstrained baseline at full data, low data, and under noise. A faint signal exists (real graph
> degree-preserving rewire under stress) but it never beats the baseline and is contradicted by
the random-graph control at full data. Disease is not decodable from held-out donors. A fair
**negative result**.

## Full-data comparison
![full](img/fig_full.png)
Cell type: PCA **0.868** > baseline 0.791 > random 0.731 > real 0.699 = sign-shuffled 0.699 >
rewired 0.650. The real graph does **not** beat its same-density controls — the decisive test is
negative: any effect is sparsity, not regulatory content.

## Low-data
![lowdata](img/fig_lowdata.png)
Baseline > grn_real > grn_rewired at every training-donor count (k=4/8/16). The prior does not
rescue the low-data regime; but grn_real consistently edges grn_rewired (weak structural signal).

## Noise robustness
![noise](img/fig_noise.png)
Same ordering under count-thinning (p=0.3, 0.1). The prior does not improve noise robustness.

## Graph corruption (decisive)
At full data grn_real (0.699) ties sign-shuffled (0.699) and is **below random** (0.731). Under
low-data/noise grn_real > grn_rewired but the gap is small and never reaches baseline. Verdict:
the *specific* DoRothEA biology adds little beyond graph density.

## Biology vs batch/donor
![leakage](img/fig_leakage.png)
Donor-prediction accuracy (lower = less leakage): baseline 0.086 < random 0.098 < grn_real 0.114
< sign-shuf 0.132 < rewired 0.136 < PCA 0.166. The prior does not reduce donor signal.
