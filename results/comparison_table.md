# Consolidated comparison (cell-type macro-F1 unless noted)

Donor-grouped 5-fold CV; primary = RA PBMC, COVID = COVID PBMC. "disease" = COVID 3-class.

| Model | Prim full | Prim k=8 | Prim noise | COVID full | COVID k=8 | COVID noise | COVID disease |
|---|---|---|---|---|---|---|---|
| PCA | 0.869 | 0.632 | 0.770 | 0.936 | 0.760 | 0.919 | 0.715 |
| baseline (dense AE) | 0.785 | 0.661 | 0.732 | 0.860 | 0.696 | 0.859 | 0.588 |
| dc_tfact | 0.847 | 0.686 | 0.720 | 0.906 | 0.763 | 0.876 | 0.659 |
| dc_tfact_collectri | 0.869 | 0.746 | 0.784 | – | – | – | – |
| dc_tfact_pca | 0.783 | 0.630 | 0.686 | 0.896 | 0.739 | 0.849 | 0.673 |
| rand_proj | 0.775 | 0.624 | 0.648 | – | – | – | – |
| dc_tfact_rewired | 0.804 | 0.661 | 0.703 | – | – | – | – |
| grn_real | 0.714 | 0.591 | 0.615 | 0.855 | 0.647 | 0.781 | 0.654 |
| grn_rewired | 0.665 | 0.569 | 0.559 | 0.838 | 0.659 | 0.749 | 0.646 |
| grn_soft:0.001 | 0.753 | 0.641 | 0.715 | 0.762 | 0.613 | 0.803 | 0.499 |
| grn_decoder | 0.738 | 0.612 | 0.656 | 0.832 | 0.622 | 0.814 | 0.664 |
| grn_symmetric | 0.675 | 0.557 | 0.595 | 0.818 | 0.588 | 0.774 | 0.639 |
