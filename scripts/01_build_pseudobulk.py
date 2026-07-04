"""Step 2 — build pseudobulk profiles from the raw single-cell data.

Aggregation unit: (donor_id x cell_type). For each group we SUM raw counts across its cells
(the standard, statistically-principled pseudobulk), then library-size normalise (CP10K) and
log1p. Groups with too few cells are dropped (noisy, unreliable means).

Input : data/raw.h5ad         (108,717 cells x 61,497 genes, raw counts)
Output: data/pseudobulk.h5ad  (n_groups x n_genes; obs carries donor/cell_type/disease/sex/n_cells)
"""
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse

import sys

ROOT = Path(__file__).resolve().parents[1]
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw.h5ad"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "pseudobulk.h5ad"

MIN_CELLS = 10          # drop (donor x cell_type) groups smaller than this
MIN_GROUP_GENES = 1     # keep genes expressed in >=1 pseudobulk sample
GROUP_KEYS = ["donor_id", "cell_type"]
CARRY_OBS = ["donor_id", "cell_type", "disease", "sex", "assay", "self_reported_ethnicity"]


def main() -> None:
    adata = sc.read_h5ad(IN)
    assert np.allclose(adata.X[:20].toarray(), np.round(adata.X[:20].toarray())), "expected raw counts"

    # gene symbols for later DoRothEA matching; keep ensembl id too
    gene_symbol = adata.var["feature_name"].astype(str).values if "feature_name" in adata.var else adata.var_names.values

    # build a (n_groups x n_cells) group-indicator matrix, then one sparse matmul sums counts
    grp = adata.obs.groupby(GROUP_KEYS, observed=True)
    group_index = list(grp.groups.keys())
    codes = grp.ngroup().values                         # cell -> group id
    counts = np.bincount(codes, minlength=len(group_index))

    keep_groups = counts >= MIN_CELLS
    print(f"[qc] {len(group_index)} groups; {keep_groups.sum()} kept (>= {MIN_CELLS} cells), "
          f"{(~keep_groups).sum()} dropped")

    n_groups, n_cells = len(group_index), adata.n_obs
    indicator = sparse.csr_matrix(
        (np.ones(n_cells), (codes, np.arange(n_cells))), shape=(n_groups, n_cells)
    )
    pb = indicator @ adata.X                            # (n_groups x n_genes) summed raw counts
    pb = sparse.csr_matrix(pb)

    # subset to kept groups
    pb = pb[keep_groups]
    kept_keys = [k for k, keep in zip(group_index, keep_groups) if keep]
    kept_counts = counts[keep_groups]

    import pandas as pd
    pdata = sc.AnnData(X=pb, dtype=np.float32)
    pdata.var_names = adata.var_names
    pdata.var["gene_symbol"] = gene_symbol
    # per-group metadata: first cell's values for each carried column, indexed by GROUP_KEYS
    carry = [c for c in CARRY_OBS if c in adata.obs.columns]
    meta = adata.obs.groupby(GROUP_KEYS, observed=True)[carry].first()
    obs = meta.loc[kept_keys].reset_index(drop=True).astype(str)
    obs["n_cells"] = kept_counts
    obs.index = [f"{d}|{ct}" for d, ct in kept_keys]
    pdata.obs = obs

    # drop all-zero genes, keep raw pseudobulk counts, then CP10K + log1p
    gene_mask = np.asarray((pdata.X > 0).sum(0)).ravel() >= MIN_GROUP_GENES
    pdata = pdata[:, gene_mask].copy()
    pdata.layers["counts"] = pdata.X.copy()
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)

    print(f"[ok] pseudobulk: {pdata.shape[0]} samples x {pdata.shape[1]} genes")
    print(f"     donors={pdata.obs['donor_id'].nunique()} cell_types={pdata.obs['cell_type'].nunique()}")
    print(f"     cells/group: min={pdata.obs.n_cells.min()} median={int(pdata.obs.n_cells.median())} max={pdata.obs.n_cells.max()}")
    pdata.write_h5ad(OUT)
    print(f"[write] {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
