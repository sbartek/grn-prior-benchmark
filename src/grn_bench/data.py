"""Shared data loading: align the pseudobulk matrix to the graph's gene order.

Both encoders and all controls consume the SAME gene set in the SAME order (graph['genes']),
so any performance difference is attributable to graph structure, not feature selection.
"""
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]


def load_aligned(pseudobulk="data/pseudobulk.h5ad", graph="data/graph.npz"):
    pdata = sc.read_h5ad(ROOT / pseudobulk)
    g = np.load(ROOT / graph, allow_pickle=True)
    genes, tfs = g["genes"], g["tfs"]

    sym = pdata.var["gene_symbol"].astype(str).values
    first_col = {}
    for i, s in enumerate(sym):
        first_col.setdefault(s, i)
    idx = np.array([first_col[s] for s in genes])          # graph gene -> pseudobulk column

    counts_full = pdata.layers["counts"].toarray().astype(np.float32)
    lib_full = counts_full.sum(1)                           # full-transcriptome library / sample
    X = pdata.X.toarray()[:, idx].astype(np.float32)        # (n_samples x n_genes), log-norm
    counts = counts_full[:, idx]                            # raw graph-gene counts, for noise sims
    return {
        "X": X,
        "counts": counts,
        "lib_full": lib_full,                               # so noise-thinning matches Step-2 CP10K
        "obs": pdata.obs.reset_index(drop=True),
        "genes": genes,
        "tfs": tfs,
        "graph": {k: g[k] for k in g.files},
    }
