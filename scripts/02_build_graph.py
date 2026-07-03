"""Step 3 — build the DoRothEA prior adjacency and its control graphs.

Design: each hidden unit of the GRN-aware encoder corresponds to a TF and aggregates that TF's
regulon (its target genes, plus the TF's own gene as a self-loop). So the encoder's input->hidden
mask IS the gene x TF adjacency. Baseline and all controls share the SAME gene set, so any
difference is attributable to graph structure, not feature selection.

Controls (identical edge count / degrees where noted -> isolate 'is it the biology?'):
  real          full DoRothEA (confidence A+B+C)
  real_A        confidence A only        (density ablation)
  real_AB       confidence A+B           (density ablation)
  rewired       degree-preserving target permutation (TF out-deg + gene in-deg multiset preserved)
  sign_shuffled real topology, signs globally permuted
  random        Erdos-Renyi over TF x gene, same edge count (degrees NOT preserved)

Input : data/pseudobulk.h5ad
Output: data/graph.npz   (genes, tfs, and <name>_{rows,cols,signs} edge lists)
"""
from pathlib import Path

import decoupler as dc
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "pseudobulk.h5ad"
OUT = ROOT / "data" / "graph.npz"
SEED = 0


def edges_to_arrays(df, gene_idx, tf_idx):
    rows = df["target"].map(gene_idx).to_numpy()
    cols = df["source"].map(tf_idx).to_numpy()
    signs = np.sign(df["weight"].to_numpy()).astype(np.int8)
    return rows, cols, signs


def main() -> None:
    rng = np.random.default_rng(SEED)
    pdata = sc.read_h5ad(IN)
    measured = set(pdata.var["gene_symbol"].astype(str))

    dor = dc.op.dorothea(organism="human")
    dor = dor[dor["target"].isin(measured)].copy()          # keep edges whose target is measured
    # add self-loops: a TF's own expression informs its activity (if that gene is measured)
    tfs_measured = sorted(set(dor["source"]) & measured)

    genes = sorted(set(dor["target"]) | set(tfs_measured))  # input gene set (shared by all models)
    tfs = sorted(set(dor["source"]))                        # hidden units
    gene_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = {t: i for i, t in enumerate(tfs)}
    print(f"[graph] input genes={len(genes)}  TFs(hidden)={len(tfs)}  edges(ABC)={len(dor)}")

    # self-loop edges (weight +1) for measured TFs
    self_df = pdata.obs.iloc[:0]  # dummy
    import pandas as pd
    self_edges = pd.DataFrame({"source": tfs_measured, "target": tfs_measured,
                               "weight": 1.0, "confidence": "self"})

    variants = {}
    real = pd.concat([dor, self_edges], ignore_index=True)
    dor_A = dor[dor.confidence == "A"]
    dor_AB = dor[dor.confidence.isin(["A", "B"])]
    variants["real"] = real
    variants["real_A"] = pd.concat([dor_A, self_edges], ignore_index=True)
    variants["real_AB"] = pd.concat([dor_AB, self_edges], ignore_index=True)

    # degree-preserving rewire controls for the confidence subsets (biology-vs-sparsity at A/AB)
    for name, sub in [("rewired_A", dor_A), ("rewired_AB", dor_AB)]:
        rw = sub.copy()
        rw["target"] = rng.permutation(sub["target"].to_numpy())
        variants[name] = pd.concat([rw, self_edges], ignore_index=True)

    # controls derived from the full real topology (dor edges only; self-loops re-added)
    rewired = dor.copy()
    rewired["target"] = rng.permutation(dor["target"].to_numpy())   # degree-preserving
    variants["rewired"] = pd.concat([rewired, self_edges], ignore_index=True)

    sign_shuf = dor.copy()
    sign_shuf["weight"] = rng.permutation(dor["weight"].to_numpy())
    variants["sign_shuffled"] = pd.concat([sign_shuf, self_edges], ignore_index=True)

    n_edges = len(dor)
    rand = pd.DataFrame({
        "source": rng.choice(tfs, size=n_edges),
        "target": rng.choice(genes, size=n_edges),
        "weight": rng.choice([1.0, -1.0], size=n_edges),
        "confidence": "rand",
    })
    variants["random"] = pd.concat([rand, self_edges], ignore_index=True)

    out = {"genes": np.array(genes), "tfs": np.array(tfs)}
    for name, df in variants.items():
        df = df.drop_duplicates(["source", "target"])
        r, c, s = edges_to_arrays(df, gene_idx, tf_idx)
        out[f"{name}_rows"], out[f"{name}_cols"], out[f"{name}_signs"] = r, c, s
        print(f"  {name:14s} edges={len(df):6d}  density={len(df)/(len(genes)*len(tfs)):.4f}")

    np.savez_compressed(OUT, **out)
    print(f"[write] {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
