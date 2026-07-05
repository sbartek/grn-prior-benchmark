"""Unsupervised (label-free) evaluation: cluster each embedding, compare to true cell types.

Complements the supervised probe (scIB-style). For each model we build the full embedding, run
KMeans(15), and score against cell type with ARI + NMI (mean over seeds). Clustering is a stricter
test than the probe — it asks whether biology is the *intrinsic* geometry, not just label-separable.

Output: results/tables/clustering.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                                    # noqa: E402
from grn_bench.data import load_aligned                             # noqa: E402
from grn_bench.experiments import compute_tfact, make_embedder      # noqa: E402

MODELS = ["pca", "baseline", "dc_tfact", "dc_tfact_collectri", "grn_real", "grn_decoder"]
SEEDS = (0, 1, 2)


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, "dorothea")
    data["tfact_collectri"] = compute_tfact(data, "collectri")
    ct = data["obs"].cell_type.values
    n = data["X"].shape[0]
    all_idx = np.arange(n)
    n_types = len(np.unique(ct))
    print(f"[data] {data['X'].shape}  cell types={n_types}  device={dev}")

    rows = []
    for m in MODELS:
        emb = make_embedder(m, data, all_idx, dev, 0, 250, data["X"])     # train/transform on all
        embs = StandardScaler().fit_transform(emb)
        aris, nmis = [], []
        for s in SEEDS:
            lab = KMeans(n_types, random_state=s, n_init=10).fit_predict(embs)
            aris.append(adjusted_rand_score(ct, lab))
            nmis.append(normalized_mutual_info_score(ct, lab))
        rows.append(dict(model=m, ARI=float(np.mean(aris)), NMI=float(np.mean(nmis))))
        print(f"  {m:20s} ARI={np.mean(aris):.3f}  NMI={np.mean(nmis):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "clustering.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
