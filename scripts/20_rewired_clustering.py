"""Add dc_tfact_rewired (and dc_tfact_sign) to the clustering/silhouette comparison.

Missing from 14_clustering.py and 18_extra_metrics_perfold.py. Since ULM is a
per-sample fixed transform, no train/test issue — compute on full data.

Output: results/tables/rewired_clustering.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench.data import load_aligned                              # noqa: E402
from grn_bench.experiments import compute_tfact, compute_tfact_mat, corrupted_net  # noqa: E402

SEEDS = (0, 1, 2)


def main():
    data = load_aligned()
    y = data["obs"].cell_type.values
    labels = sorted(np.unique(y).tolist())
    n_types = len(labels)

    # build embeddings — all fixed transforms
    embs = {
        "dc_tfact": compute_tfact(data, "dorothea"),
        "dc_tfact_rewired": compute_tfact_mat(
            data["X"], data["genes"], net=corrupted_net("dorothea", "rewired", 0)
        ),
        "dc_tfact_sign": compute_tfact_mat(
            data["X"], data["genes"], net=corrupted_net("dorothea", "sign", 0)
        ),
        "dc_tfact_collectri": compute_tfact(data, "collectri"),
    }
    print("[embeddings]")
    for n, e in embs.items():
        print(f"  {n:25s} shape={e.shape}")

    rows = []
    for name, emb in embs.items():
        E = StandardScaler().fit_transform(emb)
        aris, nmis = [], []
        for s in SEEDS:
            lab = KMeans(n_types, random_state=s, n_init=10).fit_predict(E)
            aris.append(adjusted_rand_score(y, lab))
            nmis.append(normalized_mutual_info_score(y, lab))
        sil = float(silhouette_score(E, y))
        per_sample = silhouette_samples(E, y)
        per_class = [per_sample[y == c].mean() for c in labels if (y == c).sum() > 1]
        asw = float(np.mean(per_class))
        rows.append(dict(
            model=name,
            ari=float(np.mean(aris)),
            nmi=float(np.mean(nmis)),
            silhouette=sil,
            ct_asw=asw,
        ))
        print(f"  {name:25s} ari={np.mean(aris):.3f} nmi={np.mean(nmis):.3f} "
              f"sil={sil:.3f} ct_asw={asw:.3f}")

    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "rewired_clustering.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
