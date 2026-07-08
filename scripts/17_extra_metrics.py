"""Extra representation-quality metrics: macro AUC-ROC, silhouette, cell-type ASW.

For each model, we compute the full-data embedding (same setup as 14_clustering.py — this
means the AE-based rows are confounded per R1) and then evaluate:

- **Silhouette score** (label-free, geometric): mean s(i) over all samples.
- **Cell-type ASW**: mean silhouette within each cell type, then average across types.
  Reported both raw (in [-1, 1]) and normalised to [0, 1] as (asw+1)/2 (scIB convention).
- **Macro AUC-ROC**: donor-grouped 5-fold CV on the frozen embedding, LR probe with
  predict_proba, one-vs-rest AUC macro-averaged over 15 cell types.
- **Macro-F1**: same probe, hard predictions — sanity vs `final.csv`.

Output: results/tables/extra_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import LabelBinarizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                                    # noqa: E402
from grn_bench.data import load_aligned                             # noqa: E402
from grn_bench.eval import donor_grouped_folds                      # noqa: E402
from grn_bench.experiments import compute_tfact, make_embedder      # noqa: E402

MODELS = ["pca", "baseline", "dc_tfact", "dc_tfact_collectri", "grn_real", "grn_decoder"]
N_FOLDS = 5
SEED = 0


def probe_metrics(emb, y, donors, labels):
    """Donor-grouped 5-fold CV LR probe -> mean macro-F1 and macro AUC-ROC."""
    aucs, f1s = [], []
    for train_idx, test_idx in donor_grouped_folds(donors, n_splits=N_FOLDS, seed=SEED):
        sc = StandardScaler().fit(emb[train_idx])
        Xtr = sc.transform(emb[train_idx])
        Xte = sc.transform(emb[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[train_idx])

        # macro-F1 (hard predictions)
        y_pred = clf.predict(Xte)
        f1s.append(f1_score(y[test_idx], y_pred, average="macro", labels=labels))

        # macro AUC-ROC (one-vs-rest, using predict_proba)
        proba = clf.predict_proba(Xte)                # (n_test, n_classes_seen)
        proba_full = np.zeros((len(test_idx), len(labels)))
        for j, c in enumerate(clf.classes_):
            proba_full[:, labels.index(c)] = proba[:, j]

        lb = LabelBinarizer().fit(labels)
        y_bin = lb.transform(y[test_idx])
        # drop classes with no positive in this fold (roc_auc undefined) via labels arg
        try:
            auc = roc_auc_score(
                y_bin, proba_full,
                average="macro", multi_class="ovr",
            )
        except ValueError:
            # some class missing in test -> compute per-column, skip degenerate
            per_class = []
            for j, c in enumerate(labels):
                if y_bin[:, j].sum() == 0 or y_bin[:, j].sum() == len(test_idx):
                    continue
                per_class.append(roc_auc_score(y_bin[:, j], proba_full[:, j]))
            auc = float(np.mean(per_class)) if per_class else np.nan
        aucs.append(auc)
    return float(np.mean(f1s)), float(np.mean(aucs))


def geometric_metrics(emb, y, labels):
    """Silhouette + cell-type ASW on standardised embedding."""
    E = StandardScaler().fit_transform(emb)
    sil = float(silhouette_score(E, y))
    per_sample = silhouette_samples(E, y)
    per_class = [per_sample[y == c].mean() for c in labels if (y == c).sum() > 1]
    ct_asw = float(np.mean(per_class))
    ct_asw_norm = (ct_asw + 1.0) / 2.0
    return sil, ct_asw, ct_asw_norm


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, "dorothea")
    data["tfact_collectri"] = compute_tfact(data, "collectri")

    y = data["obs"].cell_type.values
    donors = data["obs"].donor_id.values
    labels = sorted(np.unique(y).tolist())
    n = data["X"].shape[0]
    print(f"[data] {data['X'].shape}  cell types={len(labels)}  donors={len(np.unique(donors))}  device={dev}")

    rows = []
    for m in MODELS:
        print(f"[model] {m}")
        all_idx = np.arange(n)
        emb = make_embedder(m, data, all_idx, dev, SEED, 250, data["X"])
        sil, asw, asw_n = geometric_metrics(emb, y, labels)
        f1, auc = probe_metrics(emb, y, donors, labels)
        row = dict(model=m, macro_f1=f1, macro_auc=auc,
                   silhouette=sil, ct_asw=asw, ct_asw_norm=asw_n,
                   ae_confounded=(m in ("baseline", "grn_real", "grn_decoder")))
        rows.append(row)
        print(f"    macro_f1={f1:.3f}  macro_auc={auc:.3f}  "
              f"sil={sil:.3f}  ct_asw={asw:.3f}  ct_asw_norm={asw_n:.3f}  "
              f"confounded={row['ae_confounded']}")

    out = pd.DataFrame(rows)
    outfile = ROOT / "results" / "tables" / "extra_metrics.csv"
    out.to_csv(outfile, index=False)
    print(f"[done] wrote {outfile}")


if __name__ == "__main__":
    main()
