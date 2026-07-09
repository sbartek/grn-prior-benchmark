"""Per-fold extras — fixes the R1 confound for AE-based models.

Same metric set as 17_extra_metrics.py but with proper per-fold AE training:

  For each fold:
    - train encoder ONLY on train_idx donors
    - probe: train LR on train-embedding, evaluate on test-embedding  ->  F1 / AUC
    - geometry: compute silhouette + cell-type ASW on TEST-donor embeddings only
                (no leakage — AE has never seen these samples)
  Average all metrics across folds.

For fixed transforms (pca, dc_tfact, dc_tfact_collectri) this is functionally
equivalent to 17 (PCA is already train_idx-aware; ULM is per-sample). For AE
rows this collapses the R1 inflation.

Output: results/tables/extra_metrics_perfold.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
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
EPOCHS = 250


def per_fold_metrics(model, data, dev, y, donors, labels):
    n = data["X"].shape[0]
    f1s, aucs, sils, asws, aris, nmis = [], [], [], [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(
        donor_grouped_folds(donors, n_splits=N_FOLDS, seed=SEED)
    ):
        emb = make_embedder(model, data, train_idx, dev, SEED, EPOCHS, data["X"])
        # probe: train on train, eval on test
        sc = StandardScaler().fit(emb[train_idx])
        Xtr = sc.transform(emb[train_idx])
        Xte = sc.transform(emb[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[train_idx])

        f1s.append(f1_score(y[test_idx], clf.predict(Xte),
                            average="macro", labels=labels))

        proba = clf.predict_proba(Xte)
        proba_full = np.zeros((len(test_idx), len(labels)))
        for j, c in enumerate(clf.classes_):
            proba_full[:, labels.index(c)] = proba[:, j]
        lb = LabelBinarizer().fit(labels)
        y_bin = lb.transform(y[test_idx])
        try:
            auc = roc_auc_score(y_bin, proba_full, average="macro", multi_class="ovr")
        except ValueError:
            per_class = []
            for j, c in enumerate(labels):
                if y_bin[:, j].sum() in (0, len(test_idx)):
                    continue
                per_class.append(roc_auc_score(y_bin[:, j], proba_full[:, j]))
            auc = float(np.mean(per_class)) if per_class else np.nan
        aucs.append(auc)

        # geometry on TEST-donor embeddings only (AE never saw them)
        emb_test_std = StandardScaler().fit_transform(emb[test_idx])
        y_test = y[test_idx]
        # need at least 2 samples in >= 2 classes to compute silhouette
        classes_here = [c for c in labels if (y_test == c).sum() > 1]
        if len(classes_here) >= 2:
            try:
                sil_fold = float(silhouette_score(emb_test_std, y_test))
                per_sample = silhouette_samples(emb_test_std, y_test)
                per_class = [per_sample[y_test == c].mean() for c in classes_here]
                asw_fold = float(np.mean(per_class))
            except ValueError:
                sil_fold, asw_fold = np.nan, np.nan
        else:
            sil_fold, asw_fold = np.nan, np.nan
        sils.append(sil_fold)
        asws.append(asw_fold)

        # ARI + NMI on TEST-donor embeddings — KMeans with k = # unique test classes
        y_test = y[test_idx]
        k_here = len(np.unique(y_test))
        if k_here >= 2 and len(test_idx) >= k_here:
            emb_test = StandardScaler().fit_transform(emb[test_idx])
            lab = KMeans(n_clusters=k_here, random_state=SEED, n_init=10).fit_predict(emb_test)
            ari_fold = float(adjusted_rand_score(y_test, lab))
            nmi_fold = float(normalized_mutual_info_score(y_test, lab))
        else:
            ari_fold, nmi_fold = np.nan, np.nan
        aris.append(ari_fold)
        nmis.append(nmi_fold)

        print(f"    fold {fold_i}: f1={f1s[-1]:.3f} auc={aucs[-1]:.3f} "
              f"sil={sil_fold:.3f} asw={asw_fold:.3f} "
              f"ari={ari_fold:.3f} nmi={nmi_fold:.3f}")

    return (float(np.mean(f1s)), float(np.mean(aucs)),
            float(np.nanmean(sils)), float(np.nanmean(asws)),
            float(np.nanmean(aris)), float(np.nanmean(nmis)))


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, "dorothea")
    data["tfact_collectri"] = compute_tfact(data, "collectri")

    y = data["obs"].cell_type.values
    donors = data["obs"].donor_id.values
    labels = sorted(np.unique(y).tolist())
    print(f"[data] {data['X'].shape}  cell types={len(labels)}  donors={len(np.unique(donors))}  device={dev}")

    rows = []
    for m in MODELS:
        print(f"[model] {m}")
        f1, auc, sil, asw, ari, nmi = per_fold_metrics(m, data, dev, y, donors, labels)
        rows.append(dict(
            model=m, macro_f1=f1, macro_auc=auc,
            silhouette=sil, ct_asw=asw, ct_asw_norm=(asw + 1) / 2,
            ari=ari, nmi=nmi,
        ))
        print(f"    -> mean f1={f1:.3f} auc={auc:.3f} sil={sil:.3f} asw={asw:.3f} "
              f"ari={ari:.3f} nmi={nmi:.3f}")

    out = pd.DataFrame(rows)
    outfile = ROOT / "results" / "tables" / "extra_metrics_perfold.csv"
    out.to_csv(outfile, index=False)
    print(f"[done] wrote {outfile}")


if __name__ == "__main__":
    main()
