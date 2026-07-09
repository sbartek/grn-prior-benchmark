"""Curiosity: does macro-F1 improve if we keep only the top-K most discriminative TFs?

Motivation for the review call — Caelan may ask "why not cell-type-specific regulons?"
A full SCENIC / SCENIC+ pipeline is 6-24h, won't finish before the call. This is a
proxy: rank TFs by ANOVA F-statistic across cell types on TRAIN samples per fold, keep
top-K, run LR probe. Tests "are irrelevant TFs adding noise?"

Per fold: rank TFs on TRAIN embeddings + labels, select top-K, evaluate on TEST.
No leakage since ranking is train-only.

Output: results/tables/tf_feature_selection.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench.data import load_aligned                              # noqa: E402
from grn_bench.eval import donor_grouped_folds                       # noqa: E402
from grn_bench.experiments import compute_tfact                      # noqa: E402

K_VALUES = [10, 30, 50, 100, 200, 293]        # 293 = all TFs
N_FOLDS = 5
SEED = 0


def probe(emb, y, donors, labels):
    f1s, aucs = [], []
    for train_idx, test_idx in donor_grouped_folds(donors, n_splits=N_FOLDS, seed=SEED):
        sc = StandardScaler().fit(emb[train_idx])
        Xtr = sc.transform(emb[train_idx])
        Xte = sc.transform(emb[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[train_idx])
        f1s.append(f1_score(y[test_idx], clf.predict(Xte), average="macro", labels=labels))
        proba = clf.predict_proba(Xte)
        proba_full = np.zeros((len(test_idx), len(labels)))
        for j, c in enumerate(clf.classes_):
            proba_full[:, labels.index(c)] = proba[:, j]
        lb = LabelBinarizer().fit(labels)
        y_bin = lb.transform(y[test_idx])
        try:
            auc = roc_auc_score(y_bin, proba_full, average="macro", multi_class="ovr")
        except ValueError:
            auc = np.nan
        aucs.append(auc)
    return float(np.mean(f1s)), float(np.mean(aucs))


def probe_topk(tfact, y, donors, labels, k):
    """LR probe using top-k TFs ranked per-fold on train ANOVA F-stat. No leakage."""
    f1s, aucs = [], []
    for train_idx, test_idx in donor_grouped_folds(donors, n_splits=N_FOLDS, seed=SEED):
        F, _ = f_classif(tfact[train_idx], y[train_idx])
        top_ix = np.argsort(-F)[:k]
        Xtr = tfact[train_idx][:, top_ix]
        Xte = tfact[test_idx][:, top_ix]
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte = sc.transform(Xte)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[train_idx])
        f1s.append(f1_score(y[test_idx], clf.predict(Xte), average="macro", labels=labels))
        proba = clf.predict_proba(Xte)
        proba_full = np.zeros((len(test_idx), len(labels)))
        for j, c in enumerate(clf.classes_):
            proba_full[:, labels.index(c)] = proba[:, j]
        lb = LabelBinarizer().fit(labels)
        y_bin = lb.transform(y[test_idx])
        try:
            auc = roc_auc_score(y_bin, proba_full, average="macro", multi_class="ovr")
        except ValueError:
            auc = np.nan
        aucs.append(auc)
    return float(np.mean(f1s)), float(np.mean(aucs))


def main():
    data = load_aligned()
    tfact = compute_tfact(data, "dorothea")
    tfact_c = compute_tfact(data, "collectri")
    y = data["obs"].cell_type.values
    donors = data["obs"].donor_id.values
    labels = sorted(np.unique(y).tolist())
    print(f"[data] tfact (dorothea) {tfact.shape}  tfact (collectri) {tfact_c.shape}")

    rows = []
    for name, tf in [("dc_tfact", tfact), ("dc_tfact_collectri", tfact_c)]:
        for k in K_VALUES:
            if k > tf.shape[1]:
                k_eff = tf.shape[1]
            else:
                k_eff = k
            f1, auc = probe_topk(tf, y, donors, labels, k_eff)
            rows.append(dict(embedding=name, k=k_eff, macro_f1=f1, macro_auc=auc))
            print(f"  {name:20s} top-{k_eff:3d}: f1={f1:.3f} auc={auc:.3f}")

    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "tf_feature_selection.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
