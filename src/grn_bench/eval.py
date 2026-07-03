"""Step 5 — evaluate a frozen embedding on biological readouts.

Probes are fit INSIDE each donor-grouped fold on the training donors and scored on held-out
donors, so neither the encoder nor the probe ever sees a test donor. We report macro-F1 (robust
to class imbalance) for a linear (logistic) probe; kNN is available as a non-linear check.
donor_predictability() measures how much donor identity leaks into the embedding (lower better).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def _fit_score(clf, Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf.fit(sc.transform(Xtr), ytr)
    return f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")


def probe_precomputed(emb, y, groups, folds, kind="linear"):
    """Score an embedding whose per-fold splits are given (so AE + probe share folds)."""
    scores = []
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0) if kind == "linear" \
            else KNeighborsClassifier(n_neighbors=min(15, len(tr) - 1))
        scores.append(_fit_score(clf, emb[tr], y[tr], emb[te], y[te]))
    return float(np.mean(scores)), float(np.std(scores))


def donor_grouped_folds(groups, n_splits=5, seed=0):
    """GroupKFold splits by donor. Deterministic; seed kept for API symmetry."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(len(groups))
    return list(gkf.split(np.arange(len(groups)), dummy_y, groups))


def donor_predictability(emb, donor, n_splits=5, seed=0):
    """How well donor identity is decodable from the embedding (leakage proxy; lower=better)."""
    y = np.asarray(donor)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(emb, y):
        sc = StandardScaler().fit(emb[tr])
        clf = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(emb[tr]), y[tr])
        accs.append((clf.predict(sc.transform(emb[te])) == y[te]).mean())
    return float(np.mean(accs))
