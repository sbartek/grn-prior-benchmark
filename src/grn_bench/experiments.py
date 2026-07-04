"""Step 6 — experiment runner: train encoders under conditions, probe the embeddings.

For every (model, condition, fold, seed): the encoder is trained on the TRAIN donors of that
fold (optionally subsampled / noised), all samples are embedded, and the frozen embedding is
probed on the held-out donors. This keeps encoder-training and probing on the same donor split,
so there is no leakage.

Conditions:
  full          all train donors, no noise
  lowdata:k     subsample train donors to k
  noise:p       binomial thinning of counts to fraction p (simulates lower sequencing depth)
"""
import numpy as np

from . import models as M
from .eval import probe_precomputed


def _thin_counts(counts, p, rng):
    """Binomial thinning: keep each molecule w.p. p, then CP10K + log1p (matches Step 2)."""
    thinned = rng.binomial(counts.astype(np.int64), p).astype(np.float32)
    lib = thinned.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(thinned / lib * 1e4)


_DOR_NET = None


def _dorothea_net():
    global _DOR_NET
    if _DOR_NET is None:
        import decoupler as dc
        _DOR_NET = dc.op.dorothea(organism="human")
    return _DOR_NET


def compute_tfact_mat(X, genes):
    """decoupler ULM TF-activity for an expression matrix (per-sample, fixed GRN transform)."""
    import anndata as ad
    import decoupler as dc
    A = ad.AnnData(np.asarray(X).copy())
    A.var_names = list(genes)
    dc.mt.ulm(A, net=_dorothea_net(), tmin=5, verbose=False)
    return np.asarray(A.obsm["score_ulm"])


def compute_tfact(data):
    return compute_tfact_mat(data["X"], data["genes"])


def make_embedder(spec, data, train_idx, device, seed, epochs, X_input):
    """Train the encoder on train_idx and return embeddings for ALL samples."""
    torch_seed(seed)
    genes, tfs = data["genes"], data["tfs"]
    n_genes, n_hidden = len(genes), len(tfs)

    if spec == "pca":
        from sklearn.decomposition import PCA
        k = min(64, len(train_idx) - 1, X_input.shape[1])
        p = PCA(n_components=k, random_state=seed).fit(X_input[train_idx])
        return p.transform(X_input)

    if spec in ("dc_tfact", "dc_tfact_pca"):
        # decoupler TF-activity embedding (canonical non-learned GRN prior). Reuse the precomputed
        # clean activity for full/low-data; recompute on the noised input under the noise condition
        # (else it would unfairly see clean data). Per-sample transform -> no leakage.
        tf = data["tfact"] if X_input is data["X"] else compute_tfact_mat(X_input, genes)
        if spec == "dc_tfact":
            return tf
        from sklearn.decomposition import PCA           # dimension-matched control (64-d)
        k = min(64, len(train_idx) - 1, tf.shape[1])
        return PCA(n_components=k, random_state=seed).fit(tf[train_idx]).transform(tf)

    # soft prior: dense first layer + penalty pulling off-regulon weights to 0 (spec 'grn_soft[:lam]')
    soft_mask = None
    soft_lambda = 0.0
    mask = sign = None
    if spec.startswith("grn_soft"):
        soft_lambda = float(spec.split(":")[1]) if ":" in spec else 1e-3
        soft_mask, _ = M.build_mask(data["graph"], "real", genes, n_hidden, device)
    elif spec != "baseline":
        variant = spec.replace("grn_", "")
        mask, sign = M.build_mask(data["graph"], variant, genes, n_hidden, device)
    model = M.AutoEncoder(n_genes, n_hidden, 64, mask=mask, sign=sign)

    rng = np.random.default_rng(seed)
    val = rng.choice(train_idx, size=max(2, int(0.15 * len(train_idx))), replace=False)
    model, _ = M.train_ae(model, X_input[train_idx], _local_val_idx(train_idx, val),
                          device, epochs=epochs, patience=25,
                          soft_mask=soft_mask, soft_lambda=soft_lambda)
    return M.embed(model, X_input, device)


def _local_val_idx(train_idx, val_global):
    pos = {g: i for i, g in enumerate(train_idx)}
    return np.array([pos[v] for v in val_global])


def torch_seed(seed):
    import torch
    torch.manual_seed(seed)


def run_cv(data, spec, task_key, condition, device, seeds=(0, 1), n_splits=5, epochs=250,
           return_flat=False):
    """Macro-F1 for one model x condition x task via nested donor-CV.

    Returns per-seed means by default; if return_flat, returns a list of
    {seed, fold, f1} for every (seed, fold) so callers can do paired comparisons.
    """
    from .eval import donor_grouped_folds
    y = data["obs"][task_key].astype(str).values
    donor = data["obs"]["donor_id"].astype(str).values
    results, flat = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if condition.startswith("noise:"):
            p = float(condition.split(":")[1])
            X_input = _thin_counts(data["counts"], p, rng)
        else:
            X_input = data["X"]

        folds = donor_grouped_folds(donor, n_splits=n_splits, seed=seed)
        fold_scores = []
        for fi, (tr, te) in enumerate(folds):
            tr_use = tr
            if condition.startswith("lowdata:"):
                k = int(condition.split(":")[1])
                tr_donors = rng.permutation(np.unique(donor[tr]))[:k]
                tr_use = tr[np.isin(donor[tr], tr_donors)]
            emb = make_embedder(spec, data, tr_use, device, seed, epochs, X_input)
            f1, _ = probe_precomputed(emb, y, donor, [(tr_use, te)])
            fold_scores.append(f1)
            flat.append({"seed": seed, "fold": fi, "f1": f1})
        results.append(float(np.mean(fold_scores)))
    return flat if return_flat else results
