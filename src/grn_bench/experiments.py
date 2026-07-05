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


def _thin_counts(counts, p, rng, lib_full):
    """Binomial thinning of graph-gene counts, renormalised by the FULL-transcriptome library
    (× p, its expected thinned value) so the scale matches Step-2 CP10K and p=1 recovers clean X."""
    thinned = rng.binomial(counts.astype(np.int64), p).astype(np.float32)
    lib = (np.asarray(lib_full, dtype=np.float32) * p).reshape(-1, 1)
    lib[lib == 0] = 1.0
    return np.log1p(thinned / lib * 1e4)


_NETS = {}


def _net(name):
    if name not in _NETS:
        import decoupler as dc
        _NETS[name] = getattr(dc.op, name)(organism="human")
    return _NETS[name]


def corrupted_net(base="dorothea", kind="rewired", seed=0):
    """Corrupt a DoRothEA/CollecTRI net for the TF-activity nulls: 'rewired' = degree-preserving
    target permutation (same regulon sizes, scrambled membership); 'sign' = permuted edge signs."""
    net = _net(base).copy()
    rng = np.random.default_rng(seed)
    if kind == "rewired":
        net["target"] = rng.permutation(net["target"].to_numpy())
    elif kind == "sign":
        net["weight"] = rng.permutation(net["weight"].to_numpy())
    return net.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)


def compute_tfact_mat(X, genes, net="dorothea"):
    """decoupler ULM TF-activity for an expression matrix (per-sample, fixed GRN transform).
    `net` may be a name (cached lookup) or a decoupler-style DataFrame (used directly)."""
    import anndata as ad
    import decoupler as dc
    A = ad.AnnData(np.asarray(X).copy())
    A.var_names = list(genes)
    dc.mt.ulm(A, net=(_net(net) if isinstance(net, str) else net), tmin=5, verbose=False)
    return np.asarray(A.obsm["score_ulm"])


def compute_tfact(data, net="dorothea"):
    return compute_tfact_mat(data["X"], data["genes"], net=net)


def make_embedder(spec, data, train_idx, device, seed, epochs, X_input, z_dim=64, early_stop=True):
    """Train the encoder on train_idx and return embeddings for ALL samples."""
    torch_seed(seed)
    genes, tfs = data["genes"], data["tfs"]
    n_genes, n_hidden = len(genes), len(tfs)

    if spec == "pca":
        from sklearn.decomposition import PCA
        k = min(z_dim, len(train_idx) - 1, X_input.shape[1])
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

    if spec == "dc_tfact_collectri":
        # second TF-activity arm (CollecTRI net) — is the ULM win DoRothEA-specific or generic?
        return data["tfact_collectri"] if X_input is data["X"] \
            else compute_tfact_mat(X_input, genes, net="collectri")

    if spec in ("dc_tfact_rewired", "dc_tfact_sign"):
        # TF-activity on a CORRUPTED DoRothEA net — the transform-side analogue of the encoder's
        # rewired control. If dc_tfact ~ dc_tfact_rewired, even the transform signal is just
        # structure, not the specific regulons.
        kind = "rewired" if spec == "dc_tfact_rewired" else "sign"
        key = f"tfact_{kind}"
        if X_input is data["X"] and key in data:
            return data[key]
        return compute_tfact_mat(X_input, genes, net=corrupted_net("dorothea", kind, 0))

    if spec == "rand_proj":
        # matched-dimension RANDOM linear features (same dim as TF-activity). Fixed transform.
        # If this matches dc_tfact, the TF-activity gain is dimensionality, not biology.
        k = data["tfact"].shape[1]
        R = np.random.default_rng(1000 + seed).standard_normal((n_genes, k)).astype(np.float32) / np.sqrt(k)
        return X_input @ R

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
    model = M.AutoEncoder(n_genes, n_hidden, z_dim, mask=mask, sign=sign)

    rng = np.random.default_rng(seed)
    if early_stop:
        val = rng.choice(train_idx, size=max(2, int(0.15 * len(train_idx))), replace=False)
        val_local = _local_val_idx(train_idx, val)
    else:
        val_local = np.array([0])                       # unused when early_stop=False
    model, _ = M.train_ae(model, X_input[train_idx], val_local, device, epochs=epochs,
                          patience=25, soft_mask=soft_mask, soft_lambda=soft_lambda,
                          early_stop=early_stop)
    return M.embed(model, X_input, device)


def _local_val_idx(train_idx, val_global):
    pos = {g: i for i, g in enumerate(train_idx)}
    return np.array([pos[v] for v in val_global])


def torch_seed(seed):
    import torch
    torch.manual_seed(seed)


def run_cv(data, spec, task_key, condition, device, seeds=(0, 1), n_splits=5, epochs=250,
           return_flat=False, z_dim=64, early_stop=True):
    """Macro-F1 for one model x condition x task via nested donor-CV.

    Returns per-seed means by default; if return_flat, returns a list of
    {seed, fold, f1} for every (seed, fold) so callers can do paired comparisons.
    """
    from .eval import donor_grouped_folds
    y = data["obs"][task_key].astype(str).values
    donor = data["obs"]["donor_id"].astype(str).values
    labels = np.unique(y)                      # fixed class set -> comparable macro-F1 across folds
    results, flat = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if condition.startswith("noise:"):
            p = float(condition.split(":")[1])
            X_input = _thin_counts(data["counts"], p, rng, data["lib_full"])
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
            emb = make_embedder(spec, data, tr_use, device, seed, epochs, X_input,
                                z_dim=z_dim, early_stop=early_stop)
            f1, _ = probe_precomputed(emb, y, donor, [(tr_use, te)], labels=labels)
            fold_scores.append(f1)
            flat.append({"seed": seed, "fold": fi, "f1": f1})
        results.append(float(np.mean(fold_scores)))
    return flat if return_flat else results
