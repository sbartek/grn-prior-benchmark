"""Step 4 — encoders that map a pseudobulk profile to an embedding.

Two encoders, identical except for the FIRST encoder layer (this isolates the graph effect):

  baseline : gene --[dense]---------------> TF-width hidden --> z   (unconstrained)
  grn      : gene --[masked, signed]------> TF activities    --> z   (DoRothEA regulons)

The masked layer's effective weight is  mask * sign * softplus(raw)  so each hidden unit is a
TF whose activity is a non-negative-magnitude, sign-constrained sum over its regulon. Feeding a
corrupted graph (rewired / sign-shuffled / random) into the same class gives the control models
at identical parameter count -> the decisive "is it the biology, or just sparsity?" test.

Decoder and bottleneck are dense and identical across models. Objective = MSE reconstruction of
the log-normalised profile (self-supervised, no labels). Embedding = the bottleneck z.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MaskedLinear(nn.Module):
    """Linear layer restricted to a fixed sparsity mask, with fixed DoRothEA edge signs.

    effective weight = mask * sign * softplus(raw)   (out_features x in_features)
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, sign: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)          # (out, in) 0/1
        self.register_buffer("sign", sign)          # (out, in) -1/+1
        self.raw = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w = self.mask * self.sign * F.softplus(self.raw)
        return F.linear(x, w, self.bias)


class AutoEncoder(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int, z_dim: int,
                 mask: torch.Tensor | None = None, sign: torch.Tensor | None = None,
                 dec_mask: torch.Tensor | None = None, dec_sign: torch.Tensor | None = None):
        super().__init__()
        # enc1 gene->TF (encoder mask); dec2 TF->gene (decoder mask, expiMap-style = the causal
        # generative direction). Masking either/both gives encoder-only / decoder-only / symmetric.
        self.enc1 = MaskedLinear(n_genes, n_hidden, mask, sign) if mask is not None \
            else nn.Linear(n_genes, n_hidden)
        self.to_z = nn.Linear(n_hidden, z_dim)
        self.dec1 = nn.Linear(z_dim, n_hidden)
        self.dec2 = MaskedLinear(n_hidden, n_genes, dec_mask, dec_sign) if dec_mask is not None \
            else nn.Linear(n_hidden, n_genes)

    def encode(self, x):
        return self.to_z(F.relu(self.enc1(x)))

    def forward(self, x):
        z = self.encode(x)
        return self.dec2(F.relu(self.dec1(z))), z


def build_mask(graph: dict, name: str, genes_order: np.ndarray, n_hidden: int, device):
    """Build (n_hidden x n_genes) mask + sign tensors for a named graph variant.

    genes_order: the gene symbols in the order the model input uses (must match graph['genes']).
    """
    n_genes = len(genes_order)
    mask = np.zeros((n_hidden, n_genes), dtype=np.float32)
    sign = np.zeros((n_hidden, n_genes), dtype=np.float32)
    rows = graph[f"{name}_rows"]      # gene index
    cols = graph[f"{name}_cols"]      # tf index (hidden unit)
    signs = graph[f"{name}_signs"]
    mask[cols, rows] = 1.0
    sign[cols, rows] = signs
    sign[sign == 0] = 1.0             # safety
    return (torch.tensor(mask, device=device), torch.tensor(sign, device=device))


def train_ae(model, X, val_idx, device, epochs=300, lr=1e-3, wd=1e-4, patience=30,
             soft_mask=None, soft_lambda=0.0, early_stop=True, verbose=False):
    """Train an autoencoder.

    early_stop=True (default): hold out `val_idx` and early-stop on its reconstruction MSE.
    early_stop=False: FIXED budget -- train `epochs` on ALL rows, no val carve-out, no checkpoint
    selection. Avoids selecting on a noisy train-donor val slice and uses the full training data;
    justified once training is known to have converged by that budget.

    soft_mask/soft_lambda: SOFT graph prior. On a dense first layer, add a penalty
    soft_lambda * ||W ⊙ (1 - mask)||^2 that shrinks off-regulon weights toward zero without
    removing them. soft_lambda=0 -> baseline; soft_lambda->inf -> approaches the hard mask.
    """
    model = model.to(device)
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    if soft_mask is not None:
        soft_mask = soft_mask.to(device)
        off = 1.0 - soft_mask
    n = X.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if not early_stop:                                  # fixed-budget: train on ALL rows
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            recon, _ = model(X)
            loss = F.mse_loss(recon, X)
            if soft_lambda > 0 and soft_mask is not None:
                loss = loss + soft_lambda * (model.enc1.weight * off).pow(2).sum()
            loss.backward()
            opt.step()
        return model, float(loss.item())

    mask_val = torch.zeros(n, dtype=torch.bool, device=device)
    mask_val[val_idx] = True
    Xtr, Xva = X[~mask_val], X[mask_val]

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        recon, _ = model(Xtr)
        loss = F.mse_loss(recon, Xtr)
        if soft_lambda > 0 and soft_mask is not None:
            loss = loss + soft_lambda * (model.enc1.weight * off).pow(2).sum()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vrecon, _ = model(Xva)
            vloss = F.mse_loss(vrecon, Xva).item()
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"    ep{ep:4d} train={loss.item():.4f} val={vloss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def embed(model, X, device):
    model.eval()
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    return model.encode(X).cpu().numpy()


def pca_embedding(X, z_dim=64, seed=0):
    from sklearn.decomposition import PCA
    return PCA(n_components=z_dim, random_state=seed).fit_transform(np.asarray(X))
