"""Static (matplotlib) versions of the toy-notebook diagrams for the docs Models wiki:
a many-to-many signed GRN, and the four autoencoder architectures. Writes to docs/img/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
N_TF, GPT, N_GENES = 3, 4, 12
gene_tf = np.repeat(np.arange(N_TF), GPT)
PAL = ["#e41a1c", "#377eb8", "#4daf4a"]


def make_W():
    W = np.zeros((N_TF, N_GENES))
    for g in range(N_GENES):
        W[gene_tf[g], g] = 1.0                       # primary activating edge
    r = np.random.default_rng(5)
    for g in range(N_GENES):
        sec = (gene_tf[g] + r.integers(1, N_TF)) % N_TF
        W[sec, g] = r.choice([0.5, -0.5])            # secondary, sometimes repressing
    return W


W = make_W()
tf_y, gene_y = np.linspace(0, 1, N_TF), np.linspace(0, 1, N_GENES)

# ---- many-to-many signed graph ----
fig, ax = plt.subplots(figsize=(6, 5))
for t in range(N_TF):
    for g in range(N_GENES):
        w = W[t, g]
        if w == 0:
            continue
        ax.plot([0, 1], [tf_y[t], gene_y[g]], color="crimson" if w > 0 else "royalblue",
                lw=2.2 if abs(w) == 1 else 1.1, alpha=0.75, zorder=1)
ax.scatter([0] * N_TF, tf_y, s=650, c=PAL, zorder=2)
for t in range(N_TF):
    ax.text(-0.06, tf_y[t], f"TF{t}", ha="right", va="center", fontweight="bold")
ax.scatter([1] * N_GENES, gene_y, s=180, c=[PAL[t] for t in gene_tf], zorder=2)
for g in range(N_GENES):
    ax.text(1.05, gene_y[g], f"g{g}", ha="left", va="center", fontsize=8)
ax.legend([Line2D([0], [0], color="crimson", lw=2), Line2D([0], [0], color="royalblue", lw=2)],
          ["activates (+)", "represses (−)"], loc="upper center", ncol=2, fontsize=9,
          bbox_to_anchor=(0.5, 1.08), frameon=False)
ax.set_xlim(-0.35, 1.45); ax.set_ylim(-0.05, 1.15); ax.axis("off")
ax.set_title("Many-to-many, signed GRN  (each gene regulated by 2 TFs)", pad=18)
fig.tight_layout(); fig.savefig(IMG / "toy_manytomany.png", dpi=130); plt.close(fig)

# ---- four autoencoder architectures ----
sizes = [N_GENES, N_TF, 2, N_TF, N_GENES]
xs = [0, 1, 2, 3, 4]
ys = [np.linspace(0.05, 0.95, s) for s in sizes]


def draw(ax, enc, dec, title):
    dense = [1, 2]
    if not enc:
        dense = [0] + dense
    if not dec:
        dense = dense + [3]
    for li in dense:
        for a in range(sizes[li]):
            for b in range(sizes[li + 1]):
                ax.plot([xs[li], xs[li + 1]], [ys[li][a], ys[li + 1][b]],
                        color="lightgray", lw=0.4, zorder=1)

    def color(src, dst, gene_src):
        for t in range(N_TF):
            for g in range(N_GENES):
                if W[t, g] == 0:
                    continue
                c = "crimson" if W[t, g] > 0 else "royalblue"
                yy = ([ys[src][g], ys[dst][t]] if gene_src else [ys[src][t], ys[dst][g]])
                ax.plot([xs[src], xs[dst]], yy, color=c, lw=1.0, zorder=1)
    if enc:
        color(0, 1, True)
    if dec:
        color(3, 4, False)
    for li in range(5):
        ax.scatter([xs[li]] * sizes[li], ys[li], s=45, c="#555", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(["genes", "TF", "z", "TF", "genes"], fontsize=7)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
draw(axes[0, 0], False, False, "baseline: dense (no graph)")
draw(axes[0, 1], True, False, "grn_real: ENCODER masked (gene→TF)")
draw(axes[1, 0], False, True, "grn_decoder: DECODER masked (TF→gene, expiMap)")
draw(axes[1, 1], True, True, "grn_symmetric: BOTH masked")
fig.suptitle("Autoencoders on the toy:  genes(12) → TF(3) → z(2) → TF(3) → genes(12) "
             "— only the coloured (graph) edges differ", fontsize=11)
fig.tight_layout(); fig.savefig(IMG / "toy_nn_architectures.png", dpi=130); plt.close(fig)

# ---- the winner: dc_tfact = a FIXED 2-layer transform (genes -> TF-activity) ----
fig, ax = plt.subplots(figsize=(6.5, 4.6))
gy = np.linspace(0.05, 0.95, N_GENES)
ty = np.linspace(0.2, 0.8, N_TF)
for t in range(N_TF):
    for g in range(N_GENES):
        w = W[t, g]
        if w == 0:
            continue
        ax.plot([0, 1], [gy[g], ty[t]], color="crimson" if w > 0 else "royalblue",
                lw=2.0 if abs(w) == 1 else 1.0, alpha=0.75, zorder=1)
ax.scatter([0] * N_GENES, gy, s=140, c=[PAL[t] for t in gene_tf], zorder=2)
for g in range(N_GENES):
    ax.text(-0.05, gy[g], f"g{g}", ha="right", va="center", fontsize=8)
ax.scatter([1] * N_TF, ty, s=700, c=PAL, zorder=2)
for t in range(N_TF):
    ax.text(1.06, ty[t], f"TF{t} activity", ha="left", va="center", fontweight="bold", fontsize=9)
ax.text(0, 1.05, "genes (12)", ha="center", fontsize=10)
ax.text(1, 0.95, "TF-activity (3)\n= the embedding", ha="center", fontsize=10)
ax.text(0.5, -0.08, "weights FIXED by the graph (no training) · output IS the embedding "
        "(no z, no decoder)", ha="center", fontsize=9, style="italic")
ax.legend([Line2D([0], [0], color="crimson", lw=2), Line2D([0], [0], color="royalblue", lw=2)],
          ["activates (+)", "represses (−)"], loc="upper center", ncol=2, fontsize=8,
          bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.set_xlim(-0.35, 1.6); ax.set_ylim(-0.2, 1.1); ax.axis("off")
ax.set_title("dc_tfact (the winner): a fixed genes → TF-activity transform", pad=12)
fig.tight_layout(); fig.savefig(IMG / "toy_tfact_transform.png", dpi=130); plt.close(fig)
print("wrote toy_manytomany.png, toy_nn_architectures.png, toy_tfact_transform.png")
