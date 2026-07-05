# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # A toy example: graph-aware vs not-graph-aware
#
# A tiny, fully-controlled illustration of the project's core idea — no real data, just **12 genes,
# 3 transcription factors, 3 cell types**. We *know* the ground truth, so we can see exactly **why**
# a GRN prior can help (it averages co-regulated genes → denoises) and **why the right graph
# matters** (a scrambled graph mixes unrelated genes and the benefit vanishes).
#
# We compare four representations by how well a simple probe recovers the 3 cell types as **noise
# increases**:
#
# | representation | graph-aware? | dim |
# |---|---|---|
# | **raw genes** | no | 12 |
# | **PCA** | no | 3 |
# | **TF-activity (true graph)** | yes | 3 |
# | **TF-activity (rewired graph)** | yes, but wrong wiring | 3 |
#
# This mirrors, in miniature, what we found on the real data: the graph helps as a **TF-activity
# transform** and mainly when the signal is degraded — but only if it's the *real* wiring.

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook"

N_TF = 3            # 3 transcription factors
GPT = 4            # 4 target genes each
N_GENES = N_TF * GPT           # 12 genes
gene_tf = np.repeat(np.arange(N_TF), GPT)   # TRUE graph: gene -> which TF regulates it
print("true graph (gene -> TF):", gene_tf)
print(f"{N_GENES} genes, {N_TF} TFs, each TF regulates {GPT} genes")

# %% [markdown]
# ## The generative story (the "biology" we control)
# There are 3 cell types. **Cell type k is the one where TF k is active** — so in a type-k sample,
# TF k's 4 target genes are elevated, the other 8 are not. Every gene also gets independent
# Gaussian **noise**. So the *true* signal is only 3-dimensional (which TF is on), but we observe
# it through 12 noisy genes.

# %%
def simulate(n_per_type=60, noise=1.0, seed=0):
    r = np.random.default_rng(seed)
    X, y = [], []
    for k in range(N_TF):                       # cell type k <-> TF k active
        signal = (gene_tf == k).astype(float) * 2.0     # that TF's 4 genes elevated
        block = signal + r.normal(0, noise, size=(n_per_type, N_GENES))
        X.append(block)
        y += [k] * n_per_type
    return np.vstack(X), np.array(y)


X, y = simulate(noise=1.5, seed=0)
print("X:", X.shape, " types:", np.bincount(y))

# %% [markdown]
# ## What the raw data looks like (a few samples)
# Rows = samples grouped by type; columns = the 12 genes. You can *see* the block structure — but
# it's noisy at the single-gene level.

# %%
order = np.argsort(y)
fig = px.imshow(X[order], aspect="auto", color_continuous_scale="RdBu_r", zmin=-3, zmax=3,
                labels=dict(x="gene", y="sample (sorted by type)", color="expression"),
                title="raw expression — 12 genes (noisy)")
fig.update_layout(height=420, width=620)
fig

# %% [markdown]
# ## The four representations
# The graph-aware ones **aggregate genes by TF** (mean of a TF's targets) — this is the toy version
# of the TF-activity transform. The rewired graph keeps 4 genes per TF but assigns them **randomly**.

# %%
def tf_activity(X, mapping):
    return np.column_stack([X[:, mapping == t].mean(1) for t in range(N_TF)])

rewired_tf = np.random.default_rng(1).permutation(gene_tf)   # scrambled gene->TF (degree preserved)
print("rewired graph (gene -> TF):", rewired_tf)

def representations(X):
    return {
        "raw genes (12d)":            X,
        "PCA (3d)":                   PCA(3, random_state=0).fit_transform(X),
        "TF-activity TRUE (3d)":      tf_activity(X, gene_tf),
        "TF-activity REWIRED (3d)":   tf_activity(X, rewired_tf),
    }

# %% [markdown]
# ## The experiment: probe accuracy as noise grows
# For each noise level we simulate fresh data, build each representation, and score a logistic-
# regression probe with 5-fold CV. Higher = the representation preserved the cell-type signal.

# %%
def probe_acc(rep, y):
    return cross_val_score(LogisticRegression(max_iter=1000), rep, y, cv=5).mean()

rows = []
for noise in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    Xn, yn = simulate(noise=noise, seed=0)
    for name, rep in representations(Xn).items():
        rows.append(dict(noise=noise, representation=name, accuracy=probe_acc(rep, yn)))
res = pd.DataFrame(rows)

fig = px.line(res, x="noise", y="accuracy", color="representation", markers=True,
              title="cell-type recovery vs noise — graph-aware (true) degrades most gracefully")
fig.add_hline(y=1/3, line_dash="dot", annotation_text="chance (1/3)")
fig.update_layout(height=460, width=760, yaxis_range=[0, 1.02])
fig

# %% [markdown]
# **What to notice:**
# - **TF-activity (true graph)** holds up best as noise rises — averaging 4 co-regulated genes
#   cancels independent noise (√4 = 2× less noise per feature).
# - **Raw genes** degrade faster (every noisy gene is a feature).
# - **TF-activity (rewired)** collapses toward chance — averaging *unrelated* genes destroys the
#   signal. *Same aggregation, wrong wiring → no benefit.* This is the toy version of our
#   rewired-graph control.
# - **PCA** does well too (the signal is low-rank) — echoing the real result that a strong simple
#   baseline is hard to beat.

# %% [markdown]
# ## Seeing it in 2-D at high noise
# At noise = 3, project each representation to 2-D and colour by true type. The true-graph
# TF-activity keeps the three types separable; the rewired one smears them together.

# %%
Xh, yh = simulate(noise=3.0, seed=2)
panels = []
for name, rep in representations(Xh).items():
    p2 = PCA(2, random_state=0).fit_transform(rep) if rep.shape[1] > 2 else rep[:, :2]
    d = pd.DataFrame({"x": p2[:, 0], "y": p2[:, 1], "type": yh.astype(str), "rep": name})
    panels.append(d)
proj = pd.concat(panels)
fig = px.scatter(proj, x="x", y="y", color="type", facet_col="rep", facet_col_wrap=2,
                 title="2-D projection at noise=3 — true graph separates types; rewired smears them",
                 opacity=0.7)
fig.update_xaxes(matches=None, showticklabels=False); fig.update_yaxes(matches=None, showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=640, width=820)
fig

# %% [markdown]
# ## Takeaway (maps directly to the real project)
# - A GRN prior used as a **feature transform** (aggregate co-regulated genes) buys **robustness to
#   noise** — exactly the "helps most when data is degraded / scarce" result on the real data.
# - The benefit is **real biology, not just dimension reduction**: the *rewired* graph has the same
#   shape and dimension but the wrong wiring, and it fails — the analogue of `dc_tfact` beating
#   `dc_tfact_rewired`.
# - A strong simple baseline (**PCA**) is still very competitive — the real-data theme that
#   "the graph rarely beats PCA except in the hard (low-data/noisy) regime."
#
# This toy has no learned encoder, but the intuition transfers: aggregating along the *correct*
# regulatory graph denoises; along a *wrong* graph it destroys signal — which is why *which* way you
# use the graph, and whether it's the real one, matters more than simply "using a graph."
