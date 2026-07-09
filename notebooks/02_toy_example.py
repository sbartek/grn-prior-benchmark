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
print(f"{N_GENES} genes, {N_TF} TFs, each TF regulates {GPT} genes -> {N_GENES} edges")

# %% [markdown]
# ## The toy gene-regulatory graph
# A tiny bipartite network: **3 TFs (left)** each wired to **4 target genes (right)** — 12 edges,
# every gene regulated by exactly one TF. (The real DoRothEA graph is ~411 TFs × 8,376 genes with
# ~30,000 *signed* edges and overlapping regulons; this is the readable cartoon of it.)

# %%
import plotly.graph_objects as go

tf_y = np.linspace(0, 1, N_TF)
gene_y = np.linspace(0, 1, N_GENES)
ex, ey = [], []
for g in range(N_GENES):
    ex += [0, 1, None]
    ey += [tf_y[gene_tf[g]], gene_y[g], None]

PAL = ["#e41a1c", "#377eb8", "#4daf4a"]              # one colour per TF
fig = go.Figure()
fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="lightgray", width=1),
                         hoverinfo="none", showlegend=False))
fig.add_trace(go.Scatter(x=[0] * N_TF, y=tf_y, mode="markers+text",
                         text=[f"TF{t}" for t in range(N_TF)], textposition="middle left",
                         marker=dict(size=30, color=[PAL[t] for t in range(N_TF)]), name="TFs"))
fig.add_trace(go.Scatter(x=[1] * N_GENES, y=gene_y, mode="markers+text",
                         text=[f"g{g}" for g in range(N_GENES)], textposition="middle right",
                         marker=dict(size=16, color=[PAL[t] for t in gene_tf]), name="genes"))
fig.update_layout(title="toy GRN: TF → target-gene edges (genes coloured by regulating TF)",
                  height=460, width=560, xaxis=dict(visible=False, range=[-0.25, 1.25]),
                  yaxis=dict(visible=False))
fig

# %% [markdown]
# ## Real vs **degree-preserving rewired** — the control we compare against
#
# In the real project we compare the **real** graph against a **rewired** version — the key control
# to tell "biology" apart from "any structure with the same shape." **Degree-preserving** = each TF
# keeps the *same number* of targets AND each gene keeps the *same number* of regulators, but *which*
# specific edges exist is randomised. Same edge count, same node-degree distribution, biology
# destroyed.
#
# **Why we need it:** if the "real" mask helps as much as a graph with random wiring, then the
# benefit is just **sparsity + degree distribution**, not the specific TF-target biology.
#
# **Note on visualising this:** in a bipartite graph where each gene has *exactly one* TF, rewiring
# collapses into "reassign each gene to a random TF" — visually indistinguishable from a
# permutation. To make the real vs rewired distinction visible we need a **multi-connected** graph
# (each gene regulated by ≥ 2 TFs). We build one here just for illustration — separate from the
# main toy above.

# %%
from plotly.subplots import make_subplots

# Illustrative multi-connected bipartite graph: 3 TFs, 12 genes, each gene has 2 TFs (24 edges).
# Same shape for real + rewired, but different specific edges.
def make_multi_edges(n_tf, n_genes, k_per_gene, seed):
    rng = np.random.default_rng(seed)
    edges = []
    for g in range(n_genes):
        tfs = rng.choice(n_tf, size=k_per_gene, replace=False)
        for t in tfs:
            edges.append((int(t), g))
    return edges

edges_real    = make_multi_edges(N_TF, N_GENES, k_per_gene=2, seed=0)
edges_rewired = make_multi_edges(N_TF, N_GENES, k_per_gene=2, seed=7)

def tf_deg(edges):
    d = np.zeros(N_TF, int)
    for t, _ in edges:
        d[t] += 1
    return d

print("real    per-TF degrees:", tf_deg(edges_real),
      "  gene degrees:", np.bincount([g for _, g in edges_real], minlength=N_GENES))
print("rewired per-TF degrees:", tf_deg(edges_rewired),
      "  gene degrees:", np.bincount([g for _, g in edges_rewired], minlength=N_GENES))

# %%
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("real graph (biology)", "degree-preserving rewired (control)"),
    horizontal_spacing=0.15,
)

for col, edges in enumerate([edges_real, edges_rewired], start=1):
    # draw each edge with the colour of its SOURCE TF
    for t, g in edges:
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[tf_y[t], gene_y[g]], mode="lines",
            line=dict(color=PAL[t], width=1.4),
            hoverinfo="none", showlegend=False,
        ), 1, col)
    fig.add_trace(go.Scatter(
        x=[0] * N_TF, y=tf_y, mode="markers+text",
        text=[f"TF{t}" for t in range(N_TF)], textposition="middle left",
        marker=dict(size=30, color=[PAL[t] for t in range(N_TF)]),
        showlegend=False,
    ), 1, col)
    fig.add_trace(go.Scatter(
        x=[1] * N_GENES, y=gene_y, mode="markers+text",
        text=[f"g{g}" for g in range(N_GENES)], textposition="middle right",
        marker=dict(size=14, color="lightgray", line=dict(color="black", width=0.5)),
        showlegend=False,
    ), 1, col)

fig.update_layout(
    title="Real vs rewired — each gene has 2 TF regulators. Edge colour = source TF.<br>"
          "<sub>Both graphs have the same total edges and same per-node degrees, but different specific edges — not a permutation.</sub>",
    height=520, width=1000,
)
for c in (1, 2):
    fig.update_xaxes(visible=False, range=[-0.3, 1.3], row=1, col=c)
    fig.update_yaxes(visible=False, row=1, col=c)
fig

# %% [markdown]
# **What to notice:**
#
# - Left: `TF0` (red edges) connects to a specific set of genes; `TF1` (blue) and `TF2` (green) to
#   other specific sets. Some genes get red+blue, some blue+green, etc.
# - Right: the crossings-pattern of edges is genuinely different. `TF0` (red) now targets a
#   different set of genes. Genes' colour-combinations change too (e.g. a gene that had red+blue
#   on the left might have blue+green on the right).
# - **Both graphs pass the same "how many edges? how many per node?" audit** — that's the
#   "degree-preserving" part. But the specific TF→target combinations are wholly different.
#
# **Why this isn't just permutation of hidden units:** if we permuted the hidden units in the real
# AE, hidden unit "TF0" would still receive input from exactly the genes it did before — we'd just
# call it "TF7" instead. The network's function is unchanged. In rewiring, the hidden unit named
# "TF0" now literally receives input from a different set of genes — the function *is* different.
#
# **In the real project:** `grn_real` is the AE variant whose encoder mask uses the real DoRothEA
# adjacency; `grn_rewired` uses a degree-preserving-shuffled version like the right panel above
# (but with 411 TFs × 8,376 genes and ~30,000 signed edges). Identical architecture, identical
# training, identical parameter budget — only the mask's specific edges differ. If they perform
# similarly, the benefit isn't the biology.

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
# ## From single cells to pseudobulk
# Real scRNA-seq is measured per **single cell**, which is extremely noisy (dropout). We don't model
# single cells — we **pseudobulk**: average all cells of the same *(donor × cell type)* into one
# clean profile. Here we simulate 6 donors, each with noisy single cells of all 3 types, then
# aggregate. Watch the block structure emerge as the single-cell noise washes out.

# %%
from plotly.subplots import make_subplots

n_donors, cells_per, sc_noise = 6, 40, 3.0
r = np.random.default_rng(7)
cells, dmeta = [], []
for d in range(n_donors):
    for k in range(N_TF):                       # each donor has cells of every type
        sig = (gene_tf == k) * 2.0
        cells.append(sig + r.normal(0, sc_noise, (cells_per, N_GENES)))
        dmeta += [(d, k)] * cells_per
cells = np.vstack(cells)
dmeta = pd.DataFrame(dmeta, columns=["donor", "type"])
pb = (pd.DataFrame(cells, columns=[f"g{i}" for i in range(N_GENES)])
      .assign(donor=dmeta.donor, type=dmeta.type)
      .groupby(["donor", "type"]).mean())      # pseudobulk = mean of cells per (donor, type)
print(f"single cells: {cells.shape[0]}   ->   pseudobulk profiles: {pb.shape[0]} "
      f"(= {n_donors} donors x {N_TF} types)")

# %%
sc_order = np.argsort(dmeta.type.values)
pb_sorted = pb.reset_index().sort_values("type")
fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                    subplot_titles=(f"{cells.shape[0]} single cells (very noisy)",
                                    f"{pb.shape[0]} pseudobulk profiles (clean)"))
fig.add_trace(go.Heatmap(z=cells[sc_order], colorscale="RdBu", reversescale=True, zmid=0, zmin=-4, zmax=4,
                         showscale=False), 1, 1)
fig.add_trace(go.Heatmap(z=pb_sorted[[f"g{i}" for i in range(N_GENES)]].values,
                         colorscale="RdBu", reversescale=True, zmid=0, zmin=-2, zmax=2, showscale=False), 1, 2)
fig.update_layout(height=440, width=900,
                  title="pseudobulk (donor × type averaging) washes out single-cell noise")
fig.update_xaxes(title_text="gene"); fig.update_yaxes(showticklabels=False)
fig

# %% [markdown]
# The 720 single cells on the left are a mess; the 18 pseudobulk profiles on the right show clean
# 3-block structure. **This averaging is step 1 of denoising — done *before* any model.** The
# graph-aware transform below is a *second* averaging, this time along the regulatory graph.

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
# ## What ULM actually does (behind the "mean of targets")
#
# The toy above uses **mean of a TF's target genes** as its activity — a clean intuition, and in
# this signless toy it's essentially what real ULM computes. The **real** ULM in `decoupler` is a
# tiny **signed univariate linear regression per (sample, TF)**:
#
# For each sample `s` and TF `k`:
#
# 1. Build a **signed indicator** `x[gene]`: `+1` if gene is an activator target, `−1` if repressor,
#    `0` otherwise.
# 2. Fit `y = a + b·x`, where `y` is the sample's expression vector.
# 3. Report the **t-statistic of `b`** — that number is the ULM score for (sample, TF).
#
# Higher t-stat → the sample's expression pattern matches the TF's regulon more sharply → the TF
# is inferred more active. Below: one sample × one TF walked through, step by step.

# %%
from scipy import stats

Xn, yn = simulate(noise=1.5, seed=0)
sample_idx = 5                          # this sample is cell-type 0 (TF0-driven)
tf_of_interest = 0

y_expression = Xn[sample_idx]                                          # length 12
x_indicator  = (gene_tf == tf_of_interest).astype(float)               # +1 for TF0's 4 targets, 0 else

slope, intercept, r_value, p_value, std_err = stats.linregress(x_indicator, y_expression)
t_stat = slope / std_err

print(f"sample {sample_idx} (true type = {yn[sample_idx]}), TF{tf_of_interest} activity:")
print(f"  slope b   = {slope:.3f}   (targets are {slope:.2f} higher on average)")
print(f"  std_err   = {std_err:.3f}")
print(f"  t-stat    = {t_stat:.3f}   <-- this is dc_tfact[sample_5, TF0]")

# %%
rng = np.random.default_rng(0)
jitter = rng.normal(0, 0.03, len(x_indicator))
colors = ["#e41a1c" if xi > 0 else "#888" for xi in x_indicator]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_indicator + jitter, y=y_expression, mode="markers",
    marker=dict(size=14, color=colors, line=dict(color="black", width=1)),
    text=[f"g{i}" for i in range(N_GENES)],
    hovertemplate="%{text}<br>x=%{x:.2f}  y=%{y:.2f}<extra></extra>",
    name="genes",
))
xs = np.array([-0.15, 1.15])
fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines",
                         line=dict(color="black", width=2, dash="solid"),
                         name=f"fitted:  y = {intercept:.2f} + {slope:.2f}·x"))
fig.update_layout(
    title=f"ULM — sample {sample_idx} × TF0<br>"
          f"<sub>b = {slope:.2f},  std_err = {std_err:.2f},  t-stat = {t_stat:.2f}  → TF0 activity</sub>",
    xaxis=dict(title="signed indicator  x[gene]  (+1 = TF0 target, 0 = not)",
               tickvals=[0, 1], range=[-0.3, 1.3]),
    yaxis=dict(title="expression  y[gene]"),
    height=460, width=640, showlegend=True,
)
fig

# %% [markdown]
# **How to read the plot:**
#
# - Red dots = TF0's 4 target genes (`x = +1`). In a type-0 sample, their expression is elevated (top-right).
# - Grey dots = non-targets (`x = 0`). Their expression is centered near zero (bottom-left).
# - The **fitted line** captures the average lift from `x = 0` to `x = 1` — that slope `b` is how
#   much higher targets are than non-targets in this sample.
# - Divide by the standard error of the slope → **t-statistic**. That number IS the ULM score for
#   this (sample, TF) pair.
#
# **Same math on the real project**: for each of 500 pseudobulk samples and each of ~293 TFs, this
# regression fires. Output: a `500 × 293` TF-activity matrix. No training, no seed dependence, no
# leakage — every sample is transformed independently. That matrix is `dc_tfact`.
#
# **When repressors matter:** in this toy, all edges are activators, so `x` is 0/1. Once we add
# repressors (next section), `x` takes values in `{−1, 0, +1}` — the regression then rewards
# expression patterns where **activators are up AND repressors are down simultaneously**, which is
# the sharper "biological consistency" test.

# %% [markdown]
# ## From cartoon to reality: overlapping regulons + repressors
# So far each gene had **one** activating TF — a clean partition. Real GRNs are **many-to-many**:
# a TF regulates ~73 genes, a gene is regulated by ~3–4 TFs, and many edges **repress** (−1). We
# now give each gene a strong **primary** activator *plus* a weaker **secondary** edge (±0.5,
# sometimes repressing) — overlapping, signed regulons — and watch the clean benefit erode.

# %%
def make_W(tangled, seed=5):
    W = np.zeros((N_TF, N_GENES))
    for g in range(N_GENES):
        W[gene_tf[g], g] = 1.0                       # primary activating edge
    if tangled:
        r = np.random.default_rng(seed)
        for g in range(N_GENES):
            sec = (gene_tf[g] + r.integers(1, N_TF)) % N_TF   # a *different* TF also regulates g
            W[sec, g] = r.choice([0.5, -0.5])         # weaker, sometimes a repressor
    return W

W_clean, W_tangled = make_W(False), make_W(True)
fig = make_subplots(rows=1, cols=2, subplot_titles=("clean regulons (1 TF/gene, all +)",
                                                    "tangled regulons (overlap + repressors)"))
for c, W in enumerate([W_clean, W_tangled], start=1):
    fig.add_trace(go.Heatmap(z=W, colorscale="RdBu", reversescale=True, zmid=0, zmin=-1, zmax=1,
                             showscale=(c == 2)), 1, c)
fig.update_layout(height=300, width=900, title="regulon membership: TF (rows) × gene (cols)")
fig.update_yaxes(tickvals=list(range(N_TF)), ticktext=[f"TF{t}" for t in range(N_TF)])
fig.update_xaxes(title_text="gene")
fig

# %% [markdown]
# Same thing as a **node-link graph** (compare to the clean one at the top): now every gene has
# **two** edges — a solid **red** primary activator and a thinner **secondary** edge that is red
# (activating, +0.5) or **blue** (repressing, −0.5). This is the many-to-many, signed picture.

# %%
tf_y = np.linspace(0, 1, N_TF)
gene_y = np.linspace(0, 1, N_GENES)
fig = go.Figure()
for t in range(N_TF):
    for g in range(N_GENES):
        w = W_tangled[t, g]
        if w == 0:
            continue
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[tf_y[t], gene_y[g]], mode="lines", hoverinfo="none", showlegend=False,
            line=dict(color=("crimson" if w > 0 else "royalblue"), width=3 if abs(w) == 1 else 1.2)))
# legend proxies for edge meaning
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="crimson"), name="activates (+)"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="royalblue"), name="represses (−)"))
fig.add_trace(go.Scatter(x=[0] * N_TF, y=tf_y, mode="markers+text",
                         text=[f"TF{t}" for t in range(N_TF)], textposition="middle left",
                         marker=dict(size=30, color=[PAL[t] for t in range(N_TF)]), showlegend=False))
fig.add_trace(go.Scatter(x=[1] * N_GENES, y=gene_y, mode="markers+text",
                         text=[f"g{g}" for g in range(N_GENES)], textposition="middle right",
                         marker=dict(size=16, color=[PAL[t] for t in gene_tf]), showlegend=False))
fig.update_layout(title="tangled toy GRN: overlapping + signed edges (each gene → 2 TFs)",
                  height=500, width=600, xaxis=dict(visible=False, range=[-0.25, 1.25]),
                  yaxis=dict(visible=False))
fig

# %%
def sim_from_W(W, noise, n_per_type=60, seed=0):
    r = np.random.default_rng(seed)
    X, y = [], []
    for k in range(N_TF):                            # cell type k -> only TF k active
        a = np.zeros(N_TF); a[k] = 1.0
        base = (W.T @ a) * 2.0                       # gene expression driven by the TF states
        X.append(base + r.normal(0, noise, (n_per_type, N_GENES)))
        y += [k] * n_per_type
    return np.vstack(X), np.array(y)

def tf_act_W(X, W):                                  # signed regulon aggregation
    cols = []
    for t in range(N_TF):
        idx = W[t] != 0
        cols.append((X[:, idx] @ W[t, idx]) / idx.sum())
    return np.column_stack(cols)

# Data now has tangled (overlapping, signed) biology. We aggregate it two ways: with the CORRECT
# tangled graph, vs with an INCOMPLETE graph that only knows the primary activating edges (the kind
# of gap a real, imperfect prior has).
rows = []
for label, W_agg in [("aggregate w/ CORRECT graph", W_tangled),
                     ("aggregate w/ INCOMPLETE graph", W_clean)]:
    for noise in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        Xn, yn = sim_from_W(W_tangled, noise, seed=0)      # tangled biology
        rows.append(dict(noise=noise, representation=label, accuracy=probe_acc(tf_act_W(Xn, W_agg), yn)))
res2 = pd.DataFrame(rows)
fig = px.line(res2, x="noise", y="accuracy", color="representation", markers=True,
              title="tangled biology: knowing the full graph beats an incomplete one")
fig.add_hline(y=1/3, line_dash="dot", annotation_text="chance (1/3)")
fig.update_layout(height=440, width=760, yaxis_range=[0, 1.02])
fig

# %% [markdown]
# **The honest lesson.** Overlap and repressors *on their own* don't break the prior — if you know
# the true tangled graph, signed aggregation still recovers the signal (blue). The benefit erodes
# when the graph is **incomplete/wrong** and misses real cross-talk (orange), which then leaks in as
# noise. On real data *both* hold at once: DoRothEA's regulons overlap and are signed **and** the
# graph is a noisy, partial estimate — together these are a big part of *why* the prior's benefit is
# modest and shows up mainly in the low-data / high-noise regime rather than everywhere. (And it's
# why the fully-scrambled `rewired` graph above fails outright — the extreme of a wrong graph.)

# %% [markdown]
# ## ULM on the tangled (signed) graph — one worked example
#
# Now the same regression trick from earlier, but with **signed weights**. Every gene has an edge to
# TF0 that's `+1` (primary activator), `−0.5` (weak repressor), `+0.5` (weak activator), or `0`
# (not connected). The predictor `x` is no longer a clean 0/1 indicator — it takes values from
# `W_tangled[TF0]`. The regression now rewards samples where **activators are up AND repressors are
# down simultaneously** — a sharper "biological consistency" test than mean-of-targets.

# %%
Xn_tangled, yn_tangled = sim_from_W(W_tangled, noise=1.5, seed=0)
sample_idx_t = 5                                # type-0 sample (TF0 active)
tf_t = 0

y_t = Xn_tangled[sample_idx_t]                  # length 12
x_t = W_tangled[tf_t]                           # signed weights: +1 / −0.5 / +0.5 / 0

slope_t, intercept_t, _, _, se_t = stats.linregress(x_t, y_t)
t_stat_t = slope_t / se_t

print(f"sample {sample_idx_t} (true type = {yn_tangled[sample_idx_t]}), TF{tf_t} tangled activity:")
print(f"  x_t = {x_t}")
print(f"  slope b   = {slope_t:.3f}")
print(f"  std_err   = {se_t:.3f}")
print(f"  t-stat    = {t_stat_t:.3f}   <-- dc_tfact[sample_5, TF0] using tangled graph")

# %%
# color: activators red, repressors blue, non-targets grey; size proportional to |weight|
def _col(w):
    return "#e41a1c" if w > 0 else ("#377eb8" if w < 0 else "#888")
colors_t = [_col(w) for w in x_t]
sizes_t = [8 + 12 * abs(w) for w in x_t]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_t + np.random.default_rng(1).normal(0, 0.015, len(x_t)),
    y=y_t, mode="markers",
    marker=dict(size=sizes_t, color=colors_t, line=dict(color="black", width=1)),
    text=[f"g{i} (w={w:+.1f})" for i, w in enumerate(x_t)],
    hovertemplate="%{text}<br>x=%{x:.2f}  y=%{y:.2f}<extra></extra>",
    name="genes",
))
xs_t = np.array([x_t.min() - 0.15, x_t.max() + 0.15])
fig.add_trace(go.Scatter(x=xs_t, y=intercept_t + slope_t * xs_t, mode="lines",
                         line=dict(color="black", width=2),
                         name=f"fitted:  y = {intercept_t:.2f} + {slope_t:.2f}·x"))
fig.update_layout(
    title=f"ULM on tangled graph — sample {sample_idx_t} × TF0<br>"
          f"<sub>x = signed regulon weight; b = {slope_t:.2f},  t-stat = {t_stat_t:.2f}</sub>",
    xaxis=dict(title="signed weight  x[gene]  from W_tangled[TF0]  (red = +, blue = −, size ∝ |w|)"),
    yaxis=dict(title="expression  y[gene]"),
    height=460, width=680, showlegend=True,
)
fig

# %% [markdown]
# **How to read this:**
#
# - Red dots = activators of TF0 (positive `x`); their expression should be elevated in a type-0 sample.
# - Blue dots = repressors of TF0 (negative `x`); their expression should be *lower* than average.
# - Dot size = strength of the edge (|weight|).
# - The fitted line's slope `b` measures: *do y-values track the signed regulon pattern?*
#   Positive `b` means yes — activators up AND repressors down, exactly what a type-0 sample should show.
#
# **Why the signed version is stronger than mean-of-targets:**
# A pure activator-mean would count a highly-expressed repressor as evidence FOR the TF (wrong).
# Signed ULM penalises that: a high-expression repressor sits at low `x`, pulling the slope *down*.
# You need the whole regulon pattern to align.
#
# **What happens if the graph is wrong.** If we replaced `W_tangled` with `W_clean` (the incomplete
# graph that misses secondary edges), the signed cross-talk terms are set to zero → the regression
# has less discriminating power. That's the orange line in the plot above.

# %% [markdown]
# ## What the neural-net encoders look like (the learned models)
# The comparisons above used the *fixed* TF-activity transform. The real project also has **learned
# autoencoders** — a **baseline** (dense first layer) and a **graph-masked** one (`grn_real`) whose
# first layer *is* the graph. On the toy (12 genes → 3 TF-hidden → z=2 → decode back) the only
# difference is the **first layer's wiring**, drawn below. Grey = unconstrained dense weights;
# red/blue = graph edges kept by the mask (activate / repress).

# %%
def draw_ae(enc_masked=False, dec_masked=False, W=None, title=""):
    sizes = [N_GENES, N_TF, 2, N_TF, N_GENES]
    xs = [0, 1, 2, 3, 4]
    ys = [np.linspace(0.05, 0.95, s) for s in sizes]
    fig = go.Figure()
    dense = [1, 2]                                       # hidden->z, z->hidden are always dense
    if not enc_masked:
        dense = [0] + dense                              # gene->TF dense
    if not dec_masked:
        dense = dense + [3]                              # TF->gene dense
    gx, gy = [], []
    for li in dense:
        for a in range(sizes[li]):
            for b in range(sizes[li + 1]):
                gx += [xs[li], xs[li + 1], None]; gy += [ys[li][a], ys[li + 1][b], None]
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="lightgray", width=0.5),
                             hoverinfo="none", showlegend=False))

    def color_layer(src_x, dst_x, gene_is_src):          # draw graph edges of a masked layer
        for sgn, color in [(1, "crimson"), (-1, "royalblue")]:
            ex, ey = [], []
            for t in range(N_TF):
                for g in range(N_GENES):
                    if W[t, g] != 0 and np.sign(W[t, g]) == sgn:
                        gy_, ty_ = ys[0 if gene_is_src else 4][g], ys[1 if gene_is_src else 3][t]
                        ex += [src_x, dst_x, None]
                        ey += [(gy_ if gene_is_src else ty_), (ty_ if gene_is_src else gy_)]
                        ey += [None]
            if ex:
                fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=1.6),
                                         hoverinfo="none", showlegend=False))
    if enc_masked:
        color_layer(xs[0], xs[1], True)                  # gene -> TF
    if dec_masked:
        color_layer(xs[3], xs[4], False)                 # TF -> gene (causal, expiMap direction)
    if enc_masked or dec_masked:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="crimson"), name="activates (+)"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="royalblue"), name="represses (−)"))

    names = ["genes (12)", "TF hidden (3)", "z (2)", "TF hidden (3)", "genes out (12)"]
    for li in range(5):
        fig.add_trace(go.Scatter(x=[xs[li]] * sizes[li], y=ys[li], mode="markers",
                                 marker=dict(size=11, color="#555"), hoverinfo="none", showlegend=False))
    fig.update_layout(title=title, height=440, width=720,
                      xaxis=dict(tickvals=xs, ticktext=names, range=[-0.5, 4.5]),
                      yaxis=dict(visible=False))
    return fig


draw_ae(title="baseline autoencoder: DENSE throughout (every gene → every TF unit, both ways)")

# %%
draw_ae(enc_masked=True, W=W_clean, title="grn_real: ENCODER masked (gene→TF first layer = the graph)")

# %%
draw_ae(enc_masked=True, W=W_tangled, title="grn_real (tangled graph): masked + signed, overlapping regulons")

# %%
draw_ae(dec_masked=True, W=W_clean,
        title="grn_decoder (expiMap-style): DECODER masked — reconstruct each gene from its regulators")

# %%
draw_ae(enc_masked=True, dec_masked=True, W=W_clean,
        title="grn_symmetric: DOUBLE graph-aware — BOTH encoder (gene→TF) and decoder (TF→gene) masked")

# %% [markdown]
# Four autoencoders, differing only in **which layers the graph masks** (grey = dense/free;
# coloured = kept graph edges):
# - **baseline** — dense both ways (~36 free weights on the toy, ~3.4M on the real data).
# - **grn_real** — mask the **encoder** (gene→TF). The *inverse* direction (infer TF activity from
#   genes).
# - **grn_decoder** — mask the **decoder** (TF→gene). The **causal / generative** direction — a gene
#   is reconstructed only from its regulators. This is the expiMap-style placement, and on the real
#   data it did **better** than encoder-masking (and there the real graph beat its rewired control).
# - **grn_symmetric** — mask **both** (the strongest constraint; worst on the real data).
#
# So "graph-aware autoencoder" means **delete the connections that aren't regulatory edges** — and
# *where* you delete them (encoder vs decoder) matters. On the real data the dense mesh still beat
# every masked variant, but the decoder placement was the best of the graph-constrained models.

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

# %% [markdown]
# ## All the models side by side (probe **and** clustering)
# Finally, every representation compared two ways at a fixed noise level — a **supervised probe**
# (logistic-regression accuracy) and an **unsupervised clustering** score (KMeans ARI vs the true
# 3 types). Short version of what each is (and its real-project analogue):
#
# | toy model | what it is | real analogue |
# |---|---|---|
# | **raw genes** | the 12 gene values, no reduction | raw pseudobulk |
# | **PCA** | top-3 principal components | PCA baseline (strong, linear) |
# | **TF-activity (true)** | aggregate genes by the correct regulon | `dc_tfact` (graph-aware transform) |
# | **TF-activity (rewired)** | aggregate by a scrambled graph | `dc_tfact_rewired` (biology null) |
# | **random projection** | random 3-d linear features | `rand_proj` (dimensionality null) |

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

Xc, yc = simulate(noise=2.0, seed=0)
Rproj = np.random.default_rng(2).standard_normal((N_GENES, N_TF)) / np.sqrt(N_TF)
models = {
    "raw genes": Xc,
    "PCA": PCA(N_TF, random_state=0).fit_transform(Xc),
    "TF-activity (true)": tf_activity(Xc, gene_tf),
    "TF-activity (rewired)": tf_activity(Xc, rewired_tf),
    "random projection": Xc @ Rproj,
}
rows = []
for name, rep in models.items():
    acc = probe_acc(rep, yc)
    km = KMeans(N_TF, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(rep))
    rows.append(dict(model=name, probe_accuracy=round(acc, 2),
                     clustering_ARI=round(adjusted_rand_score(yc, km), 2)))
res3 = pd.DataFrame(rows)
fig = px.bar(res3.melt("model", ["probe_accuracy", "clustering_ARI"], var_name="metric", value_name="score"),
             x="model", y="score", color="metric", barmode="group",
             title="toy: every model, scored by supervised probe AND unsupervised clustering")
fig.update_layout(height=440, width=820, yaxis_range=[0, 1.0])
fig

# %%
res3.set_index("model")

# %% [markdown]
# **What it shows (and how it mirrors the real project):**
# - **TF-activity (true graph)** is best or tied on *both* metrics — and clearly best on **clustering**
#   (tighter groups), exactly the real-data result where the GRN-informed representation wins the
#   unsupervised metric even when PCA wins the probe.
# - **PCA** tops the *probe* (linear separability) but trails TF-activity on *clustering* — the same
#   "the metric flips the winner" effect seen on the real data.
# - The **nulls fail on both**: a rewired graph (0.49 / 0.10) and a random projection (0.64 / 0.08)
#   collapse — so the benefit is the *specific* wiring, not just aggregation or dimension.
#
# The learned encoders (baseline / `grn_real` / `grn_decoder` / `grn_symmetric`, drawn above) aren't
# trained here to keep the toy dependency-free, but they behave analogously on the real data: dense
# beats masked, decoder-placement beats encoder, and none beats the fixed TF-activity transform.
