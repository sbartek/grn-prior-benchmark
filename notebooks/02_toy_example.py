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
        title="grn_symmetric: BOTH layers masked (gene→TF and TF→gene)")

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
