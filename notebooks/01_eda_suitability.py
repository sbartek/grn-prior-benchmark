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
# # EDA & dataset suitability — GRN-Prior Expression Embedding Benchmark
#
# A tight exploratory pass over the RA PBMC pseudobulk and the DoRothEA graph, to answer:
# **is this dataset suitable for the question, and what can/can't it show?** The experiments
# themselves live in `scripts/03_run_experiments.py`; this notebook is the narrative EDA layer.
#
# Headline that falls out below: **cell type separates cleanly and linearly; disease does not,
# and is aliased with donor** — which is why we treat cell type as the trustworthy readout and
# disease as suggestive only. Plots are interactive Plotly (white theme).

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook"   # embed plotly.js -> figures render in Lab, GitHub, and HTML export

sys.path.insert(0, str(Path.cwd().parent / "src"))
from grn_bench.data import load_aligned  # noqa: E402

d = load_aligned()
X, obs, genes, tfs = d["X"], d["obs"], d["genes"], d["tfs"]
print(f"pseudobulk: {X.shape[0]} samples x {X.shape[1]} genes (aligned to graph)")
print(f"donors={obs.donor_id.nunique()}  cell_types={obs.cell_type.nunique()}")

# %% [markdown]
# ## 1. Composition & balance
# Donor × cell-type coverage (how many cells went into each pseudobulk sample), and the
# disease / sex balance that determines what confounds we must control.

# %%
piv = obs.pivot_table(index="donor_id", columns="cell_type", values="n_cells", aggfunc="sum")
fig = px.imshow(np.log1p(piv.fillna(0)), aspect="auto", color_continuous_scale="Viridis",
                labels=dict(color="log1p cells"), title="cells per (donor × cell_type)")
fig.update_xaxes(tickangle=45, tickfont_size=9)
fig.update_layout(height=650, width=1000)
fig

# %%
donor_meta = obs.drop_duplicates("donor_id")
print("donors per disease:\n", donor_meta.disease.value_counts().to_string())
print("\ndonor sex x disease:\n", donor_meta.groupby(["disease", "sex"]).size().to_string())
print(f"\ncells/group: min={obs.n_cells.min()} median={int(obs.n_cells.median())} max={obs.n_cells.max()}")

# %% [markdown]
# Balanced **18 RA / 18 normal**, **12F/6M in both arms** (no sex–disease confound), single assay.
# The design is **between-subjects**: each donor is only RA *or* normal, so disease is confounded
# with donor identity. This is the key limitation the evaluation must respect.

# %% [markdown]
# ## 2. Library size (QC sanity)

# %%
lib = np.asarray(d["counts"].sum(1)).ravel()
fig = px.histogram(x=np.log10(lib), nbins=40, title="pseudobulk library sizes",
                   labels=dict(x="log10 total counts / pseudobulk"))
fig.update_layout(height=380, width=680, bargap=0.05)
fig

# %% [markdown]
# ## 3. The confound, visually: PCA & UMAP of the pseudobulk
# If cell type is a strong linear signal and disease is not, a 2-D projection should separate
# **cell type** but mix **disease** and show **donor** substructure. Hover to inspect points.

# %%
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

Xs = StandardScaler().fit_transform(X)
pcs = PCA(n_components=30, random_state=0).fit_transform(Xs)

import umap  # noqa: E402

um = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(pcs)

emb_df = obs.copy()
emb_df["PC1"], emb_df["PC2"] = pcs[:, 0], pcs[:, 1]
emb_df["UMAP1"], emb_df["UMAP2"] = um[:, 0], um[:, 1]


def proj(kind, color, showlegend=True):
    x, y = (("PC1", "PC2") if kind == "PCA" else ("UMAP1", "UMAP2"))
    fig = px.scatter(emb_df, x=x, y=y, color=color, title=f"{kind} — coloured by {color}",
                     hover_data=["donor_id", "cell_type", "disease"], opacity=0.85)
    fig.update_traces(marker_size=7)
    fig.update_layout(height=460, width=760, showlegend=showlegend,
                      legend=dict(font_size=9, itemsizing="constant"))
    return fig


# %% [markdown]
# **PCA coloured by cell type** — clean, well-separated clusters.

# %%
proj("PCA", "cell_type")

# %% [markdown]
# **PCA coloured by disease** — RA and normal mixed across every cluster.

# %%
proj("PCA", "disease")

# %% [markdown]
# **PCA coloured by donor** — fine substructure within each cell type.

# %%
proj("PCA", "donor_id", showlegend=False)

# %% [markdown]
# **UMAP coloured by cell type / disease / donor** — same story, non-linear projection.

# %%
proj("UMAP", "cell_type")

# %%
proj("UMAP", "disease")

# %%
proj("UMAP", "donor_id", showlegend=False)

# %% [markdown]
# As expected: **cell type** forms clean clusters, **disease** is thoroughly mixed, and **donor**
# shows fine substructure inside each cell type. This is the whole reason cell type is our primary
# readout and disease is suggestive-only, always split by donor.

# %% [markdown]
# ## 4. DoRothEA graph structure

# %%
g = d["graph"]
out_deg = np.bincount(g["real_cols"], minlength=len(tfs))
in_deg = np.bincount(g["real_rows"], minlength=len(genes))
print(f"input genes={len(genes)}  TFs(hidden)={len(tfs)}  edges={len(g['real_rows'])}  "
      f"density={len(g['real_rows'])/(len(genes)*len(tfs)):.4f}")

# connectivity is many-to-many: a TF regulates MANY genes; a gene is regulated by SEVERAL TFs
od = out_deg[out_deg > 0]           # TF -> #genes (regulon size)
idg = in_deg[in_deg > 0]            # gene -> #TFs
print(f"TF -> genes   (regulon size):    min={od.min():4d}  max={od.max():5d}  avg={od.mean():6.1f}  median={int(np.median(od))}")
print(f"gene -> TFs   (# regulators):     min={idg.min():4d}  max={idg.max():5d}  avg={idg.mean():6.1f}  median={int(np.median(idg))}")

deg_df = pd.concat([
    pd.DataFrame({"degree": out_deg, "which": "TF out-degree (regulon size)"}),
    pd.DataFrame({"degree": in_deg, "which": "gene in-degree (# regulating TFs)"}),
])
fig = px.histogram(deg_df, x="degree", facet_col="which", nbins=40,
                   title="DoRothEA degree distributions")
fig.update_xaxes(matches=None); fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=380, width=1000, showlegend=False)
fig

# %% [markdown]
# ## 5. Headline result
# Cell-type macro-F1 (donor-grouped CV, full data). Full sweep + interpretation in
# `results/tables/results.csv` and `memo/memo.md`.

# %%
res = pd.read_csv(Path.cwd().parent / "results" / "tables" / "results.csv")
full = res[(res.condition == "full") & (res.task == "cell_type")].sort_values("mean")
fig = px.bar(full, x="mean", y="model", orientation="h", error_x="std", range_x=[0, 1],
             title="full-data cell-type embedding quality", labels=dict(mean="macro-F1"),
             color="mean", color_continuous_scale="Teal")
fig.update_layout(height=380, width=720, coloraxis_showscale=False)
fig

# %% [markdown]
# **Takeaway.** PCA and the unconstrained baseline beat every DoRothEA-masked variant, and the
# real graph does not beat its same-density controls (random ≥ real). The GRN prior adds
# complexity without adding usable biological signal on this dataset — a fair negative result.
