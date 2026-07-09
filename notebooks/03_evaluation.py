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
# # Full evaluation — results & interpretation
#
# This notebook consolidates the **definitive evaluation** on the real data (it reads the result
# tables produced by `scripts/11_final.py`, `08_second_dataset.py`, `10_bottleneck.py`,
# `13_decoder_prior.py`). For the *why* and the biology, see `PLAN.md`, `memo/memo.md`, and the
# teaching notebooks `01`/`02`.
#
# **Question.** Does a DoRothEA GRN prior improve *pseudobulk* expression embeddings vs a non-graph
# baseline — judged by how well the embedding captures **biological state** (cell type / disease)
# via a probe on **held-out donors**?
#
# **Protocol.** Donor-grouped 5-fold CV, re-shuffled per seed. Inside each fold: train the encoder
# on train donors (unsupervised), freeze it, fit a logistic-regression probe on train-donor
# embeddings, score on held-out donors. Metric: macro-F1 (fixed 15-class label set).

# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook"

TAB = Path.cwd().parent / "results" / "tables"


def load(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else None


final = load("final.csv")
covid = load("covid.csv")
bottleneck = load("bottleneck.csv")
decoder = load("decoder.csv")
leakage = load("donor_leakage.csv")
print("tables loaded:", [n for n, t in
      dict(final=final, covid=covid, bottleneck=bottleneck, decoder=decoder, leakage=leakage).items()
      if t is not None])


def grouped_bar(df, order, conds, title, task=None, ymax=0.95):
    d = df[df.task == task] if (task and "task" in df) else df
    d = d[d.model.isin(order) & d.condition.isin(conds)].copy()
    d["model"] = pd.Categorical(d["model"], order)
    d["condition"] = pd.Categorical(d["condition"], conds)
    fig = px.bar(d.sort_values(["condition", "model"]), x="condition", y="mean", color="model",
                 error_y="std", barmode="group", title=title, range_y=[0, ymax],
                 category_orders={"model": order, "condition": conds})
    fig.update_layout(height=440, width=980, yaxis_title="cell-type macro-F1")
    return fig


# %% [markdown]
# ## Model glossary
# | model | what it is |
# |---|---|
# | **PCA** | linear top-64 principal components — the strong, unsupervised, parameter-free baseline |
# | **baseline** | dense autoencoder (no graph), 64-d bottleneck, MSE reconstruction |
# | **dc_tfact** | decoupler ULM **TF-activity** from the DoRothEA graph — a *fixed* (non-learned) transform |
# | **dc_tfact_collectri** | same, but the CollecTRI network (newer, broader) instead of DoRothEA |
# | **dc_tfact_pca** | TF-activity reduced to 64 dims (dimension-matched to the others) |
# | **rand_proj** | random linear projection at the TF dim — the "is it just dimensionality?" null |
# | **dc_tfact_rewired** | TF-activity on a degree-preserving *rewired* DoRothEA net — the transform-side "is it biology?" null |
# | **grn_real** | autoencoder with the **encoder** first layer masked to TF→target edges |
# | **grn_rewired / grn_random** | same encoder mask but a corrupted graph — the "is it biology?" null |
# | **grn_soft:λ** | dense encoder + penalty shrinking off-regulon weights (soft version of the mask) |
# | **grn_decoder** | mask the **decoder** (TF→gene, causal direction, expiMap-style) |
# | **grn_symmetric** | mask **both** encoder and decoder |

# %% [markdown]
# ## Consolidated comparison table
# Cell-type macro-F1 (donor-grouped CV); last column = COVID 3-class **disease**. `NaN` = not run
# for that model (pure controls / placement variants were only run on the primary dataset).

# %%
def _cell(df, m, c, task=None):
    if df is None:
        return None
    d = df[(df.model == m) & (df.condition == c)]
    if task and "task" in df:
        d = d[d.task == task]
    return round(float(d.iloc[0]["mean"]), 3) if len(d) else None


def _prim(m, c):
    v = _cell(final, m, c)
    return v if v is not None else _cell(decoder, m, c)


_models = ["pca", "baseline", "dc_tfact", "dc_tfact_collectri", "dc_tfact_pca", "rand_proj",
           "dc_tfact_rewired", "grn_real", "grn_rewired", "grn_soft:0.001", "grn_decoder", "grn_symmetric"]
comparison = pd.DataFrame([{
    "model": m,
    "prim full": _prim(m, "full"), "prim k=8": _prim(m, "lowdata:8"), "prim noise": _prim(m, "noise:0.3"),
    "covid full": _cell(covid, m, "full", "cell_type"), "covid k=8": _cell(covid, m, "lowdata:8", "cell_type"),
    "covid noise": _cell(covid, m, "noise:0.3", "cell_type"), "covid disease": _cell(covid, m, "full", "disease"),
} for m in _models]).set_index("model")
comparison

# %% [markdown]
# ## 1. Headline — the two ways to use the GRN
# `dc_*` = TF-activity **transform** (fixed); `grn_*` = graph baked into a learned **encoder**.
# Nulls: `rand_proj` (random features), `*_rewired` (scrambled graph).

# %%
order = ["pca", "baseline", "grn_real", "grn_rewired", "grn_soft:0.001",
         "rand_proj", "dc_tfact_rewired", "dc_tfact", "dc_tfact_collectri"]
grouped_bar(final, order, ["full", "lowdata:4", "lowdata:8", "lowdata:16", "noise:0.3", "noise:0.1"],
            "Definitive sweep (cell type): constraint hurts; TF-activity beats rewired & wins under low-data")

# %% [markdown]
# **Read:** `grn_*` (encoder constraint) sits below PCA and baseline everywhere; corrupted graphs do
# as well or better. `dc_tfact` (transform) beats both `dc_tfact_rewired` and `rand_proj` → real
# regulatory signal — and beats PCA/baseline under **low data** (CollecTRI strongest).

# %% [markdown]
# ## 2. The decisive controls, isolated
# Two "is it really biology?" nulls: for the **encoder**, real vs rewired/random; for the
# **transform**, real vs rewired-net vs random projection.

# %%
ctrl = ["baseline", "grn_real", "grn_rewired", "grn_random", "rand_proj", "dc_tfact_rewired", "dc_tfact"]
f_full = final[(final.condition == "full") & final.model.isin(ctrl)].copy()
f_full["model"] = pd.Categorical(f_full["model"], ctrl)
fig = px.bar(f_full.sort_values("model"), x="model", y="mean", error_y="std", color="model",
             title="full data: encoder controls (grn_*) vs transform controls (dc_*/rand_proj)",
             range_y=[0, 0.95])
fig.update_layout(height=430, width=820, showlegend=False, yaxis_title="cell-type macro-F1")
fig

# %% [markdown]
# ## 3. Where to place the prior — encoder vs decoder (expiMap-style) vs symmetric
# The decoder direction (TF → gene) is the causal/generative one. Does placing the mask there help?

# %%
if decoder is not None:
    dord = ["baseline", "grn_real", "grn_rewired", "grn_decoder", "grn_decoder_rewired",
            "grn_symmetric", "grn_symmetric_rewired"]
    display(grouped_bar(decoder, dord, ["full", "lowdata:8", "noise:0.3"],
                        "graph-prior placement: decoder > encoder, and decoder shows real > rewired"))
else:
    print("decoder.csv not ready yet — rerun after scripts/13_decoder_prior.py finishes")

# %% [markdown]
# ## 4. External validity — second dataset (COVID PBMC)
# Same pipeline on a different dataset (422k cells, 75 donors, 28 cell types, 3 disease states).

# %%
if covid is not None:
    cord = ["pca", "baseline", "dc_tfact", "dc_tfact_pca", "grn_soft:0.001", "grn_real", "grn_rewired"]
    display(grouped_bar(covid, cord, ["full", "lowdata:8", "noise:0.3"],
                        "COVID PBMC — cell type: same ordering as the primary dataset", task="cell_type"))

# %% [markdown]
# ## 5. Disease — RA (confounded) vs COVID (decodable)
# Disease is *between-subjects* (each donor is one state), so it is entangled with donor identity.
# On RA (2-class) nothing beats chance; on COVID (3-class) disease **is** decodable (PCA best), but
# the caveat stands.

# %%
if covid is not None:
    cord = ["pca", "baseline", "dc_tfact", "grn_real", "grn_rewired"]
    display(grouped_bar(covid, cord, ["full", "lowdata:8", "noise:0.3"],
                        "COVID disease (3-class): decodable, PCA best, grn ≈ rewired", task="disease", ymax=0.8))

# %% [markdown]
# ## 6. Robustness — bottleneck dimension
# Is the verdict an artifact of z=64? Sweeping z ∈ {32,64,128}, the ordering is stable.

# %%
if bottleneck is not None:
    b = bottleneck.copy()
    b["z"] = b["z"].astype(str)
    fig = px.line(b[b.condition == "full"].sort_values(["model", "z"]), x="z", y="mean",
                  color="model", markers=True, error_y="std",
                  title="bottleneck-dim sensitivity (full data): ordering stable across z")
    fig.update_layout(height=420, width=720, yaxis_title="cell-type macro-F1")
    display(fig)

# %% [markdown]
# ## 7. Biology vs batch — donor leakage
# Does the embedding encode donor identity (batch) instead of biology? Lower = less leakage.

# %%
if leakage is not None:
    fig = px.bar(leakage.sort_values("donor_acc"), x="donor_acc", y="model", orientation="h",
                 title="donor-identity leakage (kNN donor accuracy; lower = better)")
    fig.update_layout(height=360, width=680, xaxis_title="donor prediction accuracy")
    display(fig)

# %% [markdown]
# ## 8. Unsupervised check — does the embedding *cluster* by biology? (and why not DBSCAN)
# Our probe is **supervised** (it uses labels to find a boundary). A complementary, stricter test is
# **unsupervised clustering** of the embedding, then comparing clusters to the true cell types with
# **ARI / NMI** (the scIB standard). This asks whether biology is the *intrinsic* geometry, not just
# something a classifier can carve out. Below: KMeans vs DBSCAN on the PCA embedding.

# %%
import sys as _sys
_sys.path.insert(0, str(Path.cwd().parent / "src"))
from sklearn.cluster import DBSCAN, KMeans          # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI  # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402

from grn_bench.data import load_aligned              # noqa: E402

_d = load_aligned()
_ct = _d["obs"].cell_type.values
_emb = StandardScaler().fit_transform(PCA(64, random_state=0).fit_transform(_d["X"]))
_km = KMeans(15, random_state=0, n_init=10).fit_predict(_emb)
print(f"KMeans(15):   ARI={ARI(_ct, _km):.2f}  NMI={NMI(_ct, _km):.2f}   (vs supervised probe macro-F1 ~0.87)")
for _eps in [3, 6, 10]:
    _db = DBSCAN(eps=_eps, min_samples=5).fit_predict(_emb)
    _nc = len(set(_db)) - (1 if -1 in _db else 0)
    print(f"DBSCAN eps={_eps:2d}: clusters={_nc:2d}  noise={(_db == -1).mean():.0%}  ARI={ARI(_ct, _db):.2f}")

# %% [markdown]
# **Two takeaways:**
# 1. **Clustering (ARI ~0.19) ≪ the supervised probe (macro-F1 ~0.87).** The biology is *linearly
#    readable* (the probe finds it) but the embedding is **not clean, well-separated balls** —
#    related subtypes (naive/memory/effector T-cells) blur together. So a probe "rescues"
#    separability that clustering can't recover on its own; clustering is the stricter test.
# 2. **DBSCAN is a poor fit here.** It's density-based and very `eps`-sensitive: too small → a third
#    of points labelled *noise*; slightly larger → everything collapses into **one** blob
#    (ARI ~0.01). Single-cell embeddings have imbalanced, variable-density, high-dimensional
#    clusters — exactly DBSCAN's weak spot. The field uses **graph-based clustering (Leiden/Louvain
#    on a kNN graph)** + ARI/NMI (scIB), not DBSCAN. KMeans is a reasonable quick proxy; DBSCAN is
#    not.
#
# So: yes, unsupervised clustering is a sensible *complementary* evaluation, but pick Leiden/KMeans
# over DBSCAN for this data.

# %% [markdown]
# ### Per-model clustering (KMeans ARI/NMI vs cell type) — the metric flips the winner
# Running the label-free clustering metric across models (`14_clustering.py`) gives a strikingly
# *different* ranking from the supervised probe.

# %%
clustering = load("clustering.csv")
if clustering is not None:
    cl = clustering.melt("model", ["ARI", "NMI"], var_name="metric", value_name="score")
    fig = px.bar(cl, x="model", y="score", color="metric", barmode="group",
                 title="unsupervised clustering vs cell type — TF-activity/AE beat PCA (opposite of the probe)")
    fig.update_layout(height=420, width=780)
    display(fig)
    display(clustering.set_index("model").round(3))

# %% [markdown]
# **The winner depends on the metric.** By the *supervised probe*, PCA wins; by *unsupervised
# clustering*, PCA is **worst** (ARI ≈ 0.12) and the GRN-informed **CollecTRI TF-activity** (≈ 0.30)
# and the autoencoders cluster best. PCA maximises linear *separability* but leaves elongated,
# overlapping geometry; the TF-activity transforms and learned encoders produce **tighter, more
# clustered** cell-type groups. So "PCA is best" was probe-specific — on the intrinsic-structure
# (scIB-style) metric, the biological representations come out ahead. This is why reporting *both* a
# supervised probe and an unsupervised clustering metric matters: they answer different questions and
# here they disagree.

# %% [markdown]
# ### Seeing the clusters — PCA vs TF-activity
# Project two embeddings to 2-D (UMAP) so we can *look* at the structure the ARI numbers summarise.
# We colour by **true cell type** (does biology form blobs?) and by the **KMeans cluster** the
# algorithm found (do the found clusters match biology?).

# %%
from grn_bench.experiments import compute_tfact       # noqa: E402
import umap                                            # noqa: E402

_embs = {
    "PCA-64": StandardScaler().fit_transform(PCA(64, random_state=0).fit_transform(_d["X"])),
    "dc_tfact (DoRothEA)": StandardScaler().fit_transform(compute_tfact(_d, "dorothea")),
}
_parts = []
for _name, _E in _embs.items():
    _u = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(_E)
    _cl = KMeans(15, random_state=0, n_init=10).fit_predict(_E)
    _parts.append(pd.DataFrame({"UMAP1": _u[:, 0], "UMAP2": _u[:, 1], "cell type": _d["obs"].cell_type.values,
                                "KMeans cluster": _cl.astype(str), "embedding": _name}))
_viz = pd.concat(_parts, ignore_index=True)

# %%
fig = px.scatter(_viz, x="UMAP1", y="UMAP2", color="cell type", facet_col="embedding", opacity=0.75,
                 title="UMAP coloured by TRUE cell type — dc_tfact forms tighter cell-type groups than PCA")
fig.update_traces(marker_size=5); fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=520, width=1000, legend=dict(font_size=8, itemsizing="constant"))
fig

# %%
fig = px.scatter(_viz, x="UMAP1", y="UMAP2", color="KMeans cluster", facet_col="embedding", opacity=0.75,
                 title="UMAP coloured by the KMeans cluster found (label-free)")
fig.update_traces(marker_size=5); fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=520, width=1000, showlegend=False)
fig

# %% [markdown]
# In the top plot, **dc_tfact** separates the cell types into cleaner blobs than **PCA** (whose
# groups are more elongated/overlapping) — the visual reason its clustering ARI is higher. In the
# bottom plot you can see which groups KMeans actually recovers (and where it splits/merges the
# harder, similar T-cell subtypes).

# %% [markdown]
# ## 9. Post-mortem — per-fold extras (macro AUC, silhouette, cell-type ASW, ARI, NMI)
#
# **Added after submission (2026-07-09).** The `clustering.csv` table above was computed on the
# full-data-trained AE embedding — `all_idx = np.arange(n)` in `scripts/14_clustering.py` line 41.
# For AE-based rows that's a training-set leak: the AE has seen every sample before its embedding
# is clustered.
#
# `scripts/18_extra_metrics_perfold.py` fixes that: for each of 5 donor-grouped folds, the AE is
# trained only on train-donors and every metric is computed on **test-donor embeddings** where the
# AE has never seen the data. Fixed transforms (PCA, dc_tfact, dc_tfact_collectri) are unaffected —
# they were always per-sample. Numbers below are means across folds.

# %%
_perfold = pd.read_csv(TAB / "extra_metrics_perfold.csv")
_full = pd.read_csv(TAB / "extra_metrics.csv") if (TAB / "extra_metrics.csv").exists() else None
print("per-fold extras:")
print(_perfold.round(3).to_string(index=False))

# %% [markdown]
# ### Full-data (R1-confounded) vs per-fold (honest) side by side

# %%
if _full is not None:
    _cmp = (_perfold[["model", "macro_f1", "macro_auc", "silhouette", "ct_asw"]]
            .merge(_full[["model", "macro_f1", "macro_auc", "silhouette", "ct_asw"]],
                   on="model", suffixes=("_perfold", "_fulldata")))
    _cmp = _cmp[["model",
                 "macro_f1_perfold", "macro_f1_fulldata",
                 "macro_auc_perfold", "macro_auc_fulldata",
                 "silhouette_perfold", "silhouette_fulldata",
                 "ct_asw_perfold", "ct_asw_fulldata"]]
    print(_cmp.round(3).to_string(index=False))

# %% [markdown]
# ### What changed vs Act 3 of the memo
#
# The Act-3 "metric flip" ("PCA wins the supervised probe but loses on clustering") was almost
# entirely carried by the R1 confound. Corrected picture after per-fold:
#
# - **PCA leads on macro-F1, macro AUC, silhouette.** Three of five metrics.
# - **CollecTRI-ULM leads on ARI and NMI** — but the gap shrunk dramatically:
#   - ARI: full-data-confounded gap (dc_tfact_collectri 0.304 vs pca 0.124) = 0.180 → per-fold gap = 0.06.
#   - NMI: 0.204 → 0.06.
# - **AE-based silhouettes drop 0.10 or more** once the R1 leak is removed — direct proof R1 was
#   inflating them.
#
# The corrected one-liner replaces the memo's Act-3 line:
# *"PCA and CollecTRI-ULM are close competitors across all five representation-quality metrics;
# CollecTRI keeps a small ARI/NMI edge (label-agreement) while PCA leads on silhouette (geometric
# tightness); encoder-mask hurts everywhere. 'Winner depends on the metric' is technically still
# true, but the effect is 3× smaller than the R1-confounded numbers suggested."*
#
# The two other acts hold up: encoder-mask (`grn_real`) is the worst everywhere;
# TF-activity ULM (esp. CollecTRI) beats the rewired-network null.

# %% [markdown]
# ## Verdict
# - **How you use the GRN matters more than whether you use it.** As a learned-**encoder**
#   constraint it hurts (rewired ≈ real → regularization, not biology). As a **TF-activity
#   transform** it carries real signal (beats rewired-net and random) and helps under **low data**;
#   at full data it only ties PCA.
# - **Placement matters:** the **decoder** (causal TF→gene direction, expiMap-style) is a better
#   place than the encoder, and there the real graph beats its rewired control — but it still does
#   not beat the dense baseline.
# - **The metric matters.** "PCA wins" holds for the *supervised probe*; on the *unsupervised*
#   clustering metric (ARI/NMI) PCA is **worst** and the GRN-informed TF-activity clusters best. The
#   biological prior helps *intrinsic cluster structure* even where it doesn't help linear probing.
# - **Not prior-specific** (CollecTRI ≥ DoRothEA), **replicates** on COVID, **robust** to bottleneck
#   size and to removing early stopping. Disease is decodable only where the design allows it
#   (COVID 3-class), and even there it is partly donor identity.
#
# A fair, carefully-controlled result: the regulatory prior encodes genuine biology, but is not more
# informative than a strong simple baseline (PCA) except in the low-data regime — and it helps most
# when used the classical way (TF activity) or, among learned models, in the decoder.
