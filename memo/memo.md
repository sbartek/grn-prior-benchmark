# GRN-Prior Expression Embedding Benchmark — Memo

*A short, decision-oriented write-up. Detailed numbers, figures, and the full model/evaluation
reference live on the project site:*
**[results](https://sbartek.github.io/grn-prior-benchmark/results/)** ·
**[models wiki](https://sbartek.github.io/grn-prior-benchmark/models/)** ·
**[evaluation wiki](https://sbartek.github.io/grn-prior-benchmark/evaluation/)**.
*Runnable notebooks:*
[EDA & suitability](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/01_eda_suitability.ipynb) ·
[toy intuition](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/02_toy_example.ipynb) ·
[full evaluation](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/03_evaluation.ipynb).

---

## TL;DR

We tested whether a public gene-regulatory-network prior (DoRothEA) makes RNA-expression embeddings
better at capturing biology. Bottom line:

- **The winner: DoRothEA TF-activity** — the graph used the classical way, as a fixed feature
  transform (how active each transcription factor is). It carries genuine biological signal (beats
  scrambled-graph and random controls), **wins when data is scarce**, and forms the **tightest,
  most biologically-clustered** embedding. It's also interpretable — each dimension is a named TF.
  *(A newer public network, CollecTRI, does the same thing a bit better; included only as a
  robustness check — see Scope.)*

- **The catch — PCA.** A plain PCA baseline is remarkably strong: it **ties or beats** the prior on
  the standard **supervised** test, and it's trivially simple (no graph, no external tool). So use
  **PCA as the default when data is plentiful**, and switch to **TF-activity when data is scarce or
  you want interpretable / well-clustered biology**. And note PCA's own weakness: it **loses** the
  stricter **unsupervised clustering** test — the winner literally depends on which metric you pick.

- **The clear loser: baking the graph into a deep encoder** (as a hard or soft mask). It's *below*
  both PCA and a plain autoencoder everywhere, and a scrambled graph does just as well — so that
  effect was regularization, never the biology. Don't do this.

*In one line: use the GRN as a TF-activity transform (great in low-data, interpretable), keep PCA
as the simple default, and never bake the graph into a large model.*

---

## The question (and why it matters)

Biological data is small, noisy, and messy. A tempting idea: inject decades of curated biology — a
map of which transcription factors regulate which genes — to help a model see through the noise.
Does it actually work?

Concretely: does a graph-aware encoder using DoRothEA regulatory edges produce **better** pseudobulk
expression embeddings than a non-graph baseline — where *better* means the embedding captures useful
biological state (cell type, disease), not that it reconstructs the input — especially under
low-data, noisy, and graph-corruption conditions?

We built this as a fair, controlled benchmark on public data. Full method:
[evaluation wiki](https://sbartek.github.io/grn-prior-benchmark/evaluation/).

## The setup in a nutshell

Public immune-cell (PBMC) data from CELLxGENE, collapsed to ~500 clean **pseudobulk** profiles
(average of cells per donor × cell type). We test embeddings by freezing them and asking a simple
classifier to recover **cell type** on **held-out donors** — the honest generalization test. We
stress it under low data and noise, and — crucially — we run **scrambled-graph controls** so we can
tell *real biology* apart from *any old structure*. (Dataset is well-suited: balanced disease, no
sex/assay confounds; the one catch is that disease is tied to donor identity, so we treat cell type
as the trustworthy readout. Details: [EDA notebook](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/01_eda_suitability.ipynb).)

**Scope & assumptions.** The exercise specifies the **DoRothEA** TF–target network — that is the
required and **primary** network here; all headline results use it. I additionally ran **CollecTRI**
(a newer public TF→target network from the same tool/lab, *not* a curated pathway gene set, so not
in the excluded list) purely as a **robustness check** — to test whether the effect is
DoRothEA-specific. It is supporting evidence, not the load-bearing result: every conclusion holds on
DoRothEA alone.

## The story of what we found

**Act 1 — the obvious idea disappoints.** We first baked the graph into the neural network: force
each hidden unit to be a transcription factor wired only to its target genes. It *underperformed* —
below both PCA and a plain autoencoder. Worse, when we **scrambled** the graph (kept its shape, shuffled
the biology), performance barely changed. That's the tell: whatever tiny effect the constraint had
was generic regularization, **not the regulatory biology**. Making the constraint softer or stronger
didn't rescue it — more prior was simply worse.

**Act 2 — the classical approach works.** Then we used the same graph the *standard* way — as a
fixed **TF-activity transform** (summarize each sample by how active each TF is). This time the
biology was real: it beat a rewired-network control *and* a random-projection control, in that order
(real regulons > rewired regulons > random). And it did something the constraint never did — it
**beat the baselines when data was scarce**, exactly where a good prior should help. Swapping
DoRothEA for the newer **CollecTRI** network helped even more, so this isn't a quirk of one
catalogue. And it all **replicated on a second dataset** (COVID).

**Act 3 — the twist: the metric changes the winner.** On the standard supervised test, plain **PCA**
is the champion — deflating, but a well-known pattern ("PCA still rules"). But we also ran the
stricter, label-free **clustering** test (do the cell types form natural groups?), and it *flipped*:
PCA came **last**, and the graph-informed representations clustered **best**. PCA makes classes
*separable* but leaves them smeared together; the biological representations make them *clumpier*. So
"PCA wins" was true only for one way of asking the question.

**A footnote on placement.** Among the network-baked variants, putting the graph on the **decoder**
(the causal "TFs → genes" direction, as in expiMap) beat putting it on the encoder — a nice, sensible
result — though it still didn't beat the plain baseline.

## What this means

- **The regulatory prior encodes genuine biology** — but it's *not more informative than a strong
  simple baseline* on well-powered data. Its value shows up in the **hard regimes** (low data) and on
  **stricter metrics** (clustering).
- **How you apply it dominates whether it helps.** Feature transform: good. Deep-network constraint:
  bad. This is the single most actionable takeaway.
- **Report more than one metric.** A supervised probe and an unsupervised clustering score answer
  different questions and here they *disagree* — reporting only one would have told a misleading story.

This mirrors where the field is going (biology-wired networks tend to win on interpretability and
small-data, not raw accuracy; the frontier "virtual cell" models use no hard-coded GRN at all). Full
citations on the [results page](https://sbartek.github.io/grn-prior-benchmark/results/) and in
`references.md`.

## Where we were careful (and honest)

- **No cherry-picking the metric:** we report the case where the prior loses (probe) *and* where it
  wins (clustering).
- **Scrambled-graph controls throughout** — the discipline that separates "biology" from "any
  structure."
- **Held-out donors** so results reflect new patients, not memorized ones.
- **Robustness checks:** the verdict survives changing the bottleneck size, removing early stopping,
  and a second dataset.
- **Stated limits:** one primary dataset; disease is donor-confounded; cell-type identity is a task
  linear methods already ace; DoRothEA is a single *global* network (real regulation is
  cell-type-specific).

## What we'd do next

A prioritized shortlist (fuller version on the results page):
1. **Cell-type-specific networks** instead of one global graph — the likeliest reason the prior
   under-delivered.
2. **A genuinely regulatory readout** (perturbation response) rather than cell-type identity, which
   linear methods already solve.
3. **Transfer / supervised embeddings** (train on one label, probe another) and a proper **GNN**.
4. **Within-subject / longitudinal data** to break the donor–disease confound.

---

*The bottom line: a public GRN prior is worth using — as a TF-activity transform, in low-data
settings, evaluated on more than one metric — but it is not a free win, and baking it into a large
model is counterproductive.*
