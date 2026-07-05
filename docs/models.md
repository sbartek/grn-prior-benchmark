# Wiki — the models

Every representation we compared, grouped by **how it uses the DoRothEA gene-regulatory network
(GRN)**. All produce an *embedding* (a compact vector per pseudobulk sample) that a probe then
reads for cell type / disease. Two big families: **`dc_*` = the graph as a fixed transform**;
**`grn_* ` = the graph baked into a learned neural network**.

Code: [`src/grn_bench/models.py`](https://github.com/sbartek/grn-prior-benchmark/blob/main/src/grn_bench/models.py) ·
[`experiments.py`](https://github.com/sbartek/grn-prior-benchmark/blob/main/src/grn_bench/experiments.py).
Toy visuals of the encoders: the [toy notebook](https://github.com/sbartek/grn-prior-benchmark/blob/main/notebooks/02_toy_example.ipynb).

---

## Baselines — no graph at all

### `PCA`
Principal Component Analysis: rotate the data onto its highest-variance directions and keep the top
64. Unsupervised, linear, no training. **The strong baseline** — hard to beat on the supervised
probe (it captures cell-type identity, a high-variance near-linear signal), but *worst* on
unsupervised clustering.

### `baseline` (dense autoencoder)
A neural net that compresses expression to a 64-d bottleneck and reconstructs it (MSE loss, no
labels). Same architecture as the graph models but with **no graph** — the fair "does the graph
add anything?" reference.

---

## The GRN as a fixed **transform** (`dc_*`)

Here the graph isn't learned — it's used as arithmetic to turn expression into **TF-activity**
(how active each transcription factor is, inferred from its target genes). This is the classical,
`decoupler`/SCENIC way of using a regulatory network.

### `dc_tfact`
decoupler **ULM** TF-activity using DoRothEA: a per-sample signed weighted sum over each TF's
regulon → one number per TF (293 dims). Fully fixed, no training. **The one graph method that
carries real signal** — beats its rewired-net and random nulls, and beats PCA/baseline under low
data.

### `dc_tfact_collectri`
Same idea, but the **CollecTRI** network (newer, broader, ~1,185 TFs) instead of DoRothEA.
Consistently ≥ DoRothEA — the effect isn't specific to one catalogue; a better network helps more.

### `dc_tfact_pca`
`dc_tfact` reduced to 64 dims with PCA — **dimension-matched** to the other models. At matched
dimension the advantage largely disappears (≈ baseline), showing part of `dc_tfact`'s edge was its
higher dimensionality.

---

## Nulls & controls — "is it really biology?"

These have the *same shape* as a real model but the biology removed. If a real model doesn't beat
its null, its "benefit" wasn't biology.

| model | what's scrambled | tests |
|---|---|---|
| `rand_proj` | random linear features at the TF dimension | is `dc_tfact`'s gain just dimensionality? |
| `dc_tfact_rewired` | TF-activity on a **degree-preserving rewired** DoRothEA net | is the transform signal the *specific* regulons? |
| `grn_rewired` | encoder mask from a rewired graph | is the encoder constraint biology or just sparsity? |
| `grn_random` | encoder mask from a random graph | same, harsher null |
| `grn_sign_shuffled` | encoder mask with the +/- signs permuted | do the activate/repress signs carry signal? |

Result: `dc_tfact` **>** `dc_tfact_rewired` **>** `rand_proj` (real biology in the transform), but
`grn_real` **≈** `grn_rewired`/`grn_random` (the encoder constraint is just regularization).

---

## The GRN as a learned **encoder constraint** (`grn_*`)

Same autoencoder as `baseline`, but the graph restricts the wiring — each hidden unit is a TF
connected only to its regulon.

### `grn_real`
First (encoder) layer **masked** to the DoRothEA edges: effective weight = `mask · sign ·
softplus(raw)`. The ±1 signs are **fixed** (biology); only the strength is learned. Consistently
*below* PCA and the dense baseline.

### `grn_soft:λ`
A **softer** version — the encoder stays dense but a penalty `λ·‖off-regulon weights‖²` pulls
non-edges toward zero. λ=0 is the baseline; larger λ approaches the hard mask. More prior = worse.

---

## The GRN — *where* to place the constraint (`grn_decoder`, `grn_symmetric`)

The encoder infers TFs *from* genes (inverse direction); the **decoder** goes TFs → genes (the
*causal* direction, how biology actually works). Biologically-informed autoencoders (expiMap) put
the mask on the decoder.

### `grn_decoder`
Mask the **decoder** (reconstruct each gene only from its regulators). The **best-performing**
graph-constrained model — better than encoder masking, and here the real graph beats its rewired
control. Still below the dense baseline.

### `grn_symmetric`
Mask **both** encoder and decoder — the strongest constraint, and the worst performer
(over-constrained).

---

## One-line summary

> `dc_*` (graph as a **fixed TF-activity transform**) carries real biology and helps under low
> data; `grn_*` (graph baked into a **learned encoder**) hurts, and where it's least bad is the
> **decoder** placement. Neither beats plain **PCA** on the supervised probe — but the biological
> ones win the **unsupervised clustering** metric.

Full numbers: [Results](results.md) · how they're measured: [Evaluation](evaluation.md).
