# Interview Vocabulary Test — GRN Benchmark

Test yourself before the Caelan call. Cover the answers, say the answer out loud, then reveal.

---

## Q1. What does "pseudobulk" mean, and why is it called that?

<details>
<summary>Answer</summary>

**Pseudo-bulk = fake bulk.**

- *Bulk RNA-seq* is the old technique: sequence all mRNA from a tissue chunk together, get one averaged measurement.
- *Pseudobulk* takes single-cell data and groups cells (e.g. by donor × cell type), then sums their counts — simulating what bulk RNA-seq of that group would have measured.
- In this project: 108K cells → 500 pseudobulk samples (donor × cell_type), each with summed counts across genes.

</details>

---

## Q2. What does UMI stand for, and why do we need it?

<details>
<summary>Answer</summary>

**Unique Molecular Identifier** — a short random DNA barcode attached to each mRNA molecule *before* PCR.

Without UMIs, PCR amplification would inflate counts (1 original mRNA copied 1000× looks like 1000 different mRNAs). Counting *distinct UMIs* gives you the true number of original mRNA molecules.

</details>

---

## Q2a. What is a molecule?

<details>
<summary>Answer</summary>

A group of atoms bonded together into one physical unit.

In this project:
- **mRNA molecule** = one single strand of RNA (~500-5000 nucleotides). Carries the coded instructions from one gene.
- **DNA molecule** = a chromosome (very long nucleotide strand).
- **Protein molecule** = a folded chain of amino acids.

"Counting mRNA molecules": inside a cell, gene TP53 might have 12 physical mRNA strands floating around. `X[cell, TP53] = 12` = the sequencer detected 12 distinct TP53-mRNA strands via their UMIs.

Scale: a cell has ~100K-1M total mRNA molecules across ~20K genes.

</details>

---

## Q3. What's the difference between a gene and an mRNA?

<details>
<summary>Answer</summary>

- **Gene** = a stretch of DNA. Physical, permanent, one per cell (well, per chromosome).
- **mRNA** = a temporary RNA copy made *from* a gene when the cell wants to express it.
- Flow: `DNA → mRNA → protein`
- scRNA-seq counts **mRNA molecules per gene** — how much each gene is being used.

</details>

---

## Q4. What does CP10K normalization do, and why is it needed?

<details>
<summary>Answer</summary>

**Counts Per 10,000.** Divide each sample's row by its total, multiply by 10,000. Every sample now sums to 10K.

Needed because pseudobulk groups have different sizes (10 cells vs 800 cells) → raw sums aren't comparable. CP10K puts every sample on the same scale.

</details>

---

## Q5. What is a regulon?

<details>
<summary>Answer</summary>

The set of target genes controlled by one transcription factor, with signs (+1 activate / −1 repress).

Example: STAT1's regulon might be 80 genes — say 50 that STAT1 activates and 30 it represses.

A GRN (gene regulatory network) like DoRothEA is a collection of regulons — one per TF. In this project: 411 TFs × their regulons = ~30K signed edges.

</details>

---

## Q6. Why is the gene set restricted to 8,376 (not the raw 61,497)?

<details>
<summary>Answer</summary>

Because the `grn_real` mask can only connect a TF (hidden unit) to genes in its regulon. Genes not in any regulon can't participate in the mask.

Restricting all models to the DoRothEA-covered genes makes the comparison fair: every model (baseline / grn_real / dc_tfact) sees the same 8,376 input features. Only the mask differs.

</details>

---

## Q6a. What's an activator vs a repressor?

<details>
<summary>Answer</summary>

Both are **transcription factors** (proteins that bind DNA near a gene) — the difference is the effect:

- **Activator** = TF whose binding **increases** transcription of the target gene → more mRNA → gene "on."
- **Repressor** = TF whose binding **decreases** transcription of the target gene → less mRNA → gene "off."

In a signed regulon:
- `+1` = TF activates that target
- `−1` = TF represses that target
- `0` = no known regulation

Many TFs are context-dependent — the same TF can activate one gene and repress another. Some can even switch role on the same gene depending on cofactors.

</details>

---

## Q7. How does a transcription factor activate or repress a gene?

<details>
<summary>Answer</summary>

A TF is a protein that binds a specific short DNA sequence near a gene.

- Activate: TF binds → recruits RNA polymerase → gene transcribed more → mRNA up.
- Repress: TF binds → blocks polymerase or recruits shutdown complexes → mRNA down.

The sign (+1 / −1) in a regulon reflects which effect dominates for that TF-target pair.

</details>

---

## Q8. Where does the TF protein come from — how does the cell make it?

<details>
<summary>Answer</summary>

The TF is encoded by a gene like any protein:

`TF-gene (DNA) → TF-mRNA → TF-protein`

The cell transcribes the TF-gene when it needs the TF, translates the mRNA to protein, and the protein binds its target sites in the nucleus. TF proteins get degraded and recycled.

Many TFs are regulated by other TFs → the GRN is a network of TFs regulating each other and non-TF genes.

</details>

---

## Q9. Is the GRN a DAG (acyclic)?

<details>
<summary>Answer</summary>

**No.** GRNs contain cycles — feedback loops are fundamental.

Common patterns:
- Auto-regulation (TF activates or represses its own gene)
- Mutual repression (bistable switch)
- Feed-forward loops (temporal filter)

Implications:
- Can't just topologically sort and propagate signal.
- Standard DAG methods (Bayesian networks etc.) don't work directly.
- ULM sidesteps cycles: it measures target-gene expression, doesn't simulate dynamics.

</details>

---

## Q10. What does ULM stand for and what does it compute?

<details>
<summary>Answer</summary>

**Univariate Linear Model.** Per (sample, TF): fit `y = a + b·x`, where:
- `y` = sample's log-normalized expression vector (length ~8,376 genes)
- `x` = TF's signed regulon indicator (+1 activator, −1 repressor, 0 non-target)

Output: the **t-statistic of the slope b**. That number is the TF activity score for (sample, TF).

Run for 500 samples × 293 TFs → the `dc_tfact` matrix (500 × 293).

</details>

---

## Q11. Which models in the project use ULM?

<details>
<summary>Answer</summary>

All `dc_tfact*` variants (the "graph-as-fixed-transform" arm):

- `dc_tfact` — DoRothEA
- `dc_tfact_collectri` — CollecTRI (robustness check)
- `dc_tfact_rewired` — degree-preserving rewired DoRothEA (topology null)
- `dc_tfact_pca` — DoRothEA + PCA-64 (dim-match control)

NOT used in: pca, baseline AE, grn_real / grn_soft / grn_decoder / grn_symmetric, rand_proj.

</details>

---

## Q12. Do we train ULM?

<details>
<summary>Answer</summary>

**No.** ULM is per-sample deterministic. Each sample gets its own tiny regression using only its own expression vector — no cross-sample parameter sharing.

Consequence: no train/test discipline needed. `dc_tfact` can be computed on the full 500 samples without leakage. That's why the code comment reads *"Per-sample transform → no leakage."*

Contrast with AEs: those have shared weights trained on many samples; using an AE trained on all 500 to evaluate on the same 500 IS leakage (the R1 clustering confound).

</details>

---

## Q13. Are ULM's coefficients ±1?

<details>
<summary>Answer</summary>

Two different things:

- **The regulon signs** (input `x`): fixed at +1 / −1 / 0. Given by DoRothEA, not learned.
- **The regression coefficients** (`a`, `b`): fitted real numbers. `b` is a real-valued slope.
- **The output activity score** = t-statistic of `b` = `b / std_err(b)`. Real number, typically −5 to +5.

Signs come from biology (fixed). Score is fitted (per sample × TF).

</details>

---

## Q14. What's `y` in the ULM regression?

<details>
<summary>Answer</summary>

`y` is the log-normalized expression vector for that sample:

`raw counts → pseudobulk sum → CP10K → log1p → y`

So `y[gene]` = `log(1 + gene's fraction of 10K UMIs in this sample)`. Real-valued, roughly Gaussian in shape.

</details>

---

## Q15. Why log-transform first, then run a *linear* regression?

<details>
<summary>Answer</summary>

Because gene expression is naturally **multiplicative**, not additive. Doubling a TF's activity roughly doubles its targets — a multiplicative effect.

Log converts multiplication to addition: `log(2E) = log(2) + log(E)`. On the log scale, "twice as active" always adds ~0.7, regardless of baseline expression.

So linear regression on log-space is equivalent to a **multiplicative (log-linear) model** on raw scale — the correct GLM framing for RNA-seq. Same principle DESeq2, edgeR, Limma-voom all use.

</details>

---

## Q16. Why ULM instead of a plain signed weighted sum (wsum)?

<details>
<summary>Answer</summary>

Wsum = signed dot product `sum(x · y)` — a valid method, but doesn't normalize for sample noise.

ULM adds normalization via t-stat = `slope / std_err`:

- **slope** = the signal (targets vs non-targets shift)
- **std_err** = the noise (how scattered the fit residuals are)
- **t-stat = signal / noise ratio**

Two samples with the same wsum score can have very different t-stats:
- Clean regulon pattern → small residuals → big t-stat
- Noisy expression that happens to sum the same → big residuals → small t-stat

ULM downweights samples where the regulon pattern is unreliable. Wsum can't do that.

</details>

---

## Q17. Do we use ULM p-values anywhere?

<details>
<summary>Answer</summary>

**No.** We keep only the t-statistic — it becomes the feature for downstream models.

P-values would be a mess:
- 500 × 293 = ~147K comparisons → multiple-testing burden.
- Thresholding to binary "significant / not" throws away continuous signal.
- Downstream classifier / KMeans want continuous features.

P-values are for humans reading tables, not for downstream models.

</details>

---

## Q9a. What is a GRN?

<details>
<summary>Answer</summary>

**Gene Regulatory Network.** A directed graph of TF → target gene relationships with signs (+1 activate / −1 repress).

DoRothEA and CollecTRI are specific human GRN resources — built by curating experiments + literature + computational predictions.

In the memo title: *"GRN-prior expression embedding benchmark"* = testing whether a GRN improves ML embeddings.

</details>

---

## Q9b. In decoupler's network format, what does the `weight` column mean?

<details>
<summary>Answer</summary>

**Signed strength of the TF → target relationship.**

For DoRothEA / CollecTRI: essentially ±1 (mode of regulation). Not to be confused with `confidence` (a separate column with A/B/C/D/E letters for how well-supported the edge is).

Other GRNs (e.g. PROGENy) can have continuous weights — ULM handles any weight value. In your tangled toy: primary edges ±1, secondary ±0.5.

</details>

---

## Q9c. If a TF represses a gene, does that mean the mRNA is absent?

<details>
<summary>Answer</summary>

**No — repression reduces mRNA, rarely to zero.** Repression is quantitative:
- Strong repressor + high TF activity → target at 10-30% of baseline.
- Weak repressor → target at 60-80%.
- True silencing (chromatin condensation) is a much stronger mechanism.

Also: even if a TF is "expressed" (its mRNA is high), it may not be *active* — needs translation to protein, correct localization to the nucleus, post-translational modification, cofactor binding. That's why ULM measures activity through the *targets*, not the TF's own mRNA.

</details>

---

## Q9d. Can we build a "protein regulation network" between TFs from a GRN?

<details>
<summary>Answer</summary>

**Partially — a subgraph, but only one layer.**

**What you can extract:** filter the GRN to edges where the target is itself a TF → the TF-TF transcriptional subnetwork. Captures feedback loops, cascades, mutual repression.

**What the GRN misses:**
- Protein-protein binding / dimerization (e.g. MYC-MAX).
- Phosphorylation by kinases.
- Ubiquitination / degradation.
- Sequestration (e.g. NFκB held by IκB in cytoplasm).

Those live in different resources: STRING / BioGRID (PPI), PhosphoSitePlus (kinase-substrate), Reactome / KEGG (pathways). A "full control panel for the cell" would combine all these layers.

</details>

---

## Q9e. What does the intercept `a` in the ULM regression mean biologically?

<details>
<summary>Answer</summary>

`a` is the predicted log-expression when `x = 0` — i.e. the **average log-expression of genes not touched by this TF** in this sample.

Not exactly "basal transcription" in the strict sense (that's a gene-level property). It's a sample-level, TF-level intercept that bundles:
- Basal transcription (contributes)
- Effects of every OTHER TF on non-target genes
- Housekeeping gene expression
- The sample's overall expression level (big / active cell → higher `a`)

Clean framing: `a` = sample's baseline expression ignoring this TF. `b` = how much this TF's regulon lifts targets above that baseline.

</details>

---

## Q17c. What is scIB?

<details>
<summary>Answer</summary>

**Single-cell Integration Benchmark** (Luecken et al. 2022, Nature Methods) — the standard suite of metrics for evaluating scRNA-seq embeddings.

Two axes:
- **Biology conservation** — cell-type macro-F1, ARI, NMI, cell-type ASW, cLISI
- **Batch removal** — batch ASW, iLISI, kBET

Your project uses the biology axis (macro-F1 + ARI/NMI). No batch removal needed — single assay, single study.

</details>

---

## Q17b. What is macro-F1 and what classifier do we use as the probe?

<details>
<summary>Answer</summary>

**Macro-F1:** compute F1 per class independently, then average across all 15 cell types with equal weight. Rare types count as much as common ones — robust to class imbalance.

Contrast:
- **Micro-F1** — weighted by class frequency; big classes dominate.
- **Weighted-F1** — also frequency-weighted.

**Probe classifier:** multinomial LogisticRegression (`sklearn`), `C=1.0`, max_iter=2000. Trained on `z` from held-out train donors, evaluated on held-out test donors.

**Why LR (not a deep classifier):** cleanest test of the *representation*. A strong classifier would confound "embedding is good" with "classifier is powerful." Linear probe isolates the embedding.

**`labels=` fix (S7):** `f1_score(..., labels=all_15)` forces macro-F1 to average over the same 15-class set every fold, even if some are absent from a fold's test set. Without it, folds average over different label sets → apples vs oranges.

</details>

---

## Q17a. What is softplus and why do we use it in MaskedLinear?

<details>
<summary>Answer</summary>

`softplus(x) = log(1 + exp(x))` — a smooth version of ReLU that's always positive.

Shape: → 0 as `x → −∞`, ≈ x as `x → +∞`, ≈ 0.69 at x=0.

**Why in MaskedLinear:** we need the trained magnitude to be positive (sign is fixed separately). Softplus lets `raw` be any real number (unconstrained gradient) while `softplus(raw) > 0` always. Prevents the trained weight from flipping the biological sign.

**Compared to alternatives:**
- `exp` — blows up faster.
- `ReLU` — zero gradient when raw < 0 (dead neurons).
- `abs` — non-smooth at 0.

Softplus = smooth + always-positive + always-nonzero-gradient.

</details>

---

## Q18a. What is the Poisson distribution, and how does it apply here?

<details>
<summary>Answer</summary>

**Poisson** = distribution of the count of independent events in a fixed interval. Single parameter `λ` (rate). `Mean = Var = λ`.

**In RNA-seq:** for one gene in one cell, `λ` = expected number of captured UMIs. Then `Y ~ Poisson(λ)`, where `Y` is the observed count.

`λ` bundles: (true expression) × (capture efficiency) × (sequencing depth).

**Key properties:**
- Sums of Poissons are Poisson → summing cells in pseudobulk stays in the family.
- Limit of Binomial(n, p) as n large, p small — matches the mRNA capture setup.

</details>

---

## Q18b. What is the Gamma distribution and why does it appear here?

<details>
<summary>Answer</summary>

**Gamma** = flexible distribution on positive reals. Two parameters: shape `k`, scale `θ`. Mean `kθ`, variance `kθ²`.

**Physical interpretation:** waiting time until the k-th event in a Poisson process (`k=1` → exponential).

**Why it appears in RNA-seq:**
- Positive-valued (perfect for a rate `Λ`).
- Flexible shape — bursty (k < 1) to concentrated (k large).
- **Conjugate prior for Poisson** — putting Gamma over the Poisson rate marginalizes to Negative Binomial cleanly.

</details>

---

## Q18c. Can λ vary between two cells of the same cell type? Why?

<details>
<summary>Answer</summary>

**Yes.** Sources:

1. **Cell size** — bigger cells = more total mRNA = higher λ for every gene.
2. **Cell cycle stage** — different phases, different expression.
3. **Bursting** — transcription happens in bursts; caught mid-burst vs between.
4. **Activation state** — resting vs stimulated cells differ dramatically.
5. **Sequencing depth** — technical: more reads per cell = higher λ everywhere.
6. **Capture efficiency** — technical: some cells lose more mRNA in prep.

That biological + technical `λ` variation is what the Gamma in the Gamma-Poisson mixture captures. CP10K normalizes out most of the size + depth effect; pseudobulking averages out cycle + bursting noise.

</details>

---

## Q18d. What's the distribution of log(Poisson)?

<details>
<summary>Answer</summary>

**Not a standard named distribution — but well-approximated as Gaussian for reasonable λ.**

Delta method: `log(Y) ≈ Normal(log(λ), 1/λ)` for `Y ~ Poisson(λ)`.

**Two key properties:**
- Roughly Gaussian → linear methods (regression, PCA) work well.
- **Variance stabilization** — raw Poisson: `Var = λ` grows with mean; after log: `Var ≈ 1/λ` shrinks with mean. Same differences mean the same thing across expression levels.

**Problem: log(0) undefined.** ~90% of scRNA-seq entries are zero. Solution: `log1p(Y) = log(1 + Y)`. `log1p(0) = 0` — no singularity; for `Y > 5` or so, `log1p ≈ log`.

**Alternatives that exist (not used here):** Freeman-Tukey `2·√(Y + 3/8)`, Anscombe, Pearson residuals (Lause 2021).

</details>

---

## Q18. Why is RNA-seq count data modeled as Negative Binomial, not the textbook "failures-until-success" interpretation?

<details>
<summary>Answer</summary>

The waiting-time interpretation doesn't fit RNA-seq. The one that does is the **Gamma-Poisson mixture** (aka "overdispersed Poisson"):

1. Each cell has its own true mRNA production rate `Λ` for a given gene.
2. Given `Λ`, the observed count is `Y | Λ ~ Poisson(Λ)` — pure counting sampling.
3. `Λ` varies across "identical" cells due to biology (bursting, cell state).
4. If `Λ ~ Gamma`, then marginally `Y ~ NegBin`.

**Mathematically:** `Y ~ NB(μ, α)` ⟺ `Λ ~ Gamma`, `Y | Λ ~ Poisson(Λ)`.

**Biological reading of the parameters:**

- `μ` = average production rate across cells for that gene.
- `α` = how much the rate varies across cells (biological + technical noise).
- `α → 0` → all cells identical → pure Poisson.
- Large `α` → bursty, heterogeneous cells → heavier tails.

This is why DESeq2 / edgeR use NB with a log link — they model `log(μ) = Xβ` and estimate `α` per gene.

</details>

---
