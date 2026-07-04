# Crash course: DNA → RNA → proteins → this project

A from-scratch primer, built to be read in small chunks (each ≤ 5 min) with an optional video
per chunk. Goal: enough biology to understand — and *defend* — every choice in this benchmark.
No background assumed.

> **How to use this:** read one chunk, watch the video only if you want it to stick, move on.
> Chunk 0 = the cells themselves. 1–3 = core biology. 4 = the regulatory network (the "GRN" in
> the title). 5 = how it's measured. 6–7 = the actual project. 8 = a cheat-sheet to skim before
> the interview.

---

## Chunk 0 — Meet the cells (the cast) *(~4 min)*

Our data is **PBMC** — *peripheral blood mononuclear cells*, i.e. the immune cells floating in a
blood sample. Different cell **types** do different jobs, and telling them apart is the main task
in this project. The headliners:

- **T-cells** — the soldiers/coordinators. Patrol for infected or abnormal cells; either kill
  them or direct the wider immune response. (Mature in the **T**hymus.) They come in subtypes:
  - **CD4 "helper" T-cells** — coordinate other immune cells.
  - **CD8 "killer" T-cells** — destroy infected cells directly.
- **B-cells** — the antibody factories. Make **antibodies**, proteins that lock onto a specific
  pathogen to neutralize or flag it. (Mature in the **B**one marrow.)
- **NK (natural killer) cells** — fast first-responders that kill stressed/infected cells without
  needing prior exposure.
- **Monocytes / dendritic cells** — the "big eaters + messengers": engulf pathogens and show
  their fragments to T-cells to kick off a targeted response.

Our dataset labels **15 fine-grained types** (e.g. *naive vs memory vs effector* CD4/CD8 T-cells,
naive/memory B-cells, classical/non-classical monocytes, NK cells, dendritic cells). Whenever this
crash course says "T-cell vs B-cell," picture two of these 15 labels — the thing a good embedding
should be able to separate.

**Why they differ (preview of Chunk 1):** every one of these cells carries the *same DNA*; they
differ because each **switches on a different set of genes**. That difference is what we measure
and model.

🎥 *Optional:* [Amoeba Sisters — Immune System](https://www.youtube.com/watch?v=fSEFXl2XQpc) (9 min)

---

## Chunk 1 — DNA: the master library *(~3 min)*

**DNA** is the cell's permanent instruction set: ~20,000 **genes**, the same copy in every cell
of your body. Think of it as a reference library that never leaves the building — you don't
check the master books out, you photocopy the page you need.

- A **gene** = one stretch of DNA that codes for one product.
- Same DNA in every cell ⇒ DNA *alone* can't explain why a T-cell differs from a B-cell, or a
  sick donor from a healthy one. **The difference is which genes each cell uses.**

Hold onto that last line — it's the reason this whole field exists.

🎥 *Optional:* [Amoeba Sisters — DNA vs RNA (Updated)](https://www.youtube.com/watch?v=JQByjprj_mA) (~8 min)

---

## Chunk 2 — RNA: the working copy (the "central dogma") *(~4 min)*

To *use* a gene, the cell makes a temporary copy of it out of **RNA**, specifically **messenger
RNA (mRNA)**. This copying is **transcription**. The mRNA then gets read to build a **protein**
(**translation**). One-way flow:

```
DNA (gene) ──transcription──▶ mRNA ──translation──▶ protein
master copy                   working copy          the machine that does work
```

This is the **central dogma of molecular biology**. Two things to remember:

1. mRNA is *disposable and made on demand* — a cell makes lots of mRNA for genes it's actively
   using, little or none for genes it isn't.
2. So the amount of a gene's mRNA right now = a readout of **how "on" that gene is** in that cell.

🎥 *Optional:* [The Central Dogma: Transcription and Translation](https://www.youtube.com/watch?v=yLQe138HY3s) (5 min)

---

## Chunk 3 — Proteins & "gene expression = mRNA counts" *(~4 min)*

**Proteins** are the molecules that actually do things — structure, signaling, enzymes, immune
receptors. A cell's identity and behavior come from *which proteins it makes*, which comes from
*which genes it transcribes into mRNA*.

**Gene expression** is just "how much mRNA of a gene is present." High mRNA → gene is "highly
expressed" (on); little → "silent" (off). Measuring every gene's mRNA gives a cell's molecular
fingerprint — a vector of ~20,000 numbers:

```
cell = [ CD3D: 42,  MS4A1: 0,  ACTB: 310,  ... ]   # one count per gene
```

**Key idea for the project:** "gene expression," "mRNA counts," and "the data" are the same
thing. Everything we model is these count vectors.

---

## Chunk 4 — Gene regulation & the GRN (this is the "prior") *(~5 min)*

Cells don't transcribe genes at random. Special proteins called **transcription factors (TFs)**
bind DNA and switch other genes **on** (activation) or **off** (repression). That control system
— which TF regulates which genes — is a **Gene Regulatory Network (GRN)**.

- A TF + all the genes it controls = a **regulon**.
- The GRN is essentially a **wiring diagram** of the cell's control logic. It's *known biology*,
  catalogued in public databases.
- **DoRothEA** (the database in this project) is one such catalogue: ~430 TFs → ~9,000 target
  genes, each edge signed **+1** (activates) or **−1** (represses), with confidence levels A→C.

**Why we care:** the raw mRNA numbers are noisy, but the *regulatory structure* behind them is
prior knowledge. The project's whole question is whether feeding a model this wiring diagram —
"these genes are controlled together" — produces better representations than treating all genes
as unrelated. That injected knowledge is the **"GRN prior."**

🎥 *Optional:* [Amoeba Sisters — Gene Expression and Regulation](https://www.youtube.com/watch?v=ebIpkw3XapE) (8 min)

---

## Chunk 5 — How it's measured: scRNA-seq, dropout, pseudobulk *(~5 min)*

**scRNA-seq** (single-cell RNA sequencing) counts mRNA per gene *in each individual cell* —
giving a big `cells × genes` matrix (our dataset: ~108k cells × ~20k genes).

The catch: each cell is measured **shallowly**, so a gene that's truly active often reads **0**
by chance. That's **dropout** — a measurement failure, not biological silence. Single cells are
sparse and noisy.

**The fix = pseudobulk.** Because the failures are random per cell, averaging many cells of the
same kind cancels the noise (like stacking blurry photos into a sharp one). We group cells by
**(donor × cell type)** and average within each group:

```
donor 7's T-cells  (avg of ~2,000 cells) → one clean pseudobulk profile
```

108k noisy cells collapse into ~500 stable pseudobulk profiles. **The project works at this
pseudobulk level**, as the brief requires.

🎥 *Optional:* [Intro to single-cell RNA-seq: empties, doublets, dropouts, UMIs](https://www.youtube.com/watch?v=8o6hspZwIYY) ·
[StatQuest — a gentle intro to RNA-seq](https://www.youtube.com/watch?v=tlf6wYJrwKY)

---

## Chunk 6 — Embeddings & the project question *(~5 min)*

A pseudobulk profile is still ~9,000-dimensional. An **embedding** is a learned, low-dimensional
summary (e.g. 64 numbers) from an **encoder**, meant to capture the sample's *biological state*
while dropping noise:

```
pseudobulk (9,000 genes) ──encoder──▶ embedding (64) ──decoder──▶ reconstruction
```

A **good** embedding is one where a simple classifier can read biology off it (T-cell vs
monocyte, sick vs healthy) — **not** one that merely rebuilds the input. An autoencoder can
reconstruct all 9,000 genes perfectly while its embedding encodes mostly batch junk: good
reconstruction, useless embedding. Avoiding that trap is the point of the evaluation.

**The project question in one line:** does building the DoRothEA GRN into the encoder produce
embeddings that capture biology *better* than a plain encoder — especially when data is scarce
or noisy — and does any benefit *survive corrupting the graph* (proving it's biology, not just
structure)?

🎥 *Optional:* [StatQuest — PCA main ideas in 5 minutes](https://www.youtube.com/watch?v=HMOI_lkzW08)
(PCA is our simplest baseline embedding — worth the 5 min)

---

## Chunk 7 — What we did & what we found *(~5 min)*

**Setup.** Pseudobulk the RA PBMC data → predict **cell type** (trustworthy) and **disease**
(confounded, see cheat-sheet) from frozen embeddings, always testing on **held-out donors**.
Compare: PCA, a plain autoencoder (**baseline**), and GRN-aware encoders.

**Three ways we injected the prior:**
1. **Hard mask** — force each hidden unit to be a TF regulon. → *hurts.*
2. **Soft penalty** — nudge, don't force. → *more prior = worse; still hurts.*
3. **TF-activity** (decoupler ULM, the standard way) — use the GRN as a *fixed feature
   transform*. → *this one beats the baseline* (though partly because it has more dimensions).

**The decisive control.** We swap the real graph for a **degree-preserving rewired** one
(scrambled biology, same shape). It works just as well → so gains from graph *sparsity* are
regularization, **not** the specific regulatory biology.

**Bottom line.** Baking the GRN into a deep encoder doesn't help; using it the classical way
(TF activity) does; and apparent structural benefits are mostly dimensionality/regularization.
This matches the literature (biology-wired nets rarely beat simple baselines on accuracy — their
value is interpretability + small-data). A fair, carefully-scoped result — which is exactly what
the brief asked for.

---

## Chunk 8 — Interview cheat-sheet *(skim, ~3 min)*

- **The cells (PBMC):** T-cells (CD4 helper / CD8 killer), B-cells (antibodies), NK cells,
  monocytes/dendritic cells. Same DNA, different genes on → different types (15 in our data).
- **Central dogma:** DNA → (transcription) → mRNA → (translation) → protein.
- **Gene expression = mRNA amount.** Our data = mRNA counts per gene per cell.
- **Dropout:** random zeros from shallow sampling. **Pseudobulk:** average cells per
  (donor × cell type) to kill dropout. We model pseudobulk, not raw cells.
- **TF / regulon / GRN:** transcription factor controls target genes; DoRothEA = signed TF→gene
  catalogue = our **prior**.
- **Embedding:** low-D summary judged by whether *biology* (not reconstruction) is recoverable.
- **Donor confound (say this unprompted):** the RA data is *between-subjects* — each donor is
  only sick or healthy — so "predicting disease" can secretly be "predicting donor." That's why
  we split by donor and treat disease as suggestive only; cell type is the trustworthy readout.
- **Our headline:** GRN-as-encoder-constraint = no help; GRN-as-TF-activity = helps but partly
  dimensionality; real graph ≈ rewired graph ⇒ regularization, not biology.
- **Why fair:** all models share gene set, capacity, and training budget; corrupted-graph and
  matched-dimension controls guard against the obvious alternative explanations.

*Deeper design rationale: [`PLAN.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/PLAN.md).
Full results + interpretation: [`memo/memo.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/memo/memo.md).*
