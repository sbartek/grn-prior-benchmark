# Concepts — a plain-language biology primer

Everything you need to follow this project, from first principles. No biology background assumed.

## DNA → mRNA → protein

**DNA** is the cell's permanent master library — the full set of ~20,000 genes, identical in
every cell of your body. A **gene** is one stretch of it.

To *use* a gene, the cell makes a temporary working copy of it called **mRNA** (messenger RNA).
This is **transcription**. The mRNA is then read to build a **protein**, the molecule that
actually does work in the cell. This one-way flow is the "central dogma":

```
DNA (gene)  ──transcription──▶  mRNA  ──translation──▶  protein
master copy                    working copy            the machine
(all genes, permanent)         (one gene, temporary)
```

**Key consequence:** every cell has the *same* DNA, so DNA can't explain why a T-cell differs
from a B-cell, or a sick donor from a healthy one. The difference is **which genes each cell
transcribes, and how much**. That choice *is* the cell's biological state.

## Gene expression = mRNA counts

**Gene expression** = how much mRNA of a gene is present. Lots of mRNA → gene is "highly
expressed" (turned on); little/none → "silent". scRNA-seq (single-cell RNA sequencing) measures
this by counting captured mRNA molecules per gene, per cell. So a cell is a vector of ~20,000
counts — its molecular fingerprint:

```
cell = [ CD3D: 42,  MS4A1: 0,  ACTB: 310,  ... ]   # one count per gene
```

## Why single cells are noisy

Each cell is measured **shallowly**: only a random fraction of its mRNA is captured. So a gene
that is truly active often reads **0** in a given cell ("dropout" — a measurement failure, not
biological silence). Individual cells are sparse and unreliable.

## Pseudobulk = averaging away the noise

The failures are **random per cell**, so averaging many cells of the *same kind* cancels them
out — like stacking 500 blurry photos of one object into a sharp one.

We group cells by **(donor × cell type)** and average within each group:

```
donor 7's T-cells   (avg of ~2,000 cells)  → one clean pseudobulk vector
donor 7's B-cells                          → another
donor 8's T-cells                          → ...
```

108k noisy cells collapse into ~540 stable **pseudobulk profiles**. We combine *repeated noisy
looks at the same state*, not different information — the signal survives, the noise averages out.
(The take-home requires working at this pseudobulk level, not on raw single cells.)

## Transcription factors and the GRN

Cells don't transcribe genes at random. **Transcription factors (TFs)** are proteins that bind
DNA and switch other genes on (**activation, +1**) or off (**repression, −1**). A
**Gene Regulatory Network (GRN)** is the wiring diagram of these TF→target relationships.

**DoRothEA** is a public, curated GRN: ~32,000 signed TF→target edges over 429 TFs and ~9,300
genes, with confidence levels A (curated) → C (predicted). It is *known biology about which
genes control which*.

## Embeddings

A pseudobulk profile is still ~9,000-dimensional. An **embedding** is a learned, low-dimensional
summary (e.g. 64 numbers) produced by an **encoder**, meant to capture the sample's biological
state while discarding noise:

```
pseudobulk (9,000 genes) ──encoder──▶ embedding (64) ──decoder──▶ reconstruction (9,000 genes)
```

A **good** embedding is one where a simple classifier can read off biology (T-cell vs monocyte,
RA vs healthy) from those 64 numbers — **not** one that merely reconstructs the input accurately.
An autoencoder can rebuild all 9,000 genes perfectly while its embedding encodes mostly
donor/batch junk: good reconstruction, useless embedding. That trap is exactly what the
evaluation is built to expose.

## The "GRN prior" in one sentence

Instead of letting the encoder treat all 9,000 genes as unrelated, we **bias it with the
DoRothEA wiring** ("these genes are co-regulated, compress them together"). The benchmark asks:
does that biological inductive bias produce embeddings that capture state better — especially
when data is scarce or noisy — and does the benefit **disappear when the graph is corrupted**
(proving it was biology, not just sparsity)?
