# GRN-Prior Expression Embedding Benchmark

*Tolemy Bio — Founding ML Engineer take-home.*

## The question

> Can a **graph-aware encoder** using **DoRothEA regulatory edges** produce **better pseudobulk
> expression embeddings** than a matched **non-graph baseline** — especially under **low-data,
> noisy, and graph-corruption** conditions?

Here **"better"** means the embedding captures **useful biological state** (cell type, disease),
**not** that it reconstructs gene expression well.

## The one idea to hold onto

We **optimize reconstruction** (self-supervised, no labels) but **evaluate on biology**
(frozen probes on held-out donors). The gap between the two is the whole point: does the GRN
prior make biology *fall out for free*?

## The decisive test

Compare the **real** DoRothEA graph against a **degree-preserving rewired** graph. If the
corrupted graph helps just as much, any benefit is **structured sparsity / regularization, not
biology**. This is the comparison the exercise is really about.

## Headline finding

**How you use the GRN matters more than whether you use it.** Baked into a learned encoder
(hard/soft mask) it *hurts* and a rewired graph works as well (regularization, not biology). Used
the classical way (decoupler TF-activity transform) it carries *real* signal — beats a
matched-dimension random projection — and **beats PCA/baseline when data is scarce**, though it
only ties PCA at full data. Not DoRothEA-specific (CollecTRI ≥ it); replicates on a second dataset.
See [Results](results.md) and the [memo](https://github.com/sbartek/grn-prior-benchmark/blob/main/memo/memo.md).

## Navigate

- **[Concepts](concepts.md)** — plain-language biology primer (DNA → mRNA → expression →
  pseudobulk → GRN → embeddings).
- **[Methods & steps](methods.md)** — the end-to-end pipeline, Step -1 through Step 9.
- **[Results](results.md)** — findings, ablations, figures (filled as experiments complete).

Design rationale lives in [`PLAN.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/PLAN.md);
the final interpretation lives in [`memo/memo.md`](https://github.com/sbartek/grn-prior-benchmark/blob/main/memo/memo.md).
