# Technical Memo — GRN-Prior Expression Embedding Benchmark

**Question.** Does a graph-aware encoder using DoRothEA TF→target edges produce *better*
pseudobulk expression embeddings than a matched non-graph baseline — where "better" means the
embedding captures biological state (cell type, disease), not that it reconstructs expression —
and does any benefit hold up under low-data, noise, and graph corruption?

**Answer (short).** Largely no, with an instructive nuance. On the trustworthy readout (cell
type) the *full* DoRothEA prior (A+B+C) underperforms both PCA and an unconstrained autoencoder
everywhere. Restricting to high-confidence edges (confidence **A**) removes almost all of that gap
— but a **density ablation with degree-preserving rewired controls shows this is mostly a sparsity
/ regularization effect, not regulatory biology**: a rewired A-graph does as well or better under
low data. The *specific* topology contributes only a small, consistent lift over its rewired
control at full data and under noise (~0.02–0.05 F1), and it never beats the plain baseline.
Disease is not decodable from held-out donors, so it cannot adjudicate. Net: a fair, mostly
negative result where what *looks* like "the prior helps" is explained by density, not biology.

## Data & task
CELLxGENE RA PBMC (`d18736c3…`): 108,717 cells → **pseudobulk by (donor × cell_type)**, 536
groups → **500 kept** (≥10 cells), CP10K+log1p, 21,572 genes. Two readouts: **cell type**
(15 classes, primary) and **disease** (RA vs normal, secondary). All scoring uses **donor-grouped
5-fold CV** with the encoder retrained on train donors inside each fold — so neither encoder nor
probe ever sees a held-out donor. Metric: macro-F1, mean over 2 seeds.

## Is the dataset suitable?
Confirmed on data: **balanced 18 RA / 18 normal**, **no sex confound** (12F/6M in both arms),
**single assay** (10x 3′ v3, so no assay–disease confound), 15 cell types (some rare). The design
is **between-subjects**, so disease is confounded with **donor identity**. Consequence: cell type
is the trustworthy readout; **disease is reported as suggestive only**.

## What was compared
Shared autoencoder (gene → hidden(411) → z=64 → … → gene), MSE reconstruction, **no labels**. The
*only* structural difference is the first encoder layer:
- **baseline** — dense; **PCA** — linear floor.
- **grn_real** — masked so each hidden unit is a TF aggregating its regulon (effective weight =
  mask·sign·softplus(raw)); 8,376 genes × 411 TFs, 30,609 edges.
- **graph controls at matched density** — degree-preserving **rewired**, **sign-shuffled**,
  **random**. All share the identical gene set, so any difference is graph *structure*, not
  feature selection. Real-vs-controls is capacity-matched exactly (same #params, same #edges).

## Results

**Cell type (macro-F1).** The prior hurts, everywhere:

| condition | PCA | baseline | grn_real | grn_rewired | grn_sign_shuf | grn_random |
|---|---|---|---|---|---|---|
| full | **0.868** | 0.791 | 0.699 | 0.650 | 0.699 | 0.731 |
| lowdata k=4 | – | **0.540** | 0.485 | 0.457 | – | – |
| lowdata k=8 | – | **0.648** | 0.553 | 0.525 | – | – |
| lowdata k=16 | – | **0.727** | 0.653 | 0.599 | – | – |
| noise p=0.3 | – | **0.743** | 0.609 | 0.545 | – | – |
| noise p=0.1 | – | **0.661** | 0.509 | 0.448 | – | – |

**Disease (macro-F1).** Everything sits at ~0.42–0.49 (≈ chance for this binary task); no model
decodes disease from held-out donors. The donor confound dominates, as predicted. Not adjudicative.

**Donor leakage** (kNN donor accuracy, lower = less leakage): baseline 0.086, random 0.098,
grn_real 0.114, sign_shuf 0.132, rewired 0.136, PCA 0.166. The prior does **not** reduce
donor/batch signal; the baseline leaks least.

**Density ablation (confidence A / A+B / A+B+C, with rewired controls; cell type macro-F1).**

| condition | baseline | real_A | rewired_A | real_AB | real (ABC) |
|---|---|---|---|---|---|
| full | 0.791 | 0.788 | 0.767 | 0.745 | 0.699 |
| lowdata k=8 | 0.648 | 0.671 | **0.686** | 0.586 | 0.553 |
| noise p=0.3 | 0.743 | 0.705 | 0.655 | 0.638 | 0.609 |

Sparser, higher-confidence graphs are *better* (real_A ≫ real). But real_A vs rewired_A is the
tell: under low data the **rewired** A-graph is best (0.686 > 0.671), so that regime's gains are
sparsity, not biology; only at full data (0.788 vs 0.767) and under noise (0.705 vs 0.655) does
the true topology add a small margin over its rewired control.

## Interpretation — did the GRN help, and where?
1. **No, on the primary readout.** grn_real trails baseline and PCA at every condition. The
   inductive bias costs more (each hidden unit sees only its regulon) than the biology it adds.
2. **The decisive corruption test is negative at full data.** grn_real (0.699) does **not** beat
   its same-density controls: sign-shuffled ties it (0.699) and **random is higher (0.731)**. So
   the *specific* DoRothEA biology adds nothing over a random graph of the same density — what
   little the mask does is explained by sparsity, not regulatory content.
3. **A faint, honestly-reported structural signal — and density is the bigger lever.** Real vs
   rewired shows some biologically-relevant structure survives (full-data and noise, ~0.02–0.05),
   but the density ablation makes clear that most of what improves the prior is *sparsity*: the
   A-subset ≈ baseline, yet a rewired A-graph matches it, and under low data the rewired graph is
   best. So the specific regulatory topology is a minor factor; graph *density/regularization*
   dominates, and neither beats the baseline.
4. **Biology vs batch.** Cell type (a strong, largely linear signal — PCA tops the table) is
   captured best by the *least* constrained methods; disease (confounded) by none. The prior buys
   neither better biology nor lower donor leakage.

## Why the prior likely didn't help here
- Cell-type identity in PBMC is high-variance and near-linear; PCA already nails it, leaving no
  gap for a regulatory inductive bias to fill.
- Pseudobulk averaging already removes most technical noise, shrinking the regularization
  advantage a prior might offer on raw single cells.
- A **hard mask + fixed sign** is a rigid encoding of the prior: it discards capacity and forces
  activation/repression directions that may be noisy in DoRothEA (esp. C-confidence edges). A
  *soft* prior (graph-Laplacian penalty, graph init, or GNN message passing) might behave better
  and is untested here.

## Limitations
Single dataset; n=500 pseudobulk; disease confounded with donor; 2 seeds; no hyperparameter
search; one prior-encoding (hard mask) of many; probes limited to logistic/kNN. The density
ablation used one low-data / noise point each rather than a full grid.

## What would make the experiment biologically stronger
- A readout that is genuinely *regulatory* (predict perturbation response or measured TF activity)
  rather than cell-type identity, which linear methods already solve.
- **Within-subjects** perturbation data, so biological state is not aliased with donor.
- Multiple datasets + assays for external validity; sweep DoRothEA confidence/density; test a
  **soft** graph prior (Laplacian reg / GNN) as an alternative to the hard mask.

## What I deliberately left out (48h scope)
Raw single-cell modeling; STATE/metabolic/pathway priors (out of scope); GNN (masked-MLP was the
primary, pure-torch choice); hyperparameter search; backup datasets. I prioritized a *fair,
capacity-matched* baseline-vs-prior-vs-corruption comparison (plus a density ablation with
rewired controls) on one readout over broad but shallow coverage.
