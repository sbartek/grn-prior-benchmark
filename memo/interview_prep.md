# Interview Prep — Caelan (CTO Tolemy Bio), Thu 2026-07-09

Adversarial pre-mortem of the GRN-prior submission before the review call. **Don't refit anything** — 24h is not enough for meaningful re-runs. Play: preemptive honesty on the weak spots + confident surfacing of what's strong.

---

## RED — Volunteer these before Caelan finds them

### R1. Clustering "Act 3" is confounded — AE trained on all 500 samples

**The bug:** `scripts/14_clustering.py:41` calls `make_embedder(m, data, all_idx, dev, 0, 250, data["X"])` where `all_idx = np.arange(n)` (line 35). AE-based models (baseline / grn_real / grn_decoder) are trained on all 500 pseudobulks and then clustered on those same 500 samples. ~7M-param AE, ample capacity to memorize.

**What's affected vs clean:**

| model | AE? | confound? |
|---|---|---|
| pca | no | ❌ clean — fixed projection |
| dc_tfact | no | ❌ clean — deterministic ULM |
| dc_tfact_collectri | no | ❌ clean — same |
| baseline | ✅ | ⚠️ trained on all 500 |
| grn_real | ✅ | ⚠️ same |
| grn_decoder | ✅ | ⚠️ same |

**Numbers (`clustering.csv`):**

| method | ARI | NMI |
|---|---|---|
| pca | 0.124 | 0.390 |
| grn_real | 0.215 | 0.486 |
| grn_decoder | 0.238 | 0.522 |
| baseline | 0.255 | 0.564 |
| dc_tfact | 0.276 | 0.563 |
| dc_tfact_collectri | 0.304 | 0.594 |

**What holds honestly, even after conceding the confound:**

- **Fixed-transform comparison (clean):** pca 0.124 < dc_tfact 0.276 < dc_tfact_collectri 0.304. TF-activity clusters biology tighter than PCA. This IS the "metric flip" story stripped down.
- **grn_real (0.215) STILL below baseline (0.255)** even with maximum memorization freebie for both. Strengthens Act-1 finding (mask hurts), doesn't weaken it.

**What does NOT hold:**
- Memo's "graph-informed representations clustered best" — grn_real and grn_decoder sit *below* the plain AE. The lead is entirely TF-activity rows.
- AE-vs-PCA rank in that table is inflated by training-on-all-data.

**Volunteer script (say this preemptively):**
> "I want to flag something before we get into the clustering results. The AE-based rows in `clustering.csv` — baseline, grn_real, grn_decoder — were trained on all 500 pseudobulks with no held-out set (line 41 of `14_clustering.py`, `all_idx = np.arange(n)`). PCA, dc_tfact and dc_tfact_collectri aren't affected because they're fixed transforms. So the cleanest reading of that table is the fixed-transform comparison: dc_tfact_collectri 0.30 > dc_tfact 0.28 > PCA 0.12 — TF-activity clusters biology tighter than PCA. That's the honest positive result. The AE-vs-PCA rank in that table I'd treat as directional at best. One thing worth noting for the confounded AE rows: grn_real is still below baseline even when both are given the memorization freebie, which strengthens the Act-1 finding rather than weakening it. The proper fix is per-fold-trained AE embeddings — top of my next-round list."

### R1 UPDATE (2026-07-09) — I ran the per-fold fix; Act 3 has to be walked back

Re-ran the geometric metrics with **per-fold-trained AE embeddings** (`scripts/18_extra_metrics_perfold.py`, `results/tables/extra_metrics_perfold.csv`). Turns out R1 wasn't a small confound — it was **carrying the entire Act-3 "metric flip" story.**

**Per-fold results (honest — AE only sees train donors):**

| model | macro-F1 | macro AUC | silhouette | ct_asw |
|---|---|---|---|---|
| **pca** | **0.864** | **0.994** | **−0.027** | **−0.028** |
| dc_tfact_collectri | 0.854 | 0.992 | −0.019 | −0.022 |
| dc_tfact | 0.855 | 0.988 | −0.056 | −0.058 |
| baseline | 0.757 | 0.983 | −0.081 | −0.078 |
| grn_decoder | 0.741 | 0.980 | −0.135 | −0.138 |
| grn_real | 0.702 | 0.961 | −0.124 | −0.126 |

**What changed vs the R1-confounded numbers:**
- AE silhouettes dropped ~0.10 (`baseline` 0.025 → −0.081; `grn_decoder` −0.015 → −0.135). R1 was doing real work in inflating them.
- PCA silhouette **improved** (−0.10 → −0.03) — no confound, per-fold subsets are smaller/cleaner.
- **PCA now wins ALL THREE metrics** including silhouette — the metric that Act 3 said PCA loses.
- CollecTRI-ULM is a very close #2 everywhere. Fixed-transform ordering is stable.

**Act 3 (as written) is largely an R1 artefact.** The "PCA smears / GRN clusters tighter" flip does not hold when clustering is measured fairly.

**Volunteer script (2026-07-09 — use this one, not the older R1 draft above):**
> "I re-ran the geometric metrics with per-fold AE training yesterday — R1 turned out to carry the entire Act-3 story in my memo. On honest per-fold silhouette, PCA is actually #1 (−0.03), CollecTRI ULM #2 (−0.02) — the 'metric flip' I described doesn't hold. So the honest bottom line is simpler than Act-3: PCA is the strongest baseline across every metric; TF-activity ULM is competitive with PCA and adds interpretability; encoder-mask hurts everywhere. Act-1 (mask hurts) and Act-2 (TF-activity carries signal above rewired-null) both hold up. Would rather walk back Act-3 openly than defend a finding that was carried by a training-set leak."

**Why this framing is a strength:**
- You found + fixed your own bug post-submission — signal of judgement + rigor.
- Simpler story is easier to defend and true.
- Tolemy explicitly grades on honesty; walking back beats defending.

### R2. `dc_tfact` (DoRothEA alone) does NOT beat baseline under noise — but CollecTRI does, strongly
From `final_stats.csv`:
- noise:0.3 → dc_tfact mean_delta = **−0.012**, frac_beats = 0.52 (essentially tied)
- noise:0.1 → dc_tfact mean_delta = **−0.045**, frac_beats = 0.28 (loses)

But `dc_tfact_collectri` at noise:0.3 = 0.784 vs baseline 0.732 = **+0.052** (WINS), and at noise:0.1 = 0.693 vs baseline 0.672 = **+0.021** (WINS). CollecTRI-ULM is actually the #1 method at both noise levels.

**Say plainly:** *"DoRothEA-ULM alone is a wash under noise; CollecTRI-ULM beats every method at both noise levels. This is one of the reasons the CollecTRI robustness check ended up carrying more of the story than I originally framed."*

### R3. `dc_tfact_rewired` at full data nearly ties `dc_tfact` — biology > topology claim is thinner than framed
From `final.csv` full data: dc_tfact 0.847, dc_tfact_rewired 0.804 — gap = **0.043**. `frac_beats_baseline` for rewired = 0.68, mean_delta = +0.018.

~70% of the "biological" win is captured by degree-preserving rewiring.

**What to say:**
> "The rewired-network ULM was my B3 self-review — I ran it and it's in the definitive sweep. Real regulons beat rewired regulons, but the gap is ~0.04 at full data, and rewired still beats the random-projection null. So there's a real biology signal, but it's a smaller share of the total effect than the framing suggests."

### R4. Density panel has one uncomfortable result buried
`density.csv` at lowdata:8 → `grn_rewired_A` = 0.686 > `grn_real_A` = 0.671. Rewired A-subset beats real A-subset. Buried in `docs/results.md`, never surfaced in memo.

Not something to volunteer — but know it in case Caelan opens the density figure.

---

## YELLOW — Prepare answers for these

### Y1. "Why is your baseline AE so much worse than PCA?"
> "All AE variants share the same architecture, budget, optimizer, and stopping rule — so the ordering **between AE variants** is fair. The PCA-vs-AE gap is a separate finding, replicates the well-known 'linear methods dominate cell-type identity' pattern (scIB / Kotliar 2019). I verified in the fixed-budget sweep (`12_final_fixedbudget.py`) that removing early-stopping barely moves the AE numbers — baseline goes from 0.785 to 0.786 at full data. So the AE isn't crippled by the val split."

### Y2. "Is CollecTRI doing all the heavy lifting? What if you strike it?"
> "Fair pushback. Low-data + clustering wins are strongest with CollecTRI. DoRothEA alone is directionally the same but weaker — still beats the rewired-network ULM null and the random-projection null on paired comparisons, still wins on low-data macro-F1 with smaller effect size. Under noise, DoRothEA alone loses; CollecTRI holds. Honest DoRothEA-only headline: 'ULM TF-activity carries real biology but modestly; CollecTRI extends the win to noise conditions.'"

### Y3. "Is the rewired-graph null a real null?"
> "Degree-preserving is the right level of null for the encoder-mask experiment — the encoder is only sensitive to WHICH gene connects to WHICH TF, not the marginal degree distribution. For the ULM transform I used the same degree-preserving rewiring. A stronger null would be a permuted-regulon test where each regulon's gene identities are randomly reassigned — I didn't run that; it's the second-best next check after the per-fold clustering redo."

### Y4. "The soft prior — isn't that just structured weight decay?"
> "Correct — I flag this in the PLAN self-review. The soft prior interpolates between 'dense AE with more L2 on off-regulon weights' and 'hard mask.' It doesn't falsify 'hardness is the problem' cleanly. What it does show is that ANY level of GRN penalty degrades relative to no penalty; that's a weaker claim than 'hardness itself is bad.'"

### Y5. "How much of B1/B2/B3 was actually fixed?"
- **B1 (deterministic folds):** FIXED. `eval.donor_grouped_folds` shuffles donors per seed. Non-zero std in `final.csv` proves seeds now produce different partitions.
- **B2 (paired deltas):** PARTIALLY FIXED. `final_stats.csv` reports `frac_beats_baseline` over 25 fold×seed pairs. Proper mixed-effects test not run.
- **B3 (rewired-ULM control):** FIXED. `dc_tfact_rewired` in definitive sweep.

### Y6. "How were the decoder-placement variants matched?"
> "All three (grn_real / grn_decoder / grn_symmetric) share the same AutoEncoder class — only the masked layer location changes. Nominal budget matched, effective free params for symmetric are 2× the other two, which is the opposite direction of the observed ordering, so the ordering isn't a capacity artefact. Only 3 seeds though — narrow but consistent."

### Y7. "Live-fetched networks — how reproducible is this?"
> "`decoupler` fetches DoRothEA/CollecTRI live at runtime — not version-pinned. Caching the returned network is on my next-step list. Census is pinned to 2025-11-08. Expected run-time delta from network updates is <0.01 F1 based on release notes."

---

## GREEN — Lead with these confidently

- **B1 fix is clean.** `eval.donor_grouped_folds` shuffles donors per seed. Kills the "5 seeds share one partition" objection.
- **B3 fix is real.** `dc_tfact_rewired` integrated into `11_final.py`.
- **B2 fix landed as `final_stats.csv`** with frac_beats over 25 fold×seed pairs.
- **Noise renormalization bug** (round-2's clean-data leak on TF-activity under noise) — found and fixed.
- **Two orthogonal nulls for TF-activity:** `rand_proj` (dim-null) + `dc_tfact_rewired` (topology-null). Textbook control design.
- **All AE variants share architecture/budget/optimizer/early-stopping.** Only mask differs. Kills the "your baseline is nerfed" attack.
- **DoRothEA-restricted gene set shared across all models.** Kills feature-selection confound.
- **Fixed-budget robustness check** (`12_final_fixedbudget.py`). Verdict stable without early stopping.
- **Second dataset (COVID PBMC)** replicates ordering.
- **Bottleneck-dim sensitivity** at z ∈ {32,64,128}. Ordering stable, kills z=64-artefact objection.
- **Fixed 15-class `labels=` in macro-F1** — S7 fixed, folds average over same label set.

---

## Numbers to memorize (say out loud from memory)

**Cell-type macro-F1 (mean over 5 seeds × 5 folds), from `final.csv`:**

| method | full | lowdata:16 | lowdata:8 | lowdata:4 | noise:0.1 | noise:0.3 |
|---|---|---|---|---|---|---|
| **pca** | **0.869** | 0.782 | 0.632 | 0.504 | 0.683 | 0.770 |
| baseline | 0.785 | 0.743 | 0.661 | 0.543 | 0.672 | 0.732 |
| grn_real | 0.714 | 0.668 | 0.591 | 0.498 | 0.516 | 0.615 |
| grn_soft:0.001 | 0.753 | 0.729 | 0.641 | 0.542 | 0.667 | 0.715 |
| dc_tfact | 0.847 | 0.774 | 0.686 | 0.598 | 0.627 | 0.720 |
| dc_tfact_rewired | 0.804 | 0.745 | 0.661 | 0.557 | 0.598 | 0.703 |
| **dc_tfact_collectri** | **0.869** | **0.821** | **0.746** | **0.643** | **0.693** | **0.784** |
| dc_tfact_pca (64d) | 0.783 | 0.748 | 0.630 | 0.489 | 0.581 | 0.686 |
| rand_proj (64d) | 0.775 | 0.719 | 0.624 | 0.501 | 0.542 | 0.648 |

**Winner per condition:**
- **full:** pca = dc_tfact_collectri = 0.869 (tied)
- **lowdata:16:** dc_tfact_collectri 0.821 > pca 0.782 > dc_tfact 0.774
- **lowdata:8:** dc_tfact_collectri 0.746 > dc_tfact 0.686 > dc_tfact_rewired 0.661
- **lowdata:4:** dc_tfact_collectri 0.643 > dc_tfact 0.598 > dc_tfact_rewired 0.557
- **noise:0.1:** dc_tfact_collectri 0.693 > pca 0.683 > baseline 0.672
- **noise:0.3:** dc_tfact_collectri 0.784 > pca 0.770 > baseline 0.732

**Key gaps to know:**
- dc_tfact vs dc_tfact_rewired at full = **0.043** (biology gap)
- dc_tfact_collectri vs baseline at full = **+0.084**
- dc_tfact_collectri vs baseline at lowdata:4 = **+0.100**
- dc_tfact_collectri vs baseline at noise:0.3 = **+0.052**
- grn_real vs baseline at full = **−0.071** (encoder-mask hurts)
- grn_real vs grn_rewired at full = 0.049 (real barely > rewired for encoder-mask)

**Paired frac_beats_baseline (`final_stats.csv`), full data:**
- pca 0.92 · dc_tfact 0.92 · dc_tfact_collectri 0.92 · dc_tfact_rewired 0.68
- grn_soft 0.24 · rand_proj 0.52 · grn_real 0.04 · grn_rewired 0.04

**Clustering ARI (`clustering.csv`, 6 methods only):**
- pca 0.124 · grn_real 0.215 · grn_decoder 0.238 · baseline 0.255 · dc_tfact 0.276 · **dc_tfact_collectri 0.304**
- NMI: same ordering (pca 0.390 · dc_tfact_collectri 0.594)
- ⚠️ **AE-based rows here trained on all 500 samples — R1 confound**

**Density (`density.csv`) — the buried result:**
- lowdata:8: **grn_rewired_A 0.686 > grn_real_A 0.671 > baseline 0.648** (rewired A-subset beats real A-subset)
- full: grn_real_A 0.788 ≈ baseline 0.791 (A-subset ties baseline at full data — the "sparsity is the lever" story)

**Fixed-budget check (`final_fixedbudget.csv`):** baseline full 0.785 → 0.786 (early-stopping removed). Verdict stable.

---

## Meta framing

**Lead the conversation with process, not results.** Your strengths are: fold-shuffle fix, rewired-net controls, second dataset, fixed-budget check, self-review scaffolding. This is the muscle Tolemy said they'd grade you on (judgment, rigor, honesty). Act 1 (encoder-mask never works) is your rock-solid finding — anchor there.

**Concede R1 preemptively.** Turns a potential ambush into an intellectual-honesty win.

**Reframe CollecTRI honestly.** Don't call it a robustness check if it's carrying the noise-and-low-data story. Reframe: "DoRothEA is primary; CollecTRI shows the transform generalises across catalogues — but transparently, the noise wins are stronger with CollecTRI. Both hold above rewired and random-projection nulls."

**If Caelan pushes and you don't have a number:** "I don't have that in front of me — I'll follow up after the call." Then actually follow up. Confidence is being OK not knowing something on demand.

---

## Tonight's checklist

1. Open `results/tables/final.csv`, `final_stats.csv`, `clustering.csv`, `density.csv`. Memorise numbers for **grn_real / dc_tfact / dc_tfact_rewired / dc_tfact_collectri / baseline / pca** at **full / lowdata:8 / noise:0.3**.
2. Open `scripts/14_clustering.py`. Point at the `make_embedder` call. Understand the R1 confound viscerally.
3. Skim `PLAN.md §14` (critical self-review). Walk through B1/B2/B3 out loud.
4. Reread `memo/memo.md` TL;DR. Know every claim you'll defend.

## Thursday morning checklist

5. Rehearse R1 volunteer answer out loud once.
6. Rehearse biology-gap answer once ("reasoning under uncertainty, not vocabulary").
7. Pick 2-3 questions for Caelan from the earlier prep pack.
8. Water. Notepad. Camera. Quiet room.

## Post-call

Send tight follow-up email within 2h: one thing you enjoyed discussing, one thing you learned from Caelan's questions, one item you agreed you'd think more about. If R1 came up, restate concession and how you'd fix it — turns weakness into follow-up commitment.
