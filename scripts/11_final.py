"""Definitive consolidated sweep (post-review, all fixes applied).

Fixes vs earlier runs: per-seed re-shuffled donor folds (real variance), fixed 15-class labels in
macro-F1, noise renormalised by full-transcriptome library, and the NEW transform-side control
`dc_tfact_rewired` (ULM on a degree-preserving rewired DoRothEA net). This table supersedes
round2.csv / dimcheck.csv / results.csv.

Output: results/tables/final.csv + final_stats.csv (paired vs baseline, reported over folds).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                                          # noqa: E402
from grn_bench.data import load_aligned                                    # noqa: E402
from grn_bench.experiments import compute_tfact, compute_tfact_mat, corrupted_net, run_cv  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
CORE = ["pca", "baseline", "grn_real", "grn_rewired", "grn_soft:0.001",
        "dc_tfact", "dc_tfact_rewired", "dc_tfact_collectri", "dc_tfact_pca", "rand_proj"]
FULL_ONLY = ["grn_random", "grn_sign_shuffled", "grn_soft:0.01"]     # corruption completeness
CONDITIONS = ["full", "lowdata:4", "lowdata:8", "lowdata:16", "noise:0.3", "noise:0.1"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, "dorothea")
    data["tfact_collectri"] = compute_tfact(data, "collectri")
    data["tfact_rewired"] = compute_tfact_mat(data["X"], data["genes"],
                                              net=corrupted_net("dorothea", "rewired", 0))
    print(f"[data] X={data['X'].shape} tfact={data['tfact'].shape} device={dev}")

    rows, flats = [], {}
    jobs = [(m, c) for c in CONDITIONS for m in CORE] + [(m, "full") for m in FULL_ONLY]
    for m, cond in jobs:
        flat = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250, return_flat=True)
        flats[(m, cond)] = flat
        vals = [r["f1"] for r in flat]
        rows.append(dict(model=m, condition=cond, mean=float(np.mean(vals)),
                         std=float(np.std([np.mean([r["f1"] for r in flat if r["seed"] == s])
                                           for s in SEEDS]))))
        print(f"  {m:20s} {cond:11s} F1={np.mean(vals):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "final.csv", index=False)

    # paired deltas vs baseline at matched (seed, fold); report over the 25 folds
    stats = []
    for cond in CONDITIONS:
        base = {(r["seed"], r["fold"]): r["f1"] for r in flats[("baseline", cond)]}
        for m in CORE:
            if m == "baseline":
                continue
            d = np.array([r["f1"] - base[(r["seed"], r["fold"])] for r in flats[(m, cond)]])
            stats.append(dict(model=m, condition=cond, mean_delta=float(d.mean()),
                              frac_beats_baseline=float((d > 0).mean()), n_folds=len(d)))
    pd.DataFrame(stats).to_csv(ROOT / "results" / "tables" / "final_stats.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
