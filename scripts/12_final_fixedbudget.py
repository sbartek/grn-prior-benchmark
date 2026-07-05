"""Same as 11_final.py but FIXED-BUDGET training (early_stop=False, 300 epochs on all train
donors) instead of early stopping on a train-donor val slice. Tests whether removing the
val-selection / recovering the 15% held-out-for-val data changes the verdict.

Output: results/tables/final_fixedbudget.csv (compare against final.csv).
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
EPOCHS = 300                 # past the observed convergence plateau (0.775 at 250 == 600)
CORE = ["pca", "baseline", "grn_real", "grn_rewired", "grn_soft:0.001",
        "dc_tfact", "dc_tfact_rewired", "dc_tfact_collectri", "dc_tfact_pca", "rand_proj"]
FULL_ONLY = ["grn_random", "grn_sign_shuffled", "grn_soft:0.01"]
CONDITIONS = ["full", "lowdata:4", "lowdata:8", "lowdata:16", "noise:0.3", "noise:0.1"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, "dorothea")
    data["tfact_collectri"] = compute_tfact(data, "collectri")
    data["tfact_rewired"] = compute_tfact_mat(data["X"], data["genes"],
                                              net=corrupted_net("dorothea", "rewired", 0))
    print(f"[data] X={data['X'].shape} device={dev}  (FIXED-BUDGET, early_stop=False)")
    rows = []
    jobs = [(m, c) for c in CONDITIONS for m in CORE] + [(m, "full") for m in FULL_ONLY]
    for m, cond in jobs:
        scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=EPOCHS,
                        early_stop=False)
        rows.append(dict(model=m, condition=cond, mean=float(np.mean(scores)),
                         std=float(np.std(scores))))
        print(f"  {m:20s} {cond:11s} F1={np.mean(scores):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "final_fixedbudget.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
