"""Round 2 — soft prior, decoupler TF-activity, and 5-seed paired statistics.

Adds to the story:
  A1 soft prior      grn_soft:<lambda>  dense encoder + penalty shrinking off-regulon weights
                     (interpolates baseline <-> hard mask WITHOUT the capacity cost)
  A2 TF-activity     dc_tfact           canonical non-learned DoRothEA embedding (decoupler ULM)
  A3 statistics      5 seeds; paired baseline-vs-model deltas across (seed, fold)

Output: results/tables/round2.csv (per-condition means) + round2_stats.csv (paired deltas).
"""
import sys
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                 # noqa: E402
from grn_bench.data import load_aligned           # noqa: E402
from grn_bench.experiments import run_cv          # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
MODELS = ["pca", "baseline", "dc_tfact", "dc_tfact_pca", "grn_soft:0.001", "grn_soft:0.01", "grn_real"]
CONDITIONS = ["full", "lowdata:4", "lowdata:8", "lowdata:16", "noise:0.3", "noise:0.1"]
TASK = "cell_type"


def compute_tfact(data):
    A = ad.AnnData(data["X"].copy())
    A.var_names = list(data["genes"])
    dc.mt.ulm(A, net=dc.op.dorothea(organism="human"), tmin=5, verbose=False)
    return np.asarray(A.obsm["score_ulm"])


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data)
    print(f"[data] X={data['X'].shape} tfact={data['tfact'].shape} device={dev}")

    means, flats = [], {}
    for cond in CONDITIONS:
        for m in MODELS:
            flat = run_cv(data, m, TASK, cond, dev, SEEDS, epochs=250, return_flat=True)
            flats[(m, cond)] = flat
            mu = np.mean([r["f1"] for r in flat])
            sd = np.std([np.mean([r["f1"] for r in flat if r["seed"] == s]) for s in SEEDS])
            means.append(dict(model=m, condition=cond, mean=float(mu), std=float(sd)))
            print(f"  {m:16s} {cond:11s} F1={mu:.3f}±{sd:.3f}")

    pd.DataFrame(means).to_csv(ROOT / "results" / "tables" / "round2.csv", index=False)

    # paired deltas vs baseline at matched (seed, fold)
    stats = []
    for cond in CONDITIONS:
        base = {(r["seed"], r["fold"]): r["f1"] for r in flats[("baseline", cond)]}
        for m in MODELS:
            if m == "baseline":
                continue
            deltas = [r["f1"] - base[(r["seed"], r["fold"])] for r in flats[(m, cond)]
                      if (r["seed"], r["fold"]) in base]
            deltas = np.array(deltas)
            stats.append(dict(model=m, condition=cond, mean_delta=float(deltas.mean()),
                              frac_positive=float((deltas > 0).mean()), n=len(deltas)))
    sdf = pd.DataFrame(stats)
    sdf.to_csv(ROOT / "results" / "tables" / "round2_stats.csv", index=False)
    print("\n[paired deltas vs baseline] (positive = beats baseline)")
    print(sdf.to_string(index=False))
    print("[done]")


if __name__ == "__main__":
    main()
