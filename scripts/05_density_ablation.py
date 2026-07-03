"""Optional ablation — does filtering DoRothEA by confidence (density) change the verdict?

Compares the GRN encoder built on confidence A (5,664 edges), A+B (14,312), and A+B+C (30,609)
against the baseline, across full / low-data / noise. If higher-confidence-but-sparser graphs
still don't beat the baseline, the negative result is robust to graph density.

Output: results/tables/density.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                 # noqa: E402
from grn_bench.data import load_aligned           # noqa: E402
from grn_bench.experiments import run_cv          # noqa: E402

SEEDS = (0, 1)
MODELS = ["baseline", "grn_real_A", "grn_rewired_A", "grn_real_AB", "grn_real"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    print(f"[data] {data['X'].shape} device={dev}")
    rows = []
    for cond in CONDITIONS:
        for m in MODELS:
            scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250)
            rows.append(dict(model=m, condition=cond, task="cell_type",
                             mean=float(np.mean(scores)), std=float(np.std(scores))))
            print(f"  {m:14s} {cond:10s} F1={np.mean(scores):.3f}±{np.std(scores):.3f}")
    out = ROOT / "results" / "tables" / "density.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
