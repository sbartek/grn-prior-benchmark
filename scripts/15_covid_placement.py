"""Fill the COVID gaps: run the placement variants (grn_decoder, grn_symmetric) on COVID too,
so the consolidated table has no NaNs for them. Appends to results/tables/covid.csv.
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

MODELS = ["grn_decoder", "grn_symmetric"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]
SEEDS = (0, 1, 2)
OUT = ROOT / "results" / "tables" / "covid.csv"


def main():
    data = load_aligned(pseudobulk="data/covid_pseudobulk.h5ad", graph="data/covid_graph.npz")
    dev = M.pick_device()
    print(f"[covid] {data['X'].shape} device={dev}")
    rows = []
    for task in ["cell_type", "disease"]:
        for cond in CONDITIONS:
            for m in MODELS:
                s = run_cv(data, m, task, cond, dev, SEEDS, epochs=250)
                rows.append(dict(model=m, condition=cond, task=task,
                                 mean=float(np.mean(s)), std=float(np.std(s))))
                print(f"  {task:9s} {m:14s} {cond:10s} F1={np.mean(s):.3f}")
    existing = pd.read_csv(OUT)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(["model", "condition", "task"], keep="last")
    combined.to_csv(OUT, index=False)
    print(f"[write] {OUT} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
