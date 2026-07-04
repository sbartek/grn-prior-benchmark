"""B4 — external validity: rerun the core comparison on a second dataset (COVID PBMC).

Same pipeline (pseudobulk, DoRothEA graph, encoders, donor-grouped CV) on
CELLxGENE 2a498ace... (422k cells, 75 donors, 28 cell types, 3 disease states). If the verdict
holds here too, it is not an artifact of the primary RA dataset.

Prereqs (already built):
  data/covid_pseudobulk.h5ad   (scripts/01 on data/covid_raw.h5ad)
  data/covid_graph.npz         (scripts/02 on the above)

Output: results/tables/covid.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                          # noqa: E402
from grn_bench.data import load_aligned                    # noqa: E402
from grn_bench.experiments import compute_tfact, run_cv    # noqa: E402

SEEDS = (0, 1, 2)
MODELS = ["pca", "baseline", "dc_tfact", "grn_soft:0.001", "grn_real", "grn_rewired"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]


def main():
    data = load_aligned(pseudobulk="data/covid_pseudobulk.h5ad", graph="data/covid_graph.npz")
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data)
    print(f"[covid] X={data['X'].shape} tfact={data['tfact'].shape} "
          f"donors={data['obs'].donor_id.nunique()} cell_types={data['obs'].cell_type.nunique()}")
    rows = []
    for cond in CONDITIONS:
        for m in MODELS:
            scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250)
            rows.append(dict(model=m, condition=cond, task="cell_type",
                             mean=float(np.mean(scores)), std=float(np.std(scores))))
            print(f"  {m:16s} {cond:10s} F1={np.mean(scores):.3f}±{np.std(scores):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "covid.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
