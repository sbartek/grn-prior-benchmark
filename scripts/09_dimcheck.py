"""Dimensionality & prior-source controls (from the literature scan).

Isolates how much of the decoupler TF-activity advantage is *biology* vs *dimensionality*, and
whether it is DoRothEA-specific:

  64-d refs   : baseline (AE), pca, dc_tfact_pca (TF-activity -> PCA-64)
  293-d arms  : rand_proj (matched-dim RANDOM linear features = the null),
                dc_tfact (DoRothEA ULM), dc_tfact_collectri (CollecTRI ULM)

If rand_proj(293) ~ dc_tfact(293), the gain is dimensionality; if dc_tfact > rand_proj, biology
adds beyond dimension. dc_tfact vs dc_tfact_collectri = prior-source sensitivity.

Output: results/tables/dimcheck.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                               # noqa: E402
from grn_bench.data import load_aligned                         # noqa: E402
from grn_bench.experiments import compute_tfact, run_cv         # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
MODELS = ["baseline", "pca", "dc_tfact_pca", "rand_proj", "dc_tfact", "dc_tfact_collectri"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    data["tfact"] = compute_tfact(data, net="dorothea")
    data["tfact_collectri"] = compute_tfact(data, net="collectri")
    print(f"[data] X={data['X'].shape} tfact={data['tfact'].shape} "
          f"collectri={data['tfact_collectri'].shape} device={dev}")
    rows = []
    for cond in CONDITIONS:
        for m in MODELS:
            scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250)
            rows.append(dict(model=m, condition=cond,
                             mean=float(np.mean(scores)), std=float(np.std(scores))))
            print(f"  {m:20s} {cond:10s} F1={np.mean(scores):.3f}±{np.std(scores):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "dimcheck.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
