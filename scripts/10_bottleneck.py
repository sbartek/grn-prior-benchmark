"""Bottleneck-dimension sensitivity — is the verdict a tuning artifact?

Sweeps the embedding size z in {32, 64, 128} for PCA, the baseline autoencoder, and the two
GRN-as-constraint encoders. If the ordering (PCA >= baseline > GRN-mask) is stable across z, the
negative-for-constraints result is not an artifact of the z=64 default.

Output: results/tables/bottleneck.csv
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

SEEDS = (0, 1, 2)
Z_DIMS = [32, 64, 128]
MODELS = ["pca", "baseline", "grn_real", "grn_soft:0.001"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    print(f"[data] {data['X'].shape} device={dev}")
    rows = []
    for z in Z_DIMS:
        for cond in CONDITIONS:
            for m in MODELS:
                scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250, z_dim=z)
                rows.append(dict(z=z, model=m, condition=cond,
                                 mean=float(np.mean(scores)), std=float(np.std(scores))))
                print(f"  z={z:3d} {m:16s} {cond:10s} F1={np.mean(scores):.3f}±{np.std(scores):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "bottleneck.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
