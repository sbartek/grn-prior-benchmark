"""Where to place the graph prior: encoder vs DECODER (expiMap-style) vs symmetric.

The decoder direction (TF-hidden -> gene) is the causal/generative one, and is where biologically-
informed autoencoders (e.g. expiMap) put the mask. Compares encoder-masked (grn_real), decoder-
masked (grn_decoder), and both (grn_symmetric), each vs its rewired control and the baseline.

Output: results/tables/decoder.csv
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
MODELS = ["pca", "baseline",
          "grn_real", "grn_rewired",
          "grn_decoder", "grn_decoder_rewired",
          "grn_symmetric", "grn_symmetric_rewired"]
CONDITIONS = ["full", "lowdata:8", "noise:0.3"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    print(f"[data] {data['X'].shape} device={dev}")
    rows = []
    for cond in CONDITIONS:
        for m in MODELS:
            scores = run_cv(data, m, "cell_type", cond, dev, SEEDS, epochs=250)
            rows.append(dict(model=m, condition=cond,
                             mean=float(np.mean(scores)), std=float(np.std(scores))))
            print(f"  {m:24s} {cond:10s} F1={np.mean(scores):.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "decoder.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
