"""Step 6 — run the full benchmark sweep and save a tidy results table.

Sweeps (each = donor-grouped 5-fold CV, 2 seeds, macro-F1):
  full        6 models x {cell_type, disease}          -> baseline vs GRN vs graph controls
  lowdata     {baseline, grn_real, grn_rewired} x k in {4,8,16}
  noise       {baseline, grn_real, grn_rewired} x thinning p in {0.3, 0.1}
Also: donor-predictability of each full-data embedding (leakage proxy).

Output: results/tables/results.csv  and  results/tables/donor_leakage.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grn_bench import models as M                       # noqa: E402
from grn_bench.data import load_aligned                 # noqa: E402
from grn_bench.eval import donor_grouped_folds, donor_predictability  # noqa: E402
from grn_bench.experiments import make_embedder, run_cv  # noqa: E402

SEEDS = (0, 1)
EPOCHS = 250
ALL_MODELS = ["pca", "baseline", "grn_real", "grn_rewired", "grn_sign_shuffled", "grn_random"]
CORE = ["baseline", "grn_real", "grn_rewired"]


def main():
    data = load_aligned()
    dev = M.pick_device()
    print(f"[data] {data['X'].shape}  device={dev}")
    rows = []

    def record(model, condition, task, scores):
        rows.append(dict(model=model, condition=condition, task=task,
                         mean=float(np.mean(scores)), std=float(np.std(scores)),
                         n=len(scores)))
        print(f"  {model:16s} {condition:10s} {task:10s} F1={np.mean(scores):.3f}±{np.std(scores):.3f}")

    print("[full]")
    for task in ["cell_type", "disease"]:
        for m in ALL_MODELS:
            record(m, "full", task, run_cv(data, m, task, "full", dev, SEEDS, epochs=EPOCHS))

    print("[lowdata]")
    for task in ["cell_type", "disease"]:
        for k in [4, 8, 16]:
            for m in CORE:
                record(m, f"lowdata:{k}", task, run_cv(data, m, task, f"lowdata:{k}", dev, SEEDS, epochs=EPOCHS))

    print("[noise]")
    for task in ["cell_type", "disease"]:
        for p in [0.3, 0.1]:
            for m in CORE:
                record(m, f"noise:{p}", task, run_cv(data, m, task, f"noise:{p}", dev, SEEDS, epochs=EPOCHS))

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out}  ({len(df)} rows)")

    # donor leakage on full-data embeddings (seed 0, all train donors)
    print("[donor-leakage]")
    donor = data["obs"]["donor_id"].astype(str).values
    all_idx = np.arange(data["X"].shape[0])
    lk = []
    for m in ALL_MODELS:
        emb = make_embedder(m, data, all_idx, dev, 0, EPOCHS, data["X"])
        acc = donor_predictability(emb, donor)
        lk.append(dict(model=m, donor_acc=acc))
        print(f"  {m:16s} donor_acc={acc:.3f}")
    pd.DataFrame(lk).to_csv(ROOT / "results" / "tables" / "donor_leakage.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
