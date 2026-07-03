"""Step 8 helper — turn results/tables/results.csv into the key figures.

Figures:
  fig_full.png       full-data macro-F1 per model (cell_type + disease) with error bars
  fig_lowdata.png    macro-F1 vs #train donors (baseline vs grn_real vs grn_rewired)
  fig_noise.png      macro-F1 vs thinning fraction (baseline vs grn_real vs grn_rewired)
  fig_leakage.png    donor-predictability per model (lower = less donor leakage)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(TAB / "results.csv")


def bar_full():
    sub = df[df.condition == "full"]
    tasks = ["cell_type", "disease"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, tasks):
        s = sub[sub.task == task].sort_values("mean")
        ax.barh(s.model, s["mean"], xerr=s["std"], color="#3aa")
        ax.set_title(f"full data — {task}"); ax.set_xlabel("macro-F1"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(FIG / "fig_full.png", dpi=130); plt.close(fig)


def line_condition(prefix, xlabel, fname, xcast):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, ["cell_type", "disease"]):
        s = df[df.condition.str.startswith(prefix) & (df.task == task)].copy()
        s["x"] = s.condition.str.split(":").str[1].astype(xcast)
        for m in ["baseline", "grn_real", "grn_rewired"]:
            g = s[s.model == m].sort_values("x")
            if len(g):
                ax.errorbar(g["x"], g["mean"], yerr=g["std"], marker="o", label=m, capsize=3)
        ax.set_title(f"{prefix.rstrip(':')} — {task}"); ax.set_xlabel(xlabel)
        ax.set_ylabel("macro-F1"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)


def bar_leakage():
    p = TAB / "donor_leakage.csv"
    if not p.exists():
        return
    s = pd.read_csv(p).sort_values("donor_acc")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.model, s.donor_acc, color="#a55")
    ax.set_xlabel("donor prediction accuracy (lower = less leakage)")
    ax.set_title("donor-identity leakage in embedding")
    fig.tight_layout(); fig.savefig(FIG / "fig_leakage.png", dpi=130); plt.close(fig)


def bar_density():
    p = TAB / "density.csv"
    if not p.exists():
        return
    dd = pd.read_csv(p)
    conds = ["full", "lowdata:8", "noise:0.3"]
    order = ["baseline", "grn_real_A", "grn_rewired_A", "grn_real_AB", "grn_real"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.16
    for j, m in enumerate(order):
        s = dd[dd.model == m].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)) + j * w, s["mean"], w, yerr=s["std"], label=m, capsize=2)
    ax.set_xticks(np.arange(len(conds)) + 2 * w); ax.set_xticklabels(conds)
    ax.set_ylabel("cell-type macro-F1"); ax.set_ylim(0, 0.9)
    ax.set_title("density ablation: real vs rewired at each confidence level")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig_density.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    bar_full()
    line_condition("lowdata:", "# train donors", "fig_lowdata.png", int)
    line_condition("noise:", "counts kept (thinning p)", "fig_noise.png", float)
    bar_leakage()
    bar_density()
    print("[figures] wrote fig_full, fig_lowdata, fig_noise, fig_leakage, fig_density")
