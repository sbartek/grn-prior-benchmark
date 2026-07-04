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


def grouped_bar(csv, order, conds, title, fname, ylim=(0, 1.0), task=None):
    p = TAB / csv
    if not p.exists():
        return
    dd = pd.read_csv(p)
    if task is not None and "task" in dd.columns:
        dd = dd[dd.task == task]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    w = 0.8 / len(order)
    for j, m in enumerate(order):
        s = dd[dd.model == m].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)) + j * w, s["mean"], w, yerr=s.get("std"),
               label=m, capsize=2)
    ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2); ax.set_xticklabels(conds)
    ax.set_ylabel("cell-type macro-F1"); ax.set_ylim(*ylim); ax.set_title(title)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)


def bottleneck():
    p = TAB / "bottleneck.csv"
    if not p.exists():
        return
    dd = pd.read_csv(p)
    conds = ["full", "lowdata:8", "noise:0.3"]
    models = ["pca", "baseline", "grn_soft:0.001", "grn_real"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, cond in zip(axes, conds):
        for m in models:
            s = dd[(dd.model == m) & (dd.condition == cond)].sort_values("z")
            ax.errorbar(s["z"], s["mean"], yerr=s["std"], marker="o", capsize=3, label=m)
        ax.set_title(cond); ax.set_xlabel("bottleneck dim z"); ax.set_xticks([32, 64, 128])
    axes[0].set_ylabel("cell-type macro-F1"); axes[0].legend(fontsize=8)
    fig.suptitle("Bottleneck-dim sensitivity: ordering stable, hard mask always worst")
    fig.tight_layout(); fig.savefig(FIG / "fig_bottleneck.png", dpi=130); plt.close(fig)


def round2_and_covid():
    grouped_bar("round2.csv",
                ["pca", "baseline", "dc_tfact", "dc_tfact_pca", "grn_soft:0.001", "grn_real"],
                ["full", "lowdata:4", "lowdata:8", "lowdata:16", "noise:0.3", "noise:0.1"],
                "Round 2 (primary): TF-activity vs learned encoders", "fig_round2.png")
    covid_models = ["pca", "baseline", "dc_tfact", "dc_tfact_pca", "grn_soft:0.001", "grn_real", "grn_rewired"]
    grouped_bar("covid.csv", covid_models, ["full", "lowdata:8", "noise:0.3"],
                "COVID PBMC — cell type: same verdict", "fig_covid.png", task="cell_type")
    grouped_bar("covid.csv", covid_models, ["full", "lowdata:8", "noise:0.3"],
                "COVID PBMC — disease (3-class): PCA best; GRN≈rewired", "fig_covid_disease.png",
                task="disease", ylim=(0, 0.8))
    grouped_bar("dimcheck.csv",
                ["baseline", "pca", "dc_tfact_pca", "rand_proj", "dc_tfact", "dc_tfact_collectri"],
                ["full", "lowdata:8", "noise:0.3"],
                "Dimensionality vs biology: 64-d refs vs 293/675-d arms", "fig_dimcheck.png")


if __name__ == "__main__":
    bar_full()
    line_condition("lowdata:", "# train donors", "fig_lowdata.png", int)
    line_condition("noise:", "counts kept (thinning p)", "fig_noise.png", float)
    bar_leakage()
    bar_density()
    round2_and_covid()
    bottleneck()
    print("[figures] wrote fig_full, fig_lowdata, fig_noise, fig_leakage, fig_density, fig_round2, fig_covid")
