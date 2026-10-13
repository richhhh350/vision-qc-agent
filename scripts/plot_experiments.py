"""Generate standard scientific figures from experiment CSVs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.directory / "metadata.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    routes = pd.read_csv(args.directory / "routing.csv")
    for policy, rows in routes.groupby("policy"):
        axes[0].plot(rows["review_rate"] * 100, rows["auto_release_fnr"] * 100, "o-", label=policy)
    axes[0].set(xlabel="Human review (%)", ylabel="Auto-release miss rate (%)", title="Routing trade-off")
    axes[0].set_ylim(0, max(1.0, float(routes["auto_release_fnr"].max() * 110)))
    axes[0].legend()
    robust = pd.read_csv(args.directory / "robustness.csv")
    axes[1].barh(robust["transform"], robust["review_rate"] * 100, color="#377f91")
    axes[1].set(xlabel="Human review (%)", title="Frozen model under perturbations")
    learning_path = args.directory / "learning.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        for strategy, rows in learning.groupby("strategy"):
            axes[2].plot(rows["queried_labels"], rows["review_rate"] * 100, "o-", label=strategy)
        axes[2].legend()
    axes[2].set(xlabel="New queried labels", ylabel="Human review (%)", title="Exploratory active learning")
    source = "SYNTHETIC pipeline demo" if meta["synthetic"] else "Real-data experiment"
    fig.suptitle(f"Visual QC | {source} | seed {meta['seed']}", fontsize=13)
    fig.savefig(args.directory / "curves.png", dpi=180)
    fig.savefig(args.directory / "curves.svg")
    print(args.directory / "curves.png")


if __name__ == "__main__":
    main()
