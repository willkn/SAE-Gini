"""
Driver for full_positioning_replication.py across seeds, producing mean +/-
std for every method (Tuned L1, Gated, Gini, Top-k, Rate-KL x2 lambdas) on
every metric (sparsity, MSE, magnitude, purity, steering) -- the numbers
that go directly into the paper's positioning table (Table "positioning"),
replacing the single-seed Gated/L1 rows and the missing Rate-KL magnitude
column flagged in review.
"""
import argparse
import json
import os
import statistics

from full_positioning_replication import run
from steering_shrinkage_benchmark import DATASETS

METRICS = [
    "relative_sparsity",
    "mse",
    "mean_active_magnitude",
    "mean_purity",
    "mean_purity_top10pct_by_importance",
    "steering_impact_ablate",
    "steering_impact_clamp",
]


def aggregate(all_results):
    models = sorted({m for r in all_results.values() for m in r})
    summary = {}
    for model in models:
        runs = [r[model] for r in all_results.values() if model in r]
        summary[model] = {}
        for metric in METRICS:
            values = [r[metric] for r in runs if metric in r]
            if not values:
                continue
            summary[model][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01])
    args = parser.parse_args()

    all_results = {}
    for seed in args.seeds:
        out_path = f"results/full_positioning/{args.dataset}_seed{seed}.json"
        if os.path.exists(out_path):
            print(f"Skipping seed={seed} (already exists)")
            with open(out_path) as f:
                all_results[seed] = json.load(f)
            continue
        all_results[seed] = run(args.dataset, seed, args.rho, args.lambdas)

    summary = aggregate(all_results)
    with open(f"results/full_positioning/{args.dataset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.dataset}: full positioning summary (mean +/- std over {len(args.seeds)} seeds) ===")
    header = (
        f"{'Model':16s} {'Sparsity':>11s} {'MSE':>13s} {'Magnitude':>14s} "
        f"{'MeanPurity':>14s} {'Top10Purity':>14s} {'Ablate':>14s} {'Clamp':>14s}"
    )
    print(header)
    for model, m in summary.items():
        def fmt(key, prec=4):
            v = m.get(key)
            return f"{v['mean']:.{prec}f}+/-{v['std']:.{prec}f}" if v else "n/a"

        print(
            f"{model:16s} {fmt('relative_sparsity', 3):>11s} {fmt('mse'):>13s} {fmt('mean_active_magnitude', 3):>14s} "
            f"{fmt('mean_purity', 3):>14s} {fmt('mean_purity_top10pct_by_importance', 3):>14s} "
            f"{fmt('steering_impact_ablate'):>14s} {fmt('steering_impact_clamp'):>14s}"
        )
