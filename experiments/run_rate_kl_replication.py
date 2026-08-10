"""
Multi-seed, multi-dataset replication of the rate-KL result (rate_kl_sae.py).

Single-seed Fashion-MNIST found rate-KL at lambda=0.01 beating the matched-
sparsity TopK reference on ALL of: steering impact (2.4x), mean purity
(0.70 vs 0.28), and top-10%-by-importance purity (0.49 vs 0.035, i.e. the
steering advantage is not a hub artifact this time), at a 2.7x MSE cost;
lambda=0.001 matched TopK's steering with ~2x its purity at only 1.6x MSE.
This driver checks whether that pattern survives seeds and datasets, with
mean +/- std aggregation -- the same protocol that validated (and then
correctly complicated) the original Gini result.
"""
import argparse
import json
import os
import statistics

from rate_kl_sae import run
from steering_shrinkage_benchmark import DATASETS

METRICS = [
    "mse",
    "relative_sparsity",
    "mean_active_features_per_sample",
    "mean_purity",
    "mean_purity_top10pct_by_importance",
    "importance_gini_coefficient",
    "top10_importance_share",
    "steering_impact_ablate",
    "steering_impact_clamp",
    "n_interventions",
]


def aggregate(all_results):
    models = sorted({m for r in all_results.values() for m in r})
    datasets_seen = sorted({d for d, s in all_results})

    summary = {}
    for dataset in datasets_seen:
        summary[dataset] = {}
        for model in models:
            runs = [
                all_results[(d, s)][model]
                for (d, s) in all_results
                if d == dataset and model in all_results[(d, s)]
            ]
            if not runs:
                continue
            summary[dataset][model] = {}
            for metric in METRICS:
                values = [r[metric] for r in runs if metric in r]
                if not values:
                    continue
                summary[dataset][model][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            # near-silent-sample fraction from the bimodality diagnostic
            silent = [r["activity_distribution"]["frac_samples_lt5_active"] for r in runs if "activity_distribution" in r]
            if silent:
                summary[dataset][model]["frac_samples_lt5_active"] = {
                    "mean": statistics.mean(silent),
                    "std": statistics.stdev(silent) if len(silent) > 1 else 0.0,
                    "n": len(silent),
                }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01])
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()

    all_results = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            out_path = f"results/rate_kl/{dataset}_seed{seed}.json"
            if os.path.exists(out_path):
                print(f"Skipping {dataset} seed={seed} (already exists)")
                with open(out_path) as f:
                    all_results[(dataset, seed)] = json.load(f)
                continue
            all_results[(dataset, seed)] = run(dataset, seed, args.rho, args.lambdas, args.tau)

    summary = aggregate(all_results)
    with open("results/rate_kl/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Rate-KL Replication Summary (mean +/- std) ===")
    for dataset, models in summary.items():
        print(f"\n-- {dataset} --")
        for model, m in models.items():
            def fmt(key, prec=4):
                v = m.get(key)
                return f"{v['mean']:.{prec}f}+/-{v['std']:.{prec}f}" if v else "n/a"

            print(
                f"{model:24s} mse={fmt('mse')}  sparsity={fmt('relative_sparsity', 3)}  "
                f"purity={fmt('mean_purity', 3)}  top10purity={fmt('mean_purity_top10pct_by_importance', 3)}  "
                f"ablate={fmt('steering_impact_ablate')}  clamp={fmt('steering_impact_clamp')}  "
                f"lt5active={fmt('frac_samples_lt5_active', 3)}"
            )
