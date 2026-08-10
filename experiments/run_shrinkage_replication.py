"""
Driver for the multi-seed, multi-dataset replication of the shrinkage/
steering benchmark (steering_shrinkage_benchmark.py). Runs each
(dataset, seed) combination, then aggregates into mean +/- std per model
per metric so we can tell whether the Fashion-MNIST result (Gini beats
matched-sparsity TopK on magnitude concentration and steering impact)
generalizes or was a single-run artifact.
"""
import argparse
import json
import os
import statistics

from steering_shrinkage_benchmark import DATASETS, run

METRICS = [
    "mse",
    "relative_sparsity",
    "mean_active_features_per_sample",
    "mean_active_magnitude",
    "steering_impact_ablate",
    "steering_impact_clamp",
]


def aggregate(all_results):
    # all_results: {(dataset, seed): {model_name: {metric: value}}}
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
            summary[dataset][model] = {}
            for metric in METRICS:
                values = [r[metric] for r in runs]
                summary[dataset][model][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    all_results = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            out_path = f"results/shrinkage_replication/{dataset}_seed{seed}.json"
            if os.path.exists(out_path):
                print(f"Skipping {dataset} seed={seed} (already exists)")
                with open(out_path) as f:
                    all_results[(dataset, seed)] = json.load(f)
                continue
            all_results[(dataset, seed)] = run(dataset, seed)

    summary = aggregate(all_results)
    with open("results/shrinkage_replication/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Replication Summary (mean +/- std) ===")
    for dataset, models in summary.items():
        print(f"\n-- {dataset} --")
        for model, metrics in models.items():
            mag = metrics["mean_active_magnitude"]
            ablate = metrics["steering_impact_ablate"]
            clamp = metrics["steering_impact_clamp"]
            sparsity = metrics["relative_sparsity"]
            print(
                f"{model:10s} sparsity={sparsity['mean']:.3f}  "
                f"magnitude={mag['mean']:.3f}+/-{mag['std']:.3f}  "
                f"ablate={ablate['mean']:.4f}+/-{ablate['std']:.4f}  "
                f"clamp={clamp['mean']:.4f}+/-{clamp['std']:.4f}"
            )
