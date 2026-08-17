"""
Closes two gaps flagged in review of the paper's Section "Positioning
Rate-KL Against Leading Methods":

  (1) Statistical power was uneven -- Top-k and Rate-KL had 3-seed means,
      Gated SAE and Tuned L1 were single-seed. This script puts all five
      methods (Tuned L1, Gated, Top-k, Gini, Rate-KL x2 lambdas) through
      the identical 3-seed protocol.
  (2) Rate-KL's per-feature magnitude was never directly logged, only
      inferred indirectly from steering impact. This script computes
      mean_active_magnitude (via steering_shrinkage_benchmark.shrinkage_metrics,
      which is model-agnostic given a (recon, z) forward interface) for
      every method, closing the asymmetry with the Gini/Top-k/Gated/L1
      table that already had it.

One script, one consistent pipeline: shrinkage_metrics for sparsity/MSE/
magnitude, feature_importance_and_purity for purity/importance-concentration,
steering_metrics for ablate/clamp -- so every number in the paper's
positioning table comes from the same evaluation code applied identically
to every method.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import (
    DATASETS,
    latent_dim,
    batch_size,
    train_gini,
    train_l1,
    train_gated,
    train_topk,
    shrinkage_metrics,
    steering_metrics,
    differentiable_gini,
)
from feature_purity_analysis import feature_importance_and_purity, summarize
from rate_kl_sae import train_rate_kl, _EvalWrapper


def evaluate_full(model, test_loader, label):
    r = shrinkage_metrics(model, test_loader)
    stats = feature_importance_and_purity(model, test_loader)
    r.update(summarize(stats, label))
    r.update(steering_metrics(model, test_loader))
    return r


def run(dataset_name, seed, rho, rate_kl_lambdas):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    print(f"[{dataset_name} seed={seed}] Training Tuned L1...")
    results["TunedL1"] = evaluate_full(train_l1(train_loader, input_dim), test_loader, "TunedL1")

    print(f"[{dataset_name} seed={seed}] Training Gated SAE...")
    results["GatedSAE"] = evaluate_full(train_gated(train_loader, input_dim), test_loader, "GatedSAE")

    print(f"[{dataset_name} seed={seed}] Training Gini SAE...")
    gini_model = train_gini(train_loader, input_dim)
    results["Gini"] = evaluate_full(gini_model, test_loader, "Gini")

    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training Top-K (k={matched_k})...")
    results["TopK"] = evaluate_full(train_topk(train_loader, input_dim, matched_k), test_loader, "TopK")

    for lam in rate_kl_lambdas:
        label = f"RateKL_lam{lam}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = _EvalWrapper(train_rate_kl(train_loader, input_dim, rho=rho, lambda_kl=lam))
        results[label] = evaluate_full(model, test_loader, label)

    out_path = f"results/full_positioning/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: full positioning ===")
    header = (
        f"{'Model':16s} {'Sparsity':>9s} {'MSE':>8s} {'Magnitude':>10s} "
        f"{'MeanPurity':>11s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:16s} {r['relative_sparsity']:9.3f} {r['mse']:8.4f} {r['mean_active_magnitude']:10.3f} "
            f"{r['mean_purity']:11.3f} {r['mean_purity_top10pct_by_importance']:12.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01])
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambdas)
