"""
Generality test for the paper's central claim: the hub-feature pathology
is a property of the CLASS of permutation-invariant, vanishing-gradient-
at-degeneracy sparsity objectives, not specific to the Gini coefficient.
This tests a second instance -- the Theil index -- through the same
diagnostic pipeline used on Gini (feature_purity_analysis.py), to see if
the same hub signature (low top-decile purity, high importance
concentration) recurs.

The Theil index (a generalized-entropy inequality measure, T = 0 at
perfect equality) for a nonnegative vector x is:

    T(x) = (1/n) * sum_i (x_i / mean(x)) * log(x_i / mean(x))

Like Gini, it is (a) scale-invariant, (b) a function of the values only
through their relative shares -- permutation-invariant over which index
produced which value, and (c) minimized (T=0) at the perfectly uniform
distribution, i.e. exactly the hub configuration, with (we predict)
similarly weak gradient there: at x_i = mean(x) for all i, each term's
local sensitivity is governed by log(1) = 0 plus its derivative, giving a
flat region around the degenerate point analogous to Gini's tie-gradient
weakness.

We train a Theil-regularized SAE (same architecture, same training
protocol as differentiable_gini in steering_shrinkage_benchmark.py, loss
= MSE - lambda * mean(-Theil) i.e. maximize Theil to encourage inequality/
sparsity) and run it through the same importance/purity diagnostic used
on Gini. If the same hub signature appears, the generality claim is
demonstrated twice rather than asserted once.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import SimpleSAE, DATASETS, latent_dim, batch_size, lr, epochs, device, train_topk
from feature_purity_analysis import feature_importance_and_purity, summarize
from full_positioning_replication import evaluate_full


def differentiable_theil(x, epsilon=1e-8):
    """Theil's T index (generalized entropy, alpha=1), computed per sample
    over the latent dimension. T=0 at perfect uniformity (the degenerate
    hub configuration); T grows with inequality, with no upper bound (T is
    NOT bounded in [0,1] like Gini, so we clip the gradient-scale via the
    epsilon floor on the mean rather than rescaling the index itself)."""
    x_abs = torch.abs(x) + epsilon
    n = x_abs.shape[1]
    mean_x = x_abs.mean(dim=1, keepdim=True)
    ratio = x_abs / mean_x
    theil = (ratio * torch.log(ratio)).mean(dim=1)
    return theil


def train_theil(train_loader, input_dim, lambda_theil=0.1):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) - lambda_theil * differentiable_theil(z).mean()
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, lambda_values):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    matched_k = None
    for lam in lambda_values:
        label = f"Theil_lam{lam}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_theil(train_loader, input_dim, lambda_theil=lam)
        r = evaluate_full(model, test_loader, label)
        results[label] = r
        if matched_k is None:
            stats = feature_importance_and_purity(model, test_loader)
            mean_active = (stats["active_counts"].sum() / len(test_ds)).item()
            matched_k = max(1, round(mean_active))

    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    results["TopK_reference"] = evaluate_full(train_topk(train_loader, input_dim, matched_k), test_loader, "TopK_reference")

    out_path = f"results/theil_diagnosis/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: Theil index generality test ===")
    header = f"{'Model':20s} {'Sparsity':>9s} {'MSE':>8s} {'ImpGini':>8s} {'Top10Share':>11s} {'MeanPurity':>11s} {'Top10Purity':>12s} {'Ablate':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:20s} {r['relative_sparsity']:9.3f} {r['mse']:8.4f} "
            f"{r['importance_gini_coefficient']:8.3f} {r['top10_importance_share']:11.3f} "
            f"{r['mean_purity']:11.3f} {r['mean_purity_top10pct_by_importance']:12.3f} {r['steering_impact_ablate']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    args = parser.parse_args()
    run(args.dataset, args.seed, args.lambdas)
