"""
Fix attempt for the hub/polysemantic-feature pathology diagnosed in
feature_purity_analysis.py: per-sample Gini rewards "few large, many
near-zero" but is permutation-invariant over the latent index, so it has
no pressure against the SAME few features being the large ones for every
input. The cheapest low-loss, high-Gini solution SGD finds is a handful of
generically-useful "hub" directions that dominate reconstruction for every
class -- exactly what we measured (Gini's top-10%-by-importance features
had ~0 class purity, vs TopK's 0.033, and Gini's overall mean purity was
roughly half of TopK's).

Fix: add a second, batch-level term alongside the existing per-sample Gini
term, analogous to Mixture-of-Experts load-balancing losses (Shazeer et
al., 2017; Switch Transformer, Fedus et al., 2021), which solve the
structurally identical problem of "the same few experts get selected for
every input." Per-sample Gini still rewards within-sample sparsity; the new
term penalizes any single feature from carrying a disproportionate share of
total activation mass across the BATCH, pushing the model toward different
inputs routing through different feature subsets rather than a fixed hub.

  usage_share_f = mean_batch(|z_f|) / sum_f mean_batch(|z_f|)
  balance_loss  = num_features * sum_f usage_share_f^2

balance_loss is a Herfindahl concentration index, rescaled to equal 1.0 at
perfectly uniform usage and num_features at maximal concentration on one
feature -- minimizing it directly opposes the hub failure mode without
touching the per-sample sparsity mechanism at all.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import SimpleSAE, DATASETS, latent_dim, batch_size, lr, epochs, device, steering_metrics
from feature_purity_analysis import feature_importance_and_purity, summarize


def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini


def load_balance_penalty(z, epsilon=1e-8):
    """Herfindahl concentration of batch-level feature usage. 1.0 = perfectly
    uniform usage across features, num_features = fully concentrated on one."""
    n = z.shape[1]
    usage = z.abs().mean(dim=0)  # [n], mean |activation| per feature over the batch
    usage_share = usage / (usage.sum() + epsilon)
    return n * (usage_share ** 2).sum()


def train_gini_balanced(train_loader, input_dim, lambda_gini=0.1, lambda_balance=0.0):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = (
                criterion(recon, batch)
                - lambda_gini * differentiable_gini(z).mean()
                + lambda_balance * load_balance_penalty(z)
            )
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, lambda_balance_values):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for lb in lambda_balance_values:
        label = f"Gini_balance{lb}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_gini_balanced(train_loader, input_dim, lambda_balance=lb)
        stats = feature_importance_and_purity(model, test_loader)
        r = summarize(stats, label)
        mean_active = (stats["active_counts"].sum() / len(test_ds)).item()
        r["mean_active_features_per_sample"] = mean_active
        r["relative_sparsity"] = 1.0 - mean_active / latent_dim
        criterion = torch.nn.MSELoss()
        mse_total, n_batches = 0.0, 0
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                mse_total += criterion(recon, batch).item()
                n_batches += 1
        r["mse"] = mse_total / n_batches
        r.update(steering_metrics(model, test_loader))
        results[label] = r

    out_path = f"results/gini_load_balance/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: load-balance sweep ===")
    header = (
        f"{'Model':16s} {'MSE':>8s} {'Sparsity':>9s} {'ImpGini':>8s} {'Top10Share':>11s} "
        f"{'MeanPurity':>11s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:16s} {r['mse']:8.4f} {r['relative_sparsity']:9.3f} "
            f"{r['importance_gini_coefficient']:8.3f} {r['top10_importance_share']:11.3f} "
            f"{r['mean_purity']:11.3f} {r['mean_purity_top10pct_by_importance']:12.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    args = parser.parse_args()
    run(args.dataset, args.seed, args.lambdas)
