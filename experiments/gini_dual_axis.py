"""
Dual-axis Gini: apply the same inequality index along BOTH axes of the
latent code matrix Z [batch x features].

Diagnosis recap (feature_purity_analysis.py, gini_load_balanced.py):
  - Row-wise Gini (per-sample) rewards "few large features per input" but is
    permutation-invariant over feature identity, so SGD satisfies it with a
    few polysemantic hub features reused for every input (top-10% importance
    features had ~0 class purity).
  - The MoE-style load-balance fix removed hubs by pushing feature usage
    toward UNIFORM across the batch -- but per-occurrence steering impact of
    feature f is |z_f| * ||W_f|| when it fires, and uniform usage forbids
    rare-but-strong features, so the fix capped exactly the quantity the
    steering metric measures (steering fell to ~45% of TopK's).

Fix: a hub feature is a COLUMN of Z that is uniformly large across samples,
i.e. a column with LOW Gini. A monosemantic feature fires hard on the few
inputs containing its concept and is zero elsewhere -- a column with HIGH
Gini. So instead of penalizing usage concentration, MAXIMIZE column-wise
Gini alongside row-wise Gini:

    loss = MSE - lambda_row * Gini(rows) - lambda_col * Gini(columns)

Column-Gini maximization pushes each feature toward low firing rate with
high magnitude when firing (concentrating fixed column mass on few samples
means large values there), which is exactly what maximizes per-occurrence
ablation/clamp impact -- while hubs are the global MINIMUM of column Gini,
so the same term structurally forbids the artifact. Same O(sort) cost,
still scale-invariant on both axes.

Known risks this sweep is designed to expose: row/column terms can fight
over which entries are zero, and column statistics are per-batch estimates
(noisy at small batch size).
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import (
    SimpleSAE,
    DATASETS,
    latent_dim,
    batch_size,
    lr,
    epochs,
    device,
    steering_metrics,
    train_topk,
    shrinkage_metrics,
)
from feature_purity_analysis import feature_importance_and_purity, summarize


def gini_along(x, dim, epsilon=1e-8):
    """Differentiable Gini along an arbitrary axis. dim=1 on [B, N] recovers
    the original per-sample formulation; dim=0 gives per-feature (column)
    Gini across the batch."""
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=dim)
    n = x_sorted.shape[dim]
    index_shape = [1, 1]
    index_shape[dim] = n
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device).view(index_shape)
    sum_weighted = torch.sum(index * x_sorted, dim=dim)
    sum_total = torch.sum(x_sorted, dim=dim) + epsilon
    return (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n


def train_gini_dual(train_loader, input_dim, lambda_row=0.1, lambda_col=0.0):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) - lambda_row * gini_along(z, dim=1).mean()
            if lambda_col > 0:
                loss = loss - lambda_col * gini_along(z, dim=0).mean()
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, test_loader, test_len, label):
    stats = feature_importance_and_purity(model, test_loader)
    r = summarize(stats, label)
    mean_active = (stats["active_counts"].sum() / test_len).item()
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
    return r


def run(dataset_name, seed, lambda_row, lambda_col_values):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    gini_active = None
    for lc in lambda_col_values:
        label = f"Gini_row{lambda_row}_col{lc}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_gini_dual(train_loader, input_dim, lambda_row=lambda_row, lambda_col=lc)
        results[label] = evaluate(model, test_loader, len(test_ds), label)
        if lc == 0.0 or gini_active is None:
            gini_active = results[label]["mean_active_features_per_sample"]

    # TopK reference at sparsity matched to the lambda_col=0 baseline, so the
    # comparison target is the same one used throughout the paper.
    matched_k = max(1, round(gini_active))
    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    results["TopK_reference"] = evaluate(topk_model, test_loader, len(test_ds), "TopK_reference")

    out_path = f"results/gini_dual_axis/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: dual-axis Gini sweep ===")
    header = (
        f"{'Model':22s} {'MSE':>8s} {'Sparsity':>9s} {'ImpGini':>8s} {'Top10Share':>11s} "
        f"{'MeanPurity':>11s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:22s} {r['mse']:8.4f} {r['relative_sparsity']:9.3f} "
            f"{r['importance_gini_coefficient']:8.3f} {r['top10_importance_share']:11.3f} "
            f"{r['mean_purity']:11.3f} {r['mean_purity_top10pct_by_importance']:12.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-row", type=float, default=0.1)
    parser.add_argument("--lambda-cols", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.3])
    args = parser.parse_args()
    run(args.dataset, args.seed, args.lambda_row, args.lambda_cols)
