"""
Tests whether the paper's design principle (identity-tracked, boundary-
escalating firing-rate control) is best framed as a standalone objective
(Rate-KL, the log-barrier variant) or as a PLUGGABLE AUXILIARY REGULARIZER
that can be added on top of any existing sparsity mechanism.

We add the log-barrier + budget penalty (rate_barrier_sae.py) on top of
two existing methods, unchanged otherwise:

  - Top-k + barrier: Top-k's hard per-sample selection is left exactly as
    is; the barrier is computed on the same pre-ReLU activations used to
    rank features for selection, as an auxiliary loss term only. Real
    hypothesis: Top-k has no persistent per-feature identity constraint at
    all (selection is independent each step), and we already measured it
    isn't hub-immune (importance-Gini 0.58-0.72 across datasets) -- adding
    the barrier should plausibly raise its purity without touching its
    sparsity mechanism.

  - Tuned L1 + barrier: L1's penalty is invariant to how mass is
    distributed across features (it only penalizes the sum), so it never
    had positive pressure toward concentration to begin with -- weaker
    hypothesis, included as a control. If the barrier does little here,
    that itself is informative: it would confirm the hub pathology is a
    property of objectives that reward concentration, not of sparsity
    objectives generally.

Evaluated through the same full_positioning_replication.evaluate_full
pipeline as everything else, so results are directly comparable to the
existing positioning table.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import SimpleSAE, DATASETS, latent_dim, batch_size, lr, epochs, device, train_topk, train_l1
from rate_barrier_sae import barrier_budget_penalty
from full_positioning_replication import evaluate_full


def train_topk_plus_barrier(train_loader, input_dim, k, lambda_barrier, lambda_budget, rho, tau=0.1):
    """Same hard top-k selection as train_topk; barrier is purely additive."""
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pre_act = model.encoder(batch)
            relu_act = torch.relu(pre_act)
            topk_z = torch.zeros_like(relu_act)
            if k < relu_act.shape[1]:
                values, indices = torch.topk(relu_act, k, dim=1)
                topk_z.scatter_(1, indices, values)
            else:
                topk_z = relu_act
            recon = model.decoder(topk_z)
            barrier, budget = barrier_budget_penalty(pre_act, rho, tau)
            loss = criterion(recon, batch) + lambda_barrier * barrier + lambda_budget * budget
            loss.backward()
            optimizer.step()
    return model


def train_l1_plus_barrier(train_loader, input_dim, l1_coef, lambda_barrier, lambda_budget, rho, tau=0.1):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pre_act = model.encoder(batch)
            z = torch.relu(pre_act)
            recon = model.decoder(z)
            barrier, budget = barrier_budget_penalty(pre_act, rho, tau)
            loss = (
                criterion(recon, batch)
                + l1_coef * z.abs().mean()
                + lambda_barrier * barrier
                + lambda_budget * budget
            )
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, rho, lambda_barrier, lambda_budget, l1_coef=0.01):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    matched_k = max(1, round(rho * latent_dim))
    results = {}

    print(f"[{dataset_name} seed={seed}] Training TopK (vanilla, k={matched_k})...")
    results["TopK_vanilla"] = evaluate_full(train_topk(train_loader, input_dim, matched_k), test_loader, "TopK_vanilla")

    print(f"[{dataset_name} seed={seed}] Training TopK + barrier...")
    topk_barrier = train_topk_plus_barrier(train_loader, input_dim, matched_k, lambda_barrier, lambda_budget, rho)
    results["TopK_plus_barrier"] = evaluate_full(topk_barrier, test_loader, "TopK_plus_barrier")

    print(f"[{dataset_name} seed={seed}] Training Tuned L1 (vanilla)...")
    results["TunedL1_vanilla"] = evaluate_full(train_l1(train_loader, input_dim), test_loader, "TunedL1_vanilla")

    print(f"[{dataset_name} seed={seed}] Training Tuned L1 + barrier...")
    l1_barrier = train_l1_plus_barrier(train_loader, input_dim, l1_coef, lambda_barrier, lambda_budget, rho)
    results["TunedL1_plus_barrier"] = evaluate_full(l1_barrier, test_loader, "TunedL1_plus_barrier")

    out_path = f"results/mixed_regularizer/{dataset_name}_seed{seed}_b{lambda_barrier}_u{lambda_budget}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: mixed regularizer (barrier bolted onto existing methods) ===")
    header = f"{'Model':22s} {'Sparsity':>9s} {'MSE':>8s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:22s} {r['relative_sparsity']:9.3f} {r['mse']:8.4f} "
            f"{r['mean_purity_top10pct_by_importance']:12.3f} {r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambda-barrier", type=float, default=0.005)
    parser.add_argument("--lambda-budget", type=float, default=0.5)
    parser.add_argument("--l1-coef", type=float, default=0.01)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambda_barrier, args.lambda_budget, args.l1_coef)
