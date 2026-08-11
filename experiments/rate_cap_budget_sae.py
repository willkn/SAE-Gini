"""
One-sided rate cap + global budget: fix for the absorption mechanism
diagnosed in rate_kl_sae.py's two-sided KL(rho || p_hat_f).

Diagnosis: KL(rho||p_hat_f) pulls every latent's firing rate toward rho
from BOTH sides. Any concept whose natural frequency exceeds rho (e.g. a
class base rate of 0.10 with rho=0.09) gets actively punished for
tracking it faithfully -- the model is forced to drop ~10% of that
concept's instances to hit the target, manufacturing exactly the
holes-with-compensation structure absorption consists of. This is a
different absorption mechanism than Chanin et al.'s (theirs comes from
sparsity pressure, ours came from rate-uniformity pressure), but the
lesion looks the same.

Fix: stop pulling every latent toward rho. Instead:
  1. A ONE-SIDED cap, applied only when p_hat_f exceeds rho_max (set above
     expected concept frequencies, e.g. 0.15): squared-hinge penalty. This
     still forbids hubs (p ~ 1) at full strength but places zero penalty on
     any latent below the cap, so a class-frequency latent (~0.10) is free
     to fire at its natural rate without being split.
  2. A GLOBAL BUDGET term holding the total expected active count
     (sum_f p_hat_f) near the sparsity target N*rho, so the cap alone
     doesn't just let the model go dense -- this replaces the sparsity
     pressure that the removed per-latent floor used to provide.

    cap_penalty    = mean_f relu(p_hat_f - rho_max)^2
    budget_penalty = ((sum_f p_hat_f) - N*rho_target)^2 / (N*rho_target)^2
    loss = MSE + lambda_cap * cap_penalty + lambda_budget * budget_penalty

Same RateKLSAE architecture and sigmoid firing-rate surrogate as
rate_kl_sae.py; only the loss changes.
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
    lr,
    epochs,
    device,
    steering_metrics,
    train_topk,
)
from feature_purity_analysis import feature_importance_and_purity, summarize
from rate_kl_sae import RateKLSAE, _EvalWrapper


def cap_budget_penalty(pre_act, rho_max, rho_target, tau=0.1, epsilon=1e-6):
    p_hat = torch.sigmoid(pre_act / tau).mean(dim=0).clamp(epsilon, 1 - epsilon)  # [n_latents]
    n = p_hat.shape[0]
    cap_penalty = torch.relu(p_hat - rho_max).pow(2).mean()
    budget_penalty = ((p_hat.sum() - n * rho_target) ** 2) / (n * rho_target) ** 2
    return cap_penalty, budget_penalty


def train_rate_cap_budget(train_loader, input_dim, rho_target, rho_max, lambda_cap, lambda_budget, tau=0.1):
    model = RateKLSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, pre_act = model(batch)
            cap_penalty, budget_penalty = cap_budget_penalty(pre_act, rho_max, rho_target, tau)
            loss = criterion(recon, batch) + lambda_cap * cap_penalty + lambda_budget * budget_penalty
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


def run(dataset_name, seed, rho_target, rho_max, lambda_pairs, tau):
    """lambda_pairs: list of (lambda_cap, lambda_budget) tuples."""
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for lc, lb in lambda_pairs:
        label = f"RateCapBudget_cap{lc}_budget{lb}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_rate_cap_budget(train_loader, input_dim, rho_target, rho_max, lc, lb, tau)
        results[label] = evaluate(_EvalWrapper(model), test_loader, len(test_ds), label)

    matched_k = max(1, round(rho_target * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    results["TopK_reference"] = evaluate(topk_model, test_loader, len(test_ds), "TopK_reference")

    out_path = f"results/rate_cap_budget/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: rate-cap-budget sweep (rho_target={rho_target}, rho_max={rho_max}) ===")
    header = (
        f"{'Model':32s} {'MSE':>8s} {'Sparsity':>9s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:32s} {r['mse']:8.4f} {r['relative_sparsity']:9.3f} "
            f"{r['mean_purity_top10pct_by_importance']:12.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho-target", type=float, default=0.09)
    parser.add_argument("--rho-max", type=float, default=0.15)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--lambda-pairs",
        nargs="+",
        default=["0.1,0.1", "0.5,0.1", "1.0,0.5"],
        help="comma-separated (lambda_cap,lambda_budget) pairs",
    )
    args = parser.parse_args()
    pairs = [tuple(float(x) for x in p.split(",")) for p in args.lambda_pairs]
    run(args.dataset, args.seed, args.rho_target, args.rho_max, pairs, args.tau)
