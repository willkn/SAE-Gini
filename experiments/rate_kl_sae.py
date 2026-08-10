"""
Firing-rate-targeted SAE ("rate-KL"): regularize how OFTEN each feature
fires, never how much.

Design rationale (from the Gini post-mortem in feature_purity_analysis.py,
gini_load_balanced.py, gini_dual_axis.py):
  - Per-occurrence steering impact of feature f is |z_f| * ||W_f|| when it
    fires, so the penalty must never touch magnitude (non-suppressive).
  - The hub pathology is a firing-RATE phenomenon: a hub is a feature
    active on ~every input. Distribution-shape objectives (row Gini,
    column Gini, usage balancing) all failed to remove hubs without killing
    steering because they constrain the shape of the activation histogram,
    not identity-level firing statistics.
  - TopK constrains per-sample count only; feature-level firing rates are
    unconstrained, which is why TopK's own top-importance features were
    fairly impure (0.036) with importance-Gini 0.58 in our runs.

Objective:

    p_hat_f = mean_batch sigmoid(a_f / tau)        (a = pre-ReLU encoder out)
    loss    = MSE + lambda * sum_f KL(rho || p_hat_f)

where KL(rho||p) = rho*log(rho/p) + (1-rho)*log((1-rho)/(1-p)) and rho is
the target firing rate. This is Ng's classic KL-sparsity autoencoder with
one crucial change: the classic version computes the feature statistic from
mean activation MAGNITUDE (suppressive -- L1 in a hat); ours uses firing
PROBABILITY, so magnitude never appears in the penalty.

Properties: expected per-sample active count = N*rho (a direct, soft,
fully differentiable version of TopK's knob -- no sort, no straight-through
estimator, O(N)); hubs (p_f ~ 1) are individually expensive with no
mean-over-features loophole; dead features (p_f ~ 0) are also penalized, so
Feature Revival is unnecessary. Using pre-ReLU activations in the surrogate
gives dead features a gradient path.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
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


class RateKLSAE(nn.Module):
    """Same Linear-ReLU-Linear architecture as SimpleSAE, but exposes the
    pre-ReLU activations needed for the firing-rate surrogate."""

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        pre_act = self.encoder(x)
        z = torch.relu(pre_act)
        recon = self.decoder(z)
        return recon, z, pre_act


def rate_kl_penalty(pre_act, rho, tau=0.1, epsilon=1e-6):
    """KL(rho || p_hat_f) summed over features, where p_hat_f is the smooth
    batch estimate of feature f's firing probability."""
    p_hat = torch.sigmoid(pre_act / tau).mean(dim=0).clamp(epsilon, 1 - epsilon)
    kl = rho * torch.log(rho / p_hat) + (1 - rho) * torch.log((1 - rho) / (1 - p_hat))
    return kl.sum()


def train_rate_kl(train_loader, input_dim, rho, lambda_kl, tau=0.1):
    model = RateKLSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, pre_act = model(batch)
            loss = criterion(recon, batch) + lambda_kl * rate_kl_penalty(pre_act, rho, tau)
            loss.backward()
            optimizer.step()
    return model


class _EvalWrapper(nn.Module):
    """Adapts RateKLSAE's 3-tuple forward to the (recon, z) interface the
    shared evaluation utilities expect."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.decoder

    def forward(self, x):
        recon, z, _ = self.model(x)
        return recon, z

    def eval(self):
        self.model.eval()
        return self


def evaluate(model, test_loader, test_len, label):
    stats = feature_importance_and_purity(model, test_loader)
    r = summarize(stats, label)
    mean_active = (stats["active_counts"].sum() / test_len).item()
    r["mean_active_features_per_sample"] = mean_active
    r["relative_sparsity"] = 1.0 - mean_active / latent_dim
    criterion = nn.MSELoss()
    mse_total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            mse_total += criterion(recon, batch).item()
            n_batches += 1
    r["mse"] = mse_total / n_batches
    r.update(steering_metrics(model, test_loader))
    r["activity_distribution"] = per_sample_activity_stats(model, test_loader)
    return r


def per_sample_activity_stats(model, test_loader):
    """Distribution of per-sample active-feature counts, to diagnose the
    n_interventions drop seen at high lambda: rate-KL pins the MEAN active
    count at N*rho, but the KL term constrains per-feature firing rates,
    not per-sample counts, so activity can go bimodal (some samples dense,
    some near-silent). Near-silent samples starve the steering metric of
    interventions and would also mean some inputs are barely represented."""
    model.eval()
    counts = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            _, z = model(batch)
            counts.append((z.abs() > 1e-6).sum(dim=1).cpu())
    counts = torch.cat(counts).float()
    quantiles = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q = torch.quantile(counts, quantiles)
    return {
        "quantiles": {f"p{int(p * 100)}": float(v) for p, v in zip(quantiles, q)},
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "frac_samples_lt5_active": float((counts < 5).float().mean()),
        "frac_samples_lt1_active": float((counts < 1).float().mean()),
    }


def run(dataset_name, seed, rho, lambda_values, tau):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for lam in lambda_values:
        label = f"RateKL_rho{rho}_lam{lam}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_rate_kl(train_loader, input_dim, rho=rho, lambda_kl=lam, tau=tau)
        results[label] = evaluate(_EvalWrapper(model), test_loader, len(test_ds), label)

    # TopK reference matched to the target firing rate's implied active count,
    # the same comparison target used throughout.
    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    results["TopK_reference"] = evaluate(topk_model, test_loader, len(test_ds), "TopK_reference")

    out_path = f"results/rate_kl/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: rate-KL sweep (rho={rho}, tau={tau}) ===")
    header = (
        f"{'Model':24s} {'MSE':>8s} {'Sparsity':>9s} {'ImpGini':>8s} {'Top10Share':>11s} "
        f"{'MeanPurity':>11s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:24s} {r['mse']:8.4f} {r['relative_sparsity']:9.3f} "
            f"{r['importance_gini_coefficient']:8.3f} {r['top10_importance_share']:11.3f} "
            f"{r['mean_purity']:11.3f} {r['mean_purity_top10pct_by_importance']:12.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09, help="target firing rate (0.09*1024 ~ 92 active, matching prior runs)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambdas, args.tau)
