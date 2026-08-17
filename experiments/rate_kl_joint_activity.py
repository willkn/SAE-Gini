"""
Fix attempt for the activity-bimodality failure mode diagnosed in
rate_kl_sae.py: KL(rho||p_hat_f) constrains each feature's MARGINAL firing
rate (averaged over the batch) but says nothing about the JOINT
distribution of active counts per sample. At CIFAR-10, lambda=0.01, this
gap let 57% of test samples collapse to <5 active features while a dense
minority absorbed the reconstruction and steering mass, satisfying every
feature's marginal target while violating the per-sample sparsity target
almost everywhere.

Fix: add a second, per-sample term alongside the existing per-feature
term, penalizing squared deviation of each sample's own active count from
the target N*rho:

    per_sample_penalty = mean_batch (count_i - N*rho)^2 / (N*rho)^2

where count_i is a smooth (sigmoid-based) per-sample active-feature count,
differentiable for the same reason the firing-rate surrogate is. This
directly constrains the joint statistic the marginal term leaves free,
without touching magnitude (still computed from firing indicators, not
activation values) or removing the per-feature term (still needed to
prevent hubs -- Section "Diagnosis" of the paper). The risk this script is
designed to expose: enforcing per-sample counts too, on top of per-feature
rates, could reintroduce hub-like behavior if the two constraints jointly
make "the same few big features, every sample" the easiest way to hit both
targets at once -- exactly the failure mode Rate-KL was built to avoid.
We evaluate on the same purity/importance-concentration diagnostic used
throughout to check for that directly, not just the bimodality gate.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import DATASETS, latent_dim, batch_size, lr, epochs, device, train_topk
from rate_kl_sae import RateKLSAE, rate_kl_penalty, _EvalWrapper, per_sample_activity_stats
from full_positioning_replication import evaluate_full


def per_sample_activity_penalty(pre_act, rho, tau=0.1):
    """Smooth per-sample active count, penalized toward N*rho per sample
    (not just on average across the batch)."""
    n = pre_act.shape[1]
    soft_count = torch.sigmoid(pre_act / tau).sum(dim=1)  # [batch]
    target = n * rho
    return ((soft_count - target) ** 2).mean() / (target ** 2)


def train_rate_kl_joint(train_loader, input_dim, rho, lambda_kl, lambda_joint, tau=0.1):
    model = RateKLSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, pre_act = model(batch)
            loss = (
                criterion(recon, batch)
                + lambda_kl * rate_kl_penalty(pre_act, rho, tau)
                + lambda_joint * per_sample_activity_penalty(pre_act, rho, tau)
            )
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, rho, lambda_kl, lambda_joint_values):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for lj in lambda_joint_values:
        label = f"RateKLJoint_lam{lambda_kl}_joint{lj}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = _EvalWrapper(train_rate_kl_joint(train_loader, input_dim, rho, lambda_kl, lj))
        r = evaluate_full(model, test_loader, label)
        r["activity_distribution"] = per_sample_activity_stats(model, test_loader)
        results[label] = r

    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    r = evaluate_full(topk_model, test_loader, "TopK_reference")
    r["activity_distribution"] = per_sample_activity_stats(topk_model, test_loader)
    results["TopK_reference"] = r

    out_path = f"results/rate_kl_joint/{dataset_name}_seed{seed}_lam{lambda_kl}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: joint activity fix (lambda_kl={lambda_kl}) ===")
    header = (
        f"{'Model':32s} {'Sparsity':>9s} {'MSE':>8s} {'Top10Purity':>12s} "
        f"{'Ablate':>8s} {'lt5active':>10s} {'lt1active':>10s}"
    )
    print(header)
    for name, r in results.items():
        ad = r["activity_distribution"]
        print(
            f"{name:32s} {r['relative_sparsity']:9.3f} {r['mse']:8.4f} "
            f"{r['mean_purity_top10pct_by_importance']:12.3f} {r['steering_impact_ablate']:8.4f} "
            f"{ad['frac_samples_lt5_active']:10.3f} {ad['frac_samples_lt1_active']:10.3f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambda-kl", type=float, default=0.01, help="the config that went bimodal on CIFAR-10")
    parser.add_argument("--lambda-joint", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambda_kl, args.lambda_joint)
