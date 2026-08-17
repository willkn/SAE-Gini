"""
Second boundary-escalating instantiation, testing whether the paper's
design principle (identity-tracked + boundary-escalating gradient) is
separable from the specific KL-divergence formula Rate-KL uses, or whether
Rate-KL's results are an accident of that one functional form.

Note on why reverse-KL is NOT a valid second instance (considered and
rejected): KL(p_hat_f || rho) = p*log(p/rho) + (1-p)*log((1-p)/(1-rho))
is actually FINITE at p=0 and p=1 (each term is of the form x*log(x),
which -> 0 as x -> 0, not infinity) -- it is smoothed by its own prefactor
and does NOT escalate at the boundary. It would not test the principle;
it would just be a weaker, non-escalating variant.

Instead we use a log-barrier, borrowed from interior-point optimization,
which is unrelated in form to KL divergence but shares exactly the two
required properties: identity-tracked (one term per feature, summed) and
boundary-escalating (diverges as p_hat_f approaches either boundary):

    barrier(p) = -log(p) - log(1-p)

This is minimized at p=0.5 (not at a target rate), so on its own it does
not induce sparsity -- it only forbids the two degenerate extremes (dead,
p~0; hub, p~1). We combine it with a separate mean-matching budget term
(the same idea as the cap+budget attempt, but paired here with a genuine
divergent barrier instead of a bounded squared-hinge) to also hit a
sparsity target:

    loss = MSE + lambda_barrier * sum_f barrier(p_hat_f)
                + lambda_budget  * (sum_f p_hat_f - N*rho)^2 / (N*rho)^2

If this also avoids the hub pathology (high top-decile purity vs. Top-k),
the design principle -- not the KL formula specifically -- is doing the
work, which is the claim the paper needs a second data point for.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import DATASETS, latent_dim, batch_size, lr, epochs, device, train_topk
from rate_kl_sae import RateKLSAE, _EvalWrapper
from full_positioning_replication import evaluate_full


def barrier_budget_penalty(pre_act, rho, tau=0.1, epsilon=1e-6):
    p_hat = torch.sigmoid(pre_act / tau).mean(dim=0).clamp(epsilon, 1 - epsilon)
    n = p_hat.shape[0]
    barrier = (-torch.log(p_hat) - torch.log(1 - p_hat)).sum()
    budget = ((p_hat.sum() - n * rho) ** 2) / (n * rho) ** 2
    return barrier, budget


def train_rate_barrier(train_loader, input_dim, rho, lambda_barrier, lambda_budget, tau=0.1):
    model = RateKLSAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, pre_act = model(batch)
            barrier, budget = barrier_budget_penalty(pre_act, rho, tau)
            loss = criterion(recon, batch) + lambda_barrier * barrier + lambda_budget * budget
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, rho, lambda_pairs):
    """lambda_pairs: list of (lambda_barrier, lambda_budget) tuples."""
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for lb, lu in lambda_pairs:
        label = f"RateBarrier_b{lb}_u{lu}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = _EvalWrapper(train_rate_barrier(train_loader, input_dim, rho, lb, lu))
        results[label] = evaluate_full(model, test_loader, label)

    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK reference (k={matched_k})...")
    results["TopK_reference"] = evaluate_full(train_topk(train_loader, input_dim, matched_k), test_loader, "TopK_reference")

    out_path = f"results/rate_barrier/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: log-barrier second instantiation ===")
    header = f"{'Model':24s} {'Sparsity':>9s} {'MSE':>8s} {'Top10Purity':>12s} {'Ablate':>8s} {'Clamp':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:24s} {r['relative_sparsity']:9.3f} {r['mse']:8.4f} "
            f"{r['mean_purity_top10pct_by_importance']:12.3f} {r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument(
        "--lambda-pairs",
        nargs="+",
        default=["0.001,0.1", "0.005,0.5", "0.01,1.0"],
        help="comma-separated (lambda_barrier,lambda_budget) pairs",
    )
    args = parser.parse_args()
    pairs = [tuple(float(x) for x in p.split(",")) for p in args.lambda_pairs]
    run(args.dataset, args.seed, args.rho, pairs)
