"""
Redesigned Top-k-compatible version of the identity-tracked, boundary-
escalating principle, fixing the incompatibility diagnosed in
mixed_regularizer_experiment.py: bolting the raw log-barrier (on
pre-activation sign) onto Top-k caused MSE to explode 6-32x and get WORSE
with more regularization, because the barrier pressures every one of N
pre-activations toward firing ~half the time while Top-k's hard mask
discards all but k of them regardless -- two mechanisms fighting over the
same quantity for no reason, since Top-k already controls sparsity exactly.

Fix: stop penalizing raw pre-activation sign. Penalize each feature's
SELECTION FREQUENCY instead -- a smooth proxy for "was this feature one of
the k selected, across the batch" -- so the auxiliary term only intervenes
on the actual pathology (one feature getting selected too often, becoming
a hub) and leaves Top-k's per-input adaptive selection otherwise
untouched. Also drops the budget term entirely: Top-k already fixes
sparsity exactly (exactly k active every sample), so there is nothing left
for a budget term to do except redundantly fight the hard mask, which was
plausibly part of the original interference.

    kth_value_i       = the k-th largest pre-activation in sample i (detached
                         -- used only as a threshold for the soft indicator,
                         not backpropagated through to avoid a second,
                         redundant selection pressure)
    soft_selected_f,i  = sigmoid((pre_act_f,i - kth_value_i) / tau)
                         ~1 if f is (near) selected in sample i, ~0 if not
    p_hat_f            = mean_batch soft_selected_f,i   (selection frequency)
    loss = MSE + lambda_barrier * sum_f [-log(p_hat_f) - log(1 - p_hat_f)]

Same barrier shape as before (identity-tracked, boundary-escalating), now
applied to the quantity Top-k actually controls.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import TopKSAE, DATASETS, latent_dim, batch_size, lr, epochs, device, train_topk
from full_positioning_replication import evaluate_full


def soft_selection_indicator(pre_act, k, tau=0.1):
    """Smooth per-sample indicator of "is feature f among the top-k",
    without altering the hard selection actually used for reconstruction.
    kth_value is detached so this is purely an auxiliary-loss proxy, not a
    second gradient path into the selection itself."""
    n = pre_act.shape[1]
    kth_value, _ = torch.kthvalue(pre_act, n - k + 1, dim=1, keepdim=True)
    return torch.sigmoid((pre_act - kth_value.detach()) / tau)


def selection_barrier_penalty(pre_act, k, tau=0.1, epsilon=1e-6):
    p_hat = soft_selection_indicator(pre_act, k, tau).mean(dim=0).clamp(epsilon, 1 - epsilon)
    return (-torch.log(p_hat) - torch.log(1 - p_hat)).sum()


def train_topk_selection_barrier(train_loader, input_dim, k, lambda_barrier, tau=0.1):
    """Uses TopKSAE so the returned model's own .forward() applies the same
    top-k selection at evaluation time that it was trained with -- using a
    plain SimpleSAE here was the bug in the first version of this script:
    evaluate_full() calls model(x), and a bare SimpleSAE's forward() has no
    idea about top-k selection, so it silently evaluated a fully dense
    reconstruction that was never trained (the model only ever saw the
    top-k-masked code as decoder input during training), producing the
    catastrophic MSE we first observed. TopKSAE.forward() bakes the
    selection in, so training and evaluation are consistent."""
    model = TopKSAE(input_dim, latent_dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            # Same post-ReLU, pre-topk quantity TopKSAE itself ranks by,
            # recomputed here only to keep the barrier's gradient path
            # independent of the (already-selected) decoder input z.
            relu_act = torch.relu(model.encoder(batch))
            barrier = selection_barrier_penalty(relu_act, k, tau)
            loss = criterion(recon, batch) + lambda_barrier * barrier
            loss.backward()
            optimizer.step()
    return model


def run(dataset_name, seed, rho, lambda_barrier_values, tau=0.1):
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

    for lb in lambda_barrier_values:
        label = f"TopK_selbarrier_{lb}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = train_topk_selection_barrier(train_loader, input_dim, matched_k, lb, tau)
        results[label] = evaluate_full(model, test_loader, label)

    out_path = f"results/topk_selection_barrier/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: Top-k + selection-frequency barrier ===")
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
    parser.add_argument("--lambda-barrier", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.05])
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambda_barrier, args.tau)
