"""
Class-level adaptation of the feature-absorption metric from Chanin et al.
2024 ("A is for Absorption: Studying Feature Splitting and Absorption in
Sparse Autoencoders").

Absorption (their sense): a concept's dedicated ("main") latent develops
holes -- it fails to fire on concept instances where a co-occurring, more
specific latent has absorbed the concept's direction into its own decoder
vector. Sparsity penalties cause it: firing one absorbed latent is cheaper
than firing two clean ones. This is distinct from the hub pathology we
diagnosed earlier (hubs OVER-fire everywhere; absorbed concepts' main
latents UNDER-fire), but both are sparsity-induced monosemanticity
failures.

Mechanistic prediction being tested: TopK has no per-latent rate
constraint, so absorption is free; under rate-KL, an absorbing latent's
firing rate rises toward the union of the merged concepts' frequencies,
which the KL term punishes per-latent. So TopK should show measurable
absorption and rate-KL should show significantly less.

Adaptation to image classes (we have no token-level concept labels):
  1. Per class c, fit a logistic-probe direction d_c in pixel space.
  2. Main latent for c = latent whose binary firing has highest F1
     predicting class c on held-out data.
  3. Absorption instance = a class-c sample where the main latent is
     SILENT, but other latents whose decoder columns are cosine-aligned
     with d_c fire and jointly carry at least ABSORB_FRAC of the
     class-direction mass the main latent typically carries when it fires.
  4. Absorption score = fraction of class-c samples that are absorption
     instances, averaged over classes. Also reported: main-latent F1
     (concept-tracking quality) and main-latent miss rate (how often the
     main latent is silent on its concept at all, absorbed or not).

Caveat stated upfront (and to state in the paper): class-level concepts
are far coarser than Chanin et al.'s token-level features, and pixel-space
probe directions are a proxy for concept directions. This is "a
class-level adaptation of the absorption metric," not their protocol.
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
    device,
    train_topk,
    train_l1,
)
from rate_kl_sae import train_rate_kl, _EvalWrapper

COS_ALIGN_THRESHOLD = 0.25
ABSORB_FRAC = 0.5
NUM_CLASSES = 10


def fit_class_probes(train_loader, input_dim, epochs=3, lr=1e-2):
    """One-vs-rest logistic probes in pixel space; returns [C, input_dim]
    unit direction vectors."""
    probe = nn.Linear(input_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch, y in train_loader:
            batch, y = batch.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(batch), y)
            loss.backward()
            optimizer.step()
    d = probe.weight.detach()  # [C, input_dim]
    return d / d.norm(dim=1, keepdim=True)


def collect_latents(model, loader):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch, y in loader:
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.cpu())
            ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def main_latent_f1(z, y):
    """For each class, the latent whose binary firing best predicts the
    class (max F1), computed densely. Returns per-class (latent_idx, f1)."""
    fired = (z.abs() > 1e-6).float()  # [N, L]
    results = []
    for c in range(NUM_CLASSES):
        target = (y == c).float()  # [N]
        n_pos = target.sum()
        tp = fired.T @ target  # [L]
        fp = fired.T @ (1 - target)
        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / n_pos.clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        best = int(f1.argmax().item())
        results.append((best, float(f1[best].item())))
    return results


def absorption_scores(model, z, y, probe_dirs):
    """Per-class absorption rate, main-latent miss rate, and F1."""
    decoder_weight = model.decoder.weight.detach().cpu()  # [input_dim, L]
    col_units = decoder_weight / decoder_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
    probe_dirs = probe_dirs.cpu()  # [C, input_dim]
    cos = probe_dirs @ col_units  # [C, L] cosine of each latent's decoder col with each class direction

    mains = main_latent_f1(z, y)
    per_class = []
    for c in range(NUM_CLASSES):
        main_idx, f1 = mains[c]
        mask_c = y == c
        z_c = z[mask_c]  # [Nc, L]
        if z_c.shape[0] == 0:
            continue

        main_fires = z_c[:, main_idx].abs() > 1e-6
        miss_rate = float((~main_fires).float().mean())

        # Class-direction mass the main latent carries when it does fire:
        # projection of its decoder contribution onto d_c.
        main_contrib = z_c[main_fires][:, main_idx] * (decoder_weight[:, main_idx] @ probe_dirs[c])
        typical_main_mass = float(main_contrib.abs().mean()) if main_fires.any() else 0.0

        # Aligned other latents (excluding the main one).
        aligned = (cos[c].abs() >= COS_ALIGN_THRESHOLD).clone()
        aligned[main_idx] = False
        aligned_idx = torch.where(aligned)[0]

        if typical_main_mass > 0 and len(aligned_idx) > 0:
            silent = ~main_fires
            z_silent = z_c[silent][:, aligned_idx]  # [Ns, A]
            contrib = z_silent * (decoder_weight[:, aligned_idx].T @ probe_dirs[c]).unsqueeze(0)
            carried = contrib.sum(dim=1).abs()  # [Ns] class-direction mass via aligned latents
            absorbed = (carried >= ABSORB_FRAC * typical_main_mass).float()
            absorption_rate = float(absorbed.sum() / z_c.shape[0])
        else:
            absorption_rate = 0.0

        per_class.append(
            {
                "class": c,
                "main_latent": main_idx,
                "main_f1": f1,
                "main_miss_rate": miss_rate,
                "absorption_rate": absorption_rate,
                "n_aligned_latents": int(len(aligned_idx)),
            }
        )

    mean = lambda k: float(torch.tensor([p[k] for p in per_class]).mean())
    return {
        "per_class": per_class,
        "mean_absorption_rate": mean("absorption_rate"),
        "mean_main_f1": mean("main_f1"),
        "mean_main_miss_rate": mean("main_miss_rate"),
    }


def run(dataset_name, seed, rho, rate_kl_lambdas, tau):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[{dataset_name} seed={seed}] Fitting class probes...")
    probe_dirs = fit_class_probes(train_loader, input_dim)

    results = {}

    for lam in rate_kl_lambdas:
        label = f"RateKL_rho{rho}_lam{lam}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        model = _EvalWrapper(train_rate_kl(train_loader, input_dim, rho=rho, lambda_kl=lam, tau=tau))
        z, y = collect_latents(model, test_loader)
        results[label] = absorption_scores(model, z, y, probe_dirs)

    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    z, y = collect_latents(topk_model, test_loader)
    results["TopK"] = absorption_scores(topk_model, z, y, probe_dirs)

    print(f"[{dataset_name} seed={seed}] Training TunedL1...")
    l1_model = train_l1(train_loader, input_dim)
    z, y = collect_latents(l1_model, test_loader)
    results["TunedL1"] = absorption_scores(l1_model, z, y, probe_dirs)

    out_path = f"results/absorption/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: absorption (class-level adaptation of Chanin et al. 2024) ===")
    header = f"{'Model':24s} {'Absorption':>11s} {'MainF1':>8s} {'MainMiss':>9s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:24s} {r['mean_absorption_rate']:11.4f} {r['mean_main_f1']:8.3f} "
            f"{r['mean_main_miss_rate']:9.3f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.09)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01])
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.rho, args.lambdas, args.tau)
