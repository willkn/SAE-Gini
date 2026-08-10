"""
Tests the shrinkage-mechanism explanation for Gini's steering-impact result
(see paper Sec. "Quantitative Interpretability and Feature Steering") against
the literature's account of why L1 hurts causal feature quality:

  - Wright & Sharkey (2024), "Addressing Feature Suppression in SAEs": L1
    penalizes activation magnitude directly, so SGD shrinks true feature
    magnitude toward zero to cut the penalty cheaply ("shrinkage").
  - Gated SAEs (Rajamanoharan et al., 2024) / JumpReLU (Rajamanoharan et al.,
    2024b) / TopK SAEs (Gao et al., 2024) all fix this by decoupling "does
    this feature fire" from "how large is it when it fires" -- Gini's
    scale-invariant penalty is a different route to the same decoupling.

Prediction: relative to L1 at matched sparsity, Gini features that DO fire
should (1) have higher mean magnitude (less shrinkage) and (2) co-occur with
fewer other active features per sample (less superposition), and single
feature interventions should have a larger, cleaner causal effect
(steering). We test all of this side by side, plus the Gated-SAE and TopK
baselines the literature already proposes as shrinkage fixes, so Gini isn't
only benchmarked against the thing everyone agrees is flawed.

Also replaces the previous steering_test (interpretability_metrics.py),
which measured a single feature on a single sample, with an estimate
averaged over many (sample, feature) pairs, and adds a "clamp" (amplify an
active feature) variant alongside the existing "ablate" (zero it) variant --
literature steering results (e.g. Templeton et al. 2024, clamping features
to induce behavior) are about adding/amplifying a direction, not just
removing one, and necessity (ablation) and sufficiency/leverage (clamping)
can behave very differently.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 1024
batch_size = 256
lr = 1e-3
epochs = 10
N_STEERING_SAMPLES = 200
N_FEATURES_PER_SAMPLE = 5
CLAMP_SCALE = 3.0

DATASETS = {
    "fashion_mnist": (datasets.FashionMNIST, 784),
    "mnist": (datasets.MNIST, 784),
    "cifar10": (datasets.CIFAR10, 3072),
}


# --- Models (same simple Linear-ReLU-Linear architecture across all four,
# so comparisons are architecture-matched, not just sparsity-matched) ---


class SimpleSAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


class GatedSAE(nn.Module):
    """Simplified Gated SAE (Rajamanoharan et al., 2024): decouples the
    sparsity gate from the reconstruction magnitude."""

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.gate_threshold = nn.Parameter(torch.zeros(d_latent))
        self.decoder = nn.Linear(d_latent, d_in)
        self.r_bias = nn.Parameter(torch.zeros(d_latent))

    def forward(self, x):
        pre_act = self.encoder(x)
        gate = torch.relu(pre_act - self.gate_threshold)
        z = (gate > 0).float() * (pre_act + self.r_bias)
        recon = self.decoder(z)
        return recon, z


class TopKSAE(nn.Module):
    """Gao et al. (2024): no magnitude penalty, hard top-k selection."""

    def __init__(self, d_in, d_latent, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        pre_act = torch.relu(self.encoder(x))
        z = torch.zeros_like(pre_act)
        k = min(self.k, pre_act.shape[1])
        values, indices = torch.topk(pre_act, k, dim=1)
        z.scatter_(1, indices, values)
        recon = self.decoder(z)
        return recon, z


def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini


# --- Training ---


def train_gini(train_loader, input_dim):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) - 0.1 * differentiable_gini(z).mean()
            loss.backward()
            optimizer.step()
    return model


def train_l1(train_loader, input_dim):
    model = SimpleSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) + 0.01 * z.abs().mean()
            loss.backward()
            optimizer.step()
    return model


def train_gated(train_loader, input_dim):
    model = GatedSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) + 0.05 * z.abs().mean()
            loss.backward()
            optimizer.step()
    return model


def train_topk(train_loader, input_dim, k):
    model = TopKSAE(input_dim, latent_dim, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
    return model


# --- Shrinkage-mechanism metrics ---


def shrinkage_metrics(model, test_loader):
    model.eval()
    all_z = []
    mse_total, n_batches = 0.0, 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            mse_total += criterion(recon, batch).item()
            n_batches += 1
            all_z.append(z.cpu())
    z_all = torch.cat(all_z, dim=0)
    active_mask = z_all.abs() > 1e-6

    active_per_sample = active_mask.sum(dim=1).float()
    active_values = z_all[active_mask].abs()

    return {
        "mse": mse_total / n_batches,
        "relative_sparsity": 1.0 - (active_per_sample.mean().item() / latent_dim),
        "mean_active_features_per_sample": active_per_sample.mean().item(),
        "mean_active_magnitude": active_values.mean().item() if active_values.numel() else 0.0,
        "dead_features": int((active_mask.sum(dim=0) == 0).sum().item()),
    }


# --- Steering: averaged over many (sample, feature) pairs, ablate + clamp ---


def steering_metrics(model, test_loader, n_samples=N_STEERING_SAMPLES, n_features=N_FEATURES_PER_SAMPLE):
    model.eval()
    batch, _ = next(iter(test_loader))
    batch = batch[:n_samples].to(device)

    with torch.no_grad():
        recon_orig, z = model(batch)

    ablate_impacts, clamp_impacts = [], []
    with torch.no_grad():
        for i in range(batch.shape[0]):
            active_idx = torch.where(z[i].abs() > 1e-6)[0]
            if len(active_idx) == 0:
                continue
            chosen = active_idx[torch.randperm(len(active_idx))[:n_features]]
            for f in chosen:
                base_norm = recon_orig[i].norm().item() + 1e-8

                z_ablate = z[i].clone()
                z_ablate[f] = 0.0
                recon_ablate = model.decoder(z_ablate.unsqueeze(0)).squeeze(0)
                ablate_impacts.append((recon_orig[i] - recon_ablate).norm().item() / base_norm)

                z_clamp = z[i].clone()
                z_clamp[f] = z_clamp[f] * CLAMP_SCALE
                recon_clamp = model.decoder(z_clamp.unsqueeze(0)).squeeze(0)
                clamp_impacts.append((recon_orig[i] - recon_clamp).norm().item() / base_norm)

    return {
        "steering_impact_ablate": float(torch.tensor(ablate_impacts).mean()) if ablate_impacts else 0.0,
        "steering_impact_clamp": float(torch.tensor(clamp_impacts).mean()) if clamp_impacts else 0.0,
        "n_interventions": len(ablate_impacts),
    }


def run(dataset_name, seed):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    print(f"[{dataset_name} seed={seed}] Training Gini SAE...")
    gini_model = train_gini(train_loader, input_dim)
    results["GiniSAE"] = shrinkage_metrics(gini_model, test_loader)
    results["GiniSAE"].update(steering_metrics(gini_model, test_loader))

    # Match TopK's k to Gini's observed active-feature count for a
    # sparsity-matched comparison, rather than guessing a k in advance.
    matched_k = max(1, round(results["GiniSAE"]["mean_active_features_per_sample"]))

    print(f"[{dataset_name} seed={seed}] Training Tuned L1 SAE...")
    l1_model = train_l1(train_loader, input_dim)
    results["TunedL1"] = shrinkage_metrics(l1_model, test_loader)
    results["TunedL1"].update(steering_metrics(l1_model, test_loader))

    print(f"[{dataset_name} seed={seed}] Training Gated SAE...")
    gated_model = train_gated(train_loader, input_dim)
    results["GatedSAE"] = shrinkage_metrics(gated_model, test_loader)
    results["GatedSAE"].update(steering_metrics(gated_model, test_loader))

    print(f"[{dataset_name} seed={seed}] Training Top-K SAE (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    results["TopK"] = shrinkage_metrics(topk_model, test_loader)
    results["TopK"].update(steering_metrics(topk_model, test_loader))

    out_path = f"results/shrinkage_replication/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{dataset_name} seed={seed}] done -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run(args.dataset, args.seed)

    print("\n=== Summary ===")
    header = f"{'Model':10s} {'MSE':>8s} {'Sparsity':>9s} {'ActFeat':>8s} {'ActMag':>8s} {'Ablate':>8s} {'Clamp':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:10s} {r['mse']:8.4f} {r['relative_sparsity']:9.3f} "
            f"{r['mean_active_features_per_sample']:8.2f} {r['mean_active_magnitude']:8.3f} "
            f"{r['steering_impact_ablate']:8.4f} {r['steering_impact_clamp']:8.4f}"
        )
