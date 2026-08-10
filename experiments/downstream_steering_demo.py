"""
Downstream-consequence steering demo.

steering_shrinkage_benchmark.py's steering_impact metric is reconstruction-
internal: it measures how much a feature intervention changes the SAE's own
output, not whether it changes anything a downstream reader would call a
"behavior." A skeptical reviewer will discount an internal-metric-only
steering claim. This script adds the missing piece: train a linear probe
(classifier) on each SAE's clean latents, then measure how often a single-
feature ablation or clamp intervention actually flips the probe's predicted
label, plus a continuous probability-shift metric (flip rate alone can be
near-zero if the classifier is confident/robust, so it's not sensitive
enough on its own).

Reuses the matched-sparsity models and training code from
steering_shrinkage_benchmark.py so this is directly comparable to the
existing shrinkage/steering results, just with a real downstream consequence
attached instead of a reconstruction-L2 proxy.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from steering_shrinkage_benchmark import (
    DATASETS,
    latent_dim,
    batch_size,
    lr,
    train_gini,
    train_l1,
    train_gated,
    train_topk,
    device,
)
from torchvision import transforms

N_STEERING_SAMPLES = 200
N_FEATURES_PER_SAMPLE = 5
CLAMP_SCALE = 3.0
PROBE_EPOCHS = 10


class LinearProbe(nn.Module):
    def __init__(self, d_in, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, z):
        return self.linear(z)


def extract_latents(model, loader):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for batch, y in loader:
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.cpu())
            labels.append(y)
    return torch.cat(zs), torch.cat(labels)


def train_probe(z_train, y_train, d_latent):
    probe = LinearProbe(d_latent).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(z_train, y_train), batch_size=batch_size, shuffle=True)
    for _ in range(PROBE_EPOCHS):
        for z_batch, y_batch in loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(z_batch), y_batch)
            loss.backward()
            optimizer.step()
    return probe


def downstream_steering_metrics(z_test, probe, n_samples=N_STEERING_SAMPLES, n_features=N_FEATURES_PER_SAMPLE):
    probe.eval()
    z_test = z_test[:n_samples].to(device)
    with torch.no_grad():
        logits_orig = probe(z_test)
        probs_orig = torch.softmax(logits_orig, dim=1)
        pred_orig = logits_orig.argmax(dim=1)

        ablate_flips, ablate_shifts = [], []
        clamp_flips, clamp_shifts = [], []

        for i in range(z_test.shape[0]):
            active_idx = torch.where(z_test[i].abs() > 1e-6)[0]
            if len(active_idx) == 0:
                continue
            chosen = active_idx[torch.randperm(len(active_idx))[:n_features]]
            for f in chosen:
                z_ablate = z_test[i].clone()
                z_ablate[f] = 0.0
                probs_ablate = torch.softmax(probe(z_ablate.unsqueeze(0)), dim=1).squeeze(0)
                ablate_flips.append(float(probs_ablate.argmax().item() != pred_orig[i].item()))
                ablate_shifts.append((probs_ablate[pred_orig[i]] - probs_orig[i, pred_orig[i]]).abs().item())

                z_clamp = z_test[i].clone()
                z_clamp[f] = z_clamp[f] * CLAMP_SCALE
                probs_clamp = torch.softmax(probe(z_clamp.unsqueeze(0)), dim=1).squeeze(0)
                clamp_flips.append(float(probs_clamp.argmax().item() != pred_orig[i].item()))
                clamp_shifts.append((probs_clamp[pred_orig[i]] - probs_orig[i, pred_orig[i]]).abs().item())

    return {
        "ablate_flip_rate": float(torch.tensor(ablate_flips).mean()) if ablate_flips else 0.0,
        "ablate_prob_shift": float(torch.tensor(ablate_shifts).mean()) if ablate_shifts else 0.0,
        "clamp_flip_rate": float(torch.tensor(clamp_flips).mean()) if clamp_flips else 0.0,
        "clamp_prob_shift": float(torch.tensor(clamp_shifts).mean()) if clamp_shifts else 0.0,
        "n_interventions": len(ablate_flips),
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

    print(f"[{dataset_name} seed={seed}] Training Gini SAE + probe...")
    gini_model = train_gini(train_loader, input_dim)
    z_train, y_train = extract_latents(gini_model, train_loader)
    z_test, y_test = extract_latents(gini_model, test_loader)
    probe = train_probe(z_train, y_train, latent_dim)
    results["GiniSAE"] = downstream_steering_metrics(z_test, probe)
    mean_active = (z_train.abs() > 1e-6).sum(dim=1).float().mean().item()
    matched_k = max(1, round(mean_active))

    print(f"[{dataset_name} seed={seed}] Training Tuned L1 SAE + probe...")
    l1_model = train_l1(train_loader, input_dim)
    z_train, y_train = extract_latents(l1_model, train_loader)
    z_test, y_test = extract_latents(l1_model, test_loader)
    probe = train_probe(z_train, y_train, latent_dim)
    results["TunedL1"] = downstream_steering_metrics(z_test, probe)

    print(f"[{dataset_name} seed={seed}] Training Gated SAE + probe...")
    gated_model = train_gated(train_loader, input_dim)
    z_train, y_train = extract_latents(gated_model, train_loader)
    z_test, y_test = extract_latents(gated_model, test_loader)
    probe = train_probe(z_train, y_train, latent_dim)
    results["GatedSAE"] = downstream_steering_metrics(z_test, probe)

    print(f"[{dataset_name} seed={seed}] Training Top-K SAE (k={matched_k}) + probe...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    z_train, y_train = extract_latents(topk_model, train_loader)
    z_test, y_test = extract_latents(topk_model, test_loader)
    probe = train_probe(z_train, y_train, latent_dim)
    results["TopK"] = downstream_steering_metrics(z_test, probe)

    out_path = f"results/downstream_steering/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed} ===")
    header = f"{'Model':10s} {'AblFlip':>8s} {'AblShift':>9s} {'ClmpFlip':>9s} {'ClmpShift':>10s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:10s} {r['ablate_flip_rate']:8.3f} {r['ablate_prob_shift']:9.4f} "
            f"{r['clamp_flip_rate']:9.3f} {r['clamp_prob_shift']:10.4f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.seed)
