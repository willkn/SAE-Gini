"""
Quantifies the gradient-inconsistency artifact flagged (but not measured) in
paper/main.tex sec. "Limitations & Discussion": when many latent entries tie
at exactly zero, torch.sort assigns them an arbitrary relative rank, and the
rank-based Gini formula routes a rank-dependent gradient back through the
inverse permutation. Entries with IDENTICAL values should, on any reasonable
notion of a well-posed objective, receive IDENTICAL gradients. This script
measures how far the hard-sort formulation departs from that property, and
compares it against the tie-aware soft-rank formulation in
experiments/soft_gini.py.

Same model/training setup as tie_analysis.py so results are directly
comparable to the existing results/tie_analysis.json.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from soft_gini import differentiable_gini_hard, differentiable_gini_soft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
lr = 1e-3
epochs = 5
TIE_TOL = 1e-6


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def tie_gradient_inconsistency(z_detached, gini_fn):
    """For each sample, group latent indices by tied |value|, compute
    d(gini)/dz via autograd, and measure the within-tie-group std of that
    gradient, normalized by the overall gradient scale for the sample.

    A perfectly tie-consistent objective gives every entry in a tie group
    the exact same gradient, so this statistic is exactly 0 for such an
    objective. Any nonzero value is gradient signal manufactured purely by
    the arbitrary tie-break order, not by anything the model did.
    """
    z = z_detached.clone().requires_grad_(True)
    gini = gini_fn(z).mean()
    (grad,) = torch.autograd.grad(gini, z)

    z_abs = torch.abs(z_detached)
    batch_size, n = z_abs.shape
    per_sample_inconsistency = []
    per_sample_tie_frac = []

    for i in range(batch_size):
        row = z_abs[i]
        order = torch.argsort(row)
        row_sorted = row[order]
        grad_sorted = grad[i][order]

        # boundaries between distinct-value groups
        diffs = torch.diff(row_sorted)
        group_id = torch.cat([torch.zeros(1, device=row.device), (diffs >= TIE_TOL).cumsum(0)])

        overall_grad_scale = grad[i].abs().mean().item() + 1e-12
        tied_entries = 0
        group_stds = []
        for gid in group_id.unique():
            mask = group_id == gid
            if mask.sum() > 1:
                tied_entries += int(mask.sum().item())
                group_stds.append(grad_sorted[mask].std(unbiased=False).item())

        per_sample_tie_frac.append(tied_entries / n)
        if group_stds:
            # normalize by overall gradient scale so this is comparable across
            # training (gradient magnitudes shrink as training converges)
            per_sample_inconsistency.append(float(np.mean(group_stds)) / overall_grad_scale)
        else:
            per_sample_inconsistency.append(0.0)

    return float(np.mean(per_sample_inconsistency)), float(np.mean(per_sample_tie_frac))


def run(gini_fn, label, out_path):
    torch.manual_seed(0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    stats = []
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            gini = gini_fn(z).mean()
            loss = mse - 0.2 * gini
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                with torch.no_grad():
                    z_eval = z.detach()
                inconsistency, tie_frac = tie_gradient_inconsistency(z_eval, gini_fn)
                stats.append(
                    {
                        "step": step,
                        "tie_frac": tie_frac,
                        "grad_inconsistency": inconsistency,
                    }
                )
            step += 1

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    late = [s["grad_inconsistency"] for s in stats[-20:]]
    late_tie = [s["tie_frac"] for s in stats[-20:]]
    print(f"[{label}] late-training tie fraction: {np.mean(late_tie):.3f}")
    print(f"[{label}] late-training normalized grad inconsistency: {np.mean(late):.4f}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["hard", "soft", "both"], default="both")
    args = parser.parse_args()

    if args.which in ("hard", "both"):
        run(differentiable_gini_hard, "hard-sort", "results/tie_gradient_hard.json")
    if args.which in ("soft", "both"):
        run(lambda x: differentiable_gini_soft(x, tau=0.01), "soft-rank", "results/tie_gradient_soft.json")
