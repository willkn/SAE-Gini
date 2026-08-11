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


HUB_FIRE_RATE_THRESHOLD = 0.5  # a "main latent" firing on >=50% of ALL samples
                                # (not just its class) is a hub, not a concept
                                # tracker -- flag it rather than trust its F1.
N_NULL_TRIALS = 5  # random-latent-set controls per class, for null calibration


def _absorption_rate_for_latents(z_c, main_fires, aligned_idx, decoder_weight, probe_dir, typical_main_mass):
    """Fraction of class-c samples where the main latent is silent but the
    given latent set jointly carries >= ABSORB_FRAC of typical_main_mass in
    the class direction."""
    if typical_main_mass <= 0 or len(aligned_idx) == 0:
        return 0.0
    silent = ~main_fires
    z_silent = z_c[silent][:, aligned_idx]  # [Ns, A]
    contrib = z_silent * (decoder_weight[:, aligned_idx].T @ probe_dir).unsqueeze(0)
    carried = contrib.sum(dim=1).abs()  # [Ns]
    absorbed = (carried >= ABSORB_FRAC * typical_main_mass).float()
    return float(absorbed.sum() / z_c.shape[0])


def absorption_scores(model, z, y, probe_dirs):
    """Per-class absorption rate, main-latent miss rate, F1, main-latent
    firing rate (hub flag), and a null-calibrated control.

    Metric fixes applied here (see module docstring for the failure modes
    that motivated them):
      - main_fire_rate_overall: a latent firing on most of the WHOLE dataset
        (not just its assigned class) is a hub the max-F1 search picked as
        "main" by default (an always-on latent gets F1 = 2p/(1+p) on a
        p-frequency class purely from recall=1, not from tracking anything).
        Flagged via is_hub_main; hub-main classes are excluded from the
        mean_absorption_rate/mean_main_f1 that drive cross-method comparison,
        and reported separately.
      - null_absorption_rate: same computation as absorption_rate but on a
        random latent set of the same size as the aligned set, averaged over
        N_NULL_TRIALS draws. absorption_rate should be judged relative to
        this, not to zero -- weak class probes / low typical_main_mass can
        make the compensation criterion pass by chance.
    """
    decoder_weight = model.decoder.weight.detach().cpu()  # [input_dim, L]
    n_latents = decoder_weight.shape[1]
    col_units = decoder_weight / decoder_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
    probe_dirs = probe_dirs.cpu()  # [C, input_dim]
    cos = probe_dirs @ col_units  # [C, L] cosine of each latent's decoder col with each class direction

    overall_fire_rate = (z.abs() > 1e-6).float().mean(dim=0)  # [L], over ALL samples

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
        main_fire_rate_overall = float(overall_fire_rate[main_idx])
        is_hub_main = main_fire_rate_overall >= HUB_FIRE_RATE_THRESHOLD

        # Class-direction mass the main latent carries when it does fire:
        # projection of its decoder contribution onto d_c.
        main_contrib = z_c[main_fires][:, main_idx] * (decoder_weight[:, main_idx] @ probe_dirs[c])
        typical_main_mass = float(main_contrib.abs().mean()) if main_fires.any() else 0.0

        # Aligned other latents (excluding the main one).
        aligned = (cos[c].abs() >= COS_ALIGN_THRESHOLD).clone()
        aligned[main_idx] = False
        aligned_idx = torch.where(aligned)[0]

        absorption_rate = _absorption_rate_for_latents(
            z_c, main_fires, aligned_idx, decoder_weight, probe_dirs[c], typical_main_mass
        )

        # Null control: same computation on random latent sets of matching
        # size, excluding the main latent, averaged over trials.
        null_rates = []
        pool = torch.tensor([i for i in range(n_latents) if i != main_idx])
        for _ in range(N_NULL_TRIALS):
            if len(aligned_idx) == 0 or len(pool) == 0:
                null_rates.append(0.0)
                continue
            rand_idx = pool[torch.randperm(len(pool))[: len(aligned_idx)]]
            null_rates.append(
                _absorption_rate_for_latents(
                    z_c, main_fires, rand_idx, decoder_weight, probe_dirs[c], typical_main_mass
                )
            )
        null_absorption_rate = float(torch.tensor(null_rates).mean())

        per_class.append(
            {
                "class": c,
                "main_latent": main_idx,
                "main_f1": f1,
                "main_miss_rate": miss_rate,
                "main_fire_rate_overall": main_fire_rate_overall,
                "is_hub_main": is_hub_main,
                "absorption_rate": absorption_rate,
                "null_absorption_rate": null_absorption_rate,
                "absorption_above_null": absorption_rate - null_absorption_rate,
                "n_aligned_latents": int(len(aligned_idx)),
            }
        )

    clean = [p for p in per_class if not p["is_hub_main"]]
    n_hub = len(per_class) - len(clean)
    mean = lambda rows, k: float(torch.tensor([p[k] for p in rows]).mean()) if rows else float("nan")
    return {
        "per_class": per_class,
        "n_hub_main_classes": n_hub,
        # Means over non-hub-main classes only, so a method that "avoids
        # absorption" merely by having F1-maximal hubs picked as main
        # latents everywhere doesn't get credit for it -- if n_hub_main
        # is high for a method, treat its absorption numbers as unreliable
        # regardless of value, and look at n_hub_main itself instead.
        "mean_absorption_rate": mean(clean, "absorption_rate"),
        "mean_null_absorption_rate": mean(clean, "null_absorption_rate"),
        "mean_absorption_above_null": mean(clean, "absorption_above_null"),
        "mean_main_f1": mean(clean, "main_f1"),
        "mean_main_miss_rate": mean(clean, "main_miss_rate"),
        # Unfiltered versions for reference / sanity checking.
        "mean_absorption_rate_all_classes": mean(per_class, "absorption_rate"),
        "mean_main_f1_all_classes": mean(per_class, "main_f1"),
    }


def make_probe_and_loaders(dataset_name, seed):
    torch.manual_seed(seed)
    dataset_cls, input_dim = DATASETS[dataset_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    probe_dirs = fit_class_probes(train_loader, input_dim)
    return input_dim, train_loader, test_loader, probe_dirs


def evaluate_trained_models(dataset_name, seed, trained_models, out_path=None):
    """trained_models: dict label -> model exposing (recon, z) forward and
    a .decoder Linear, already trained. Fits class probes once and scores
    every model against them -- reused so different training runs (e.g. a
    rho sweep, or an entirely different objective) share one evaluation
    path and stay comparable."""
    _, _, test_loader, probe_dirs = make_probe_and_loaders(dataset_name, seed)

    results = {}
    for label, model in trained_models.items():
        z, y = collect_latents(model, test_loader)
        results[label] = absorption_scores(model, z, y, probe_dirs)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: absorption (class-level adaptation of Chanin et al. 2024) ===")
    header = f"{'Model':28s} {'Absorption':>11s} {'vsNull':>8s} {'MainF1':>8s} {'MainMiss':>9s} {'HubMain':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:28s} {r['mean_absorption_rate']:11.4f} {r['mean_absorption_above_null']:8.4f} "
            f"{r['mean_main_f1']:8.3f} {r['mean_main_miss_rate']:9.3f} {r['n_hub_main_classes']:8d}"
        )
    return results


def run(dataset_name, seed, rho, rate_kl_lambdas, tau):
    """Original entry point: RateKL (rho, lambdas) vs TopK vs TunedL1."""
    input_dim, train_loader, test_loader, probe_dirs = make_probe_and_loaders(dataset_name, seed)
    print(f"[{dataset_name} seed={seed}] Fitting class probes...")  # (already fit above)

    trained_models = {}
    for lam in rate_kl_lambdas:
        label = f"RateKL_rho{rho}_lam{lam}"
        print(f"[{dataset_name} seed={seed}] Training {label}...")
        trained_models[label] = _EvalWrapper(train_rate_kl(train_loader, input_dim, rho=rho, lambda_kl=lam, tau=tau))

    matched_k = max(1, round(rho * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK (k={matched_k})...")
    trained_models["TopK"] = train_topk(train_loader, input_dim, matched_k)

    print(f"[{dataset_name} seed={seed}] Training TunedL1...")
    trained_models["TunedL1"] = train_l1(train_loader, input_dim)

    out_path = f"results/absorption/{dataset_name}_seed{seed}.json"
    results = {}
    for label, model in trained_models.items():
        z, y = collect_latents(model, test_loader)
        results[label] = absorption_scores(model, z, y, probe_dirs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: absorption (class-level adaptation of Chanin et al. 2024) ===")
    header = f"{'Model':24s} {'Absorption':>11s} {'vsNull':>8s} {'MainF1':>8s} {'MainMiss':>9s} {'HubMain':>8s}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:24s} {r['mean_absorption_rate']:11.4f} {r['mean_absorption_above_null']:8.4f} "
            f"{r['mean_main_f1']:8.3f} {r['mean_main_miss_rate']:9.3f} {r['n_hub_main_classes']:8d}"
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
