"""
Tests whether Gini's higher per-feature steering importance (see
steering_shrinkage_benchmark.py results) reflects genuinely monosemantic
features carrying their true signal (the "shrinkage" story), or a small
number of polysemantic bottleneck features absorbing disproportionate
reconstruction responsibility because Gini's training objective explicitly
rewards concentrating magnitude unevenly -- which would make "importance
went up" a near-tautological confirmation of the loss function, not
independent evidence of anything good happening.

Since every SAE variant here (SimpleSAE, GatedSAE, TopKSAE) has a plain
linear decoder, ablating feature f from a given latent code changes the
reconstruction by EXACTLY z_f * W_decoder[:, f] (no forward pass needed).
So for every feature we can compute, over the whole test set:

  - total importance = sum over active occurrences of |z_f| * ||W_decoder[:, f]||
    (exact per-occurrence ablation-impact norm, summed)
  - purity = 1 - H(y | feature f active) / log2(num_classes)
    (1.0 = fires for exactly one class; 0.0 = fires uniformly across all
    classes, i.e. carries no class-discriminative signal on its own)

Then for Gini vs. matched-sparsity TopK:
  - importance_gini_coefficient: is causal importance itself concentrated
    on a handful of features (bottleneck signature) or spread evenly?
  - top10_importance_share: fraction of total importance held by the most
    important 10% of features that are ever active
  - importance_purity_correlation: do the most important features tend to
    be class-pure (good story) or class-promiscuous (bad story)?
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import (
    DATASETS,
    latent_dim,
    batch_size,
    train_gini,
    train_topk,
    train_l1,
    train_gated,
    device,
)


def gini_coefficient(values):
    """Standard (non-differentiable, evaluation-only) Gini index of a
    nonnegative vector, for measuring inequality of per-feature importance."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v)) / (n * np.sum(v)) - (n + 1) / n)


def feature_importance_and_purity(model, loader, num_classes=10, min_active=5):
    """Exact per-feature ablation-impact (importance) and label-purity,
    computed once over the whole test set (no intervention loop needed)."""
    model.eval()
    decoder_weight = model.decoder.weight.detach()  # [d_in, d_latent]
    col_norms = decoder_weight.norm(dim=0)  # [d_latent], ||W[:, f]||

    d_latent = col_norms.shape[0]
    total_importance = torch.zeros(d_latent)
    class_counts = torch.zeros(d_latent, num_classes)
    active_counts = torch.zeros(d_latent)

    with torch.no_grad():
        for batch, y in loader:
            batch = batch.to(device)
            _, z = model(batch)
            z = z.cpu()
            active = z.abs() > 1e-6  # [B, d_latent]
            importance = z.abs() * col_norms.unsqueeze(0)  # [B, d_latent], exact ablation-impact norm
            total_importance += importance.sum(dim=0)
            active_counts += active.sum(dim=0).float()
            for c in range(num_classes):
                mask_c = (y == c).unsqueeze(1) & active
                class_counts[:, c] += mask_c.sum(dim=0).float()

    valid = active_counts >= min_active
    probs = class_counts / active_counts.clamp(min=1).unsqueeze(1)
    entropy = -(probs * torch.log2(probs.clamp(min=1e-12))).sum(dim=1)
    purity = 1.0 - entropy / np.log2(num_classes)

    return {
        "total_importance": total_importance,
        "purity": purity,
        "valid": valid,
        "active_counts": active_counts,
    }


def summarize(stats, label):
    valid = stats["valid"]
    importance = stats["total_importance"][valid].numpy()
    purity = stats["purity"][valid].numpy()

    if len(importance) < 2:
        return {"label": label, "n_valid_features": int(len(importance)), "note": "too few active features"}

    order = np.argsort(-importance)
    top10_n = max(1, len(importance) // 10)
    top10_share = float(importance[order[:top10_n]].sum() / importance.sum())

    corr = float(np.corrcoef(importance, purity)[0, 1]) if importance.std() > 0 and purity.std() > 0 else 0.0

    return {
        "label": label,
        "n_valid_features": int(len(importance)),
        "importance_gini_coefficient": gini_coefficient(importance),
        "top10_importance_share": top10_share,
        "importance_purity_correlation": corr,
        "mean_purity": float(purity.mean()),
        "mean_purity_top10pct_by_importance": float(purity[order[:top10_n]].mean()),
        "mean_purity_rest": float(purity[order[top10_n:]].mean()) if len(order) > top10_n else None,
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
    gini_stats = feature_importance_and_purity(gini_model, test_loader)
    results["GiniSAE"] = summarize(gini_stats, "GiniSAE")

    mean_active_per_sample = (gini_stats["active_counts"].sum() / len(test_ds)).item()
    matched_k = max(1, round(mean_active_per_sample))

    print(f"[{dataset_name} seed={seed}] Training Top-K SAE (k={matched_k})...")
    topk_model = train_topk(train_loader, input_dim, matched_k)
    topk_stats = feature_importance_and_purity(topk_model, test_loader)
    results["TopK"] = summarize(topk_stats, "TopK")

    print(f"[{dataset_name} seed={seed}] Training Tuned L1 SAE...")
    l1_model = train_l1(train_loader, input_dim)
    l1_stats = feature_importance_and_purity(l1_model, test_loader)
    results["TunedL1"] = summarize(l1_stats, "TunedL1")

    print(f"[{dataset_name} seed={seed}] Training Gated SAE...")
    gated_model = train_gated(train_loader, input_dim)
    gated_stats = feature_importance_and_purity(gated_model, test_loader)
    results["GatedSAE"] = summarize(gated_stats, "GatedSAE")

    out_path = f"results/feature_purity/{dataset_name}_seed{seed}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {dataset_name} seed={seed}: importance concentration & purity ===")
    header = f"{'Model':10s} {'ImpGini':>8s} {'Top10%Share':>12s} {'Imp-PurityCorr':>15s} {'MeanPurity':>11s} {'Top10Purity':>12s} {'RestPurity':>11s}"
    print(header)
    for name, r in results.items():
        if "note" in r:
            print(f"{name:10s} {r['note']}")
            continue
        print(
            f"{name:10s} {r['importance_gini_coefficient']:8.3f} {r['top10_importance_share']:12.3f} "
            f"{r['importance_purity_correlation']:15.3f} {r['mean_purity']:11.3f} "
            f"{r['mean_purity_top10pct_by_importance']:12.3f} {(r['mean_purity_rest'] or 0):11.3f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.seed)
