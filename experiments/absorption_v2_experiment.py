"""
Follow-up to absorption_metric.py's first result, which failed the
pre-registered prediction (TopK ~0 absorption, RateKL nonzero) for three
separate reasons diagnosed after the fact -- see this script's three
fixes, each addressing one:

  1. TopK's ~0 absorption was largely a metric artifact: max-F1 latent
     selection picks always-on HUB latents as "main" when nothing else
     tracks a class well, and a latent that never goes silent can never
     register an absorption instance by this metric's definition. Fixed
     in absorption_metric.py: main_fire_rate_overall + is_hub_main flag,
     hub-main classes excluded from the headline means.

  2. RateKL's CIFAR-10 number (~0.54, ~= its own miss rate) indicated the
     compensation criterion was passing near-trivially on weak CIFAR pixel
     probes. Fixed in absorption_metric.py: null_absorption_rate control
     (same computation on random latent sets) -- absorption_above_null is
     the number to trust, not raw absorption_rate.

  3. RateKL's own two-sided KL(rho||p_hat_f) manufactures absorption
     pressure for any concept whose natural frequency exceeds rho (class
     base rate 0.10 > rho=0.09 forces ~10% of a perfectly-tracking latent's
     instances to be dropped). Two independent tests of this diagnosis:
       (a) rho=0.12 (above class frequency) with the ORIGINAL two-sided
           RateKL objective -- if the mechanism is right, absorption
           should drop sharply relative to rho=0.09.
       (b) the one-sided cap+budget objective (rate_cap_budget_sae.py),
           which removes the sub-rho pull entirely.

Trains all conditions for one (dataset, seed) and scores them against a
single set of class probes via absorption_metric.evaluate_trained_models,
so every model is compared on identical, freshly-fit probe directions.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from steering_shrinkage_benchmark import DATASETS, latent_dim, train_topk, train_l1
from rate_kl_sae import train_rate_kl, _EvalWrapper
from rate_cap_budget_sae import train_rate_cap_budget
from absorption_metric import make_probe_and_loaders, evaluate_trained_models


def run(dataset_name, seed, rho_low, rho_high, rho_max, tau):
    input_dim, train_loader, test_loader, probe_dirs = make_probe_and_loaders(dataset_name, seed)

    trained_models = {}

    label = f"RateKL_rho{rho_low}_two-sided"
    print(f"[{dataset_name} seed={seed}] Training {label} (baseline, reproduces first-run absorption)...")
    trained_models[label] = _EvalWrapper(train_rate_kl(train_loader, input_dim, rho=rho_low, lambda_kl=0.01, tau=tau))

    label = f"RateKL_rho{rho_high}_two-sided"
    print(f"[{dataset_name} seed={seed}] Training {label} (mechanism test: rho above class frequency)...")
    trained_models[label] = _EvalWrapper(train_rate_kl(train_loader, input_dim, rho=rho_high, lambda_kl=0.01, tau=tau))

    label = "RateCapBudget"
    print(f"[{dataset_name} seed={seed}] Training {label} (one-sided fix)...")
    trained_models[label] = _EvalWrapper(
        train_rate_cap_budget(train_loader, input_dim, rho_target=rho_low, rho_max=rho_max, lambda_cap=0.5, lambda_budget=0.5, tau=tau)
    )

    matched_k = max(1, round(rho_low * latent_dim))
    print(f"[{dataset_name} seed={seed}] Training TopK (k={matched_k})...")
    trained_models["TopK"] = train_topk(train_loader, input_dim, matched_k)

    print(f"[{dataset_name} seed={seed}] Training TunedL1...")
    trained_models["TunedL1"] = train_l1(train_loader, input_dim)

    out_path = f"results/absorption_v2/{dataset_name}_seed{seed}.json"
    return evaluate_trained_models(dataset_name, seed, trained_models, out_path=out_path)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="fashion_mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho-low", type=float, default=0.09, help="original rho, below class base rate (0.10)")
    parser.add_argument("--rho-high", type=float, default=0.12, help="rho above class base rate, mechanism test")
    parser.add_argument("--rho-max", type=float, default=0.15, help="cap threshold for RateCapBudget")
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    os.makedirs("results/absorption_v2", exist_ok=True)
    run(args.dataset, args.seed, args.rho_low, args.rho_high, args.rho_max, args.tau)
