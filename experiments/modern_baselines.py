import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 1024  # Increased for modern standards
batch_size = 256
lr = 1e-3
epochs = 10

# --- Models ---

class GatedSAE(nn.Module):
    """
    Simplified Gated SAE based on Lieberum et al. 2024.
    Decouples the 'gate' (for sparsity) from the 'magnitude' (for reconstruction).
    """
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.gate_threshold = nn.Parameter(torch.zeros(d_latent))
        self.decoder = nn.Linear(d_latent, d_in)
        self.r_bias = nn.Parameter(torch.zeros(d_latent))

    def forward(self, x):
        pre_act = self.encoder(x)
        # Gate
        gate = torch.relu(pre_act - self.gate_threshold)
        # Magnitude (simplified: use the same pre_act but without the threshold bias for recon)
        # In a real Gated SAE, we often use a separate path or a jump
        z = (gate > 0).float() * (pre_act + self.r_bias)
        recon = self.decoder(z)
        return recon, z, gate

class JumpReLU(nn.Module):
    """
    JumpReLU uses a threshold. Gradient is handled via straight-through.
    """
    def __init__(self, d_in, d_latent, threshold=0.1):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.threshold = threshold
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        pre_act = self.encoder(x)
        mask = (pre_act > self.threshold).float()
        # Straight-through estimator
        z = pre_act * mask + (pre_act - pre_act.detach())
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

class GiniSAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

# --- Training and Evaluation ---

def get_dead_features(model, dataloader, device):
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            _, z, *rest = model(batch)
            all_z.append((z > 0).float().cpu())
    all_z = torch.cat(all_z, dim=0) # [N, latent_dim]
    activated = all_z.sum(dim=0) # [latent_dim]
    dead_count = (activated == 0).sum().item()
    return dead_count

def run_experiment(dataset_name="fashion"):
    print(f"\n🚀 Running Modern Benchmarks on {dataset_name}...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    if dataset_name == "fashion":
        train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    # 1. Gini SAE
    print("Training Gini SAE...")
    model = GiniSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) - 0.1 * differentiable_gini(z).mean()
            loss.backward()
            optimizer.step()
    
    dead_count = get_dead_features(model, test_loader, device)
    mse = 0
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            mse += criterion(recon, batch).item()
    mse /= len(test_loader)
    results["GiniSAE"] = {"mse": mse, "dead_features": dead_count}
    print(f"GiniSAE: MSE={mse:.4f}, Dead={dead_count}")

    # 2. Gated SAE
    print("Training Gated SAE...")
    model = GatedSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, gate = model(batch)
            # Loss: MSE + L1 on gate
            loss = criterion(recon, batch) + 0.05 * gate.abs().mean()
            loss.backward()
            optimizer.step()
            
    dead_count = get_dead_features(model, test_loader, device)
    mse = 0
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z, gate = model(batch)
            mse += criterion(recon, batch).item()
    mse /= len(test_loader)
    results["GatedSAE"] = {"mse": mse, "dead_features": dead_count}
    print(f"GatedSAE: MSE={mse:.4f}, Dead={dead_count}")

    # 3. Tuned L1 SAE
    print("Training Tuned L1 SAE...")
    model = GiniSAE(input_dim, latent_dim).to(device) # Reusing simple AE structure
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Using a more careful lambda and potentially a warmup if needed (skipped for speed, but tuned lambda)
    for epoch in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch) + 0.01 * z.abs().mean()
            loss.backward()
            optimizer.step()
            
    dead_count = get_dead_features(model, test_loader, device)
    mse = 0
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            mse += criterion(recon, batch).item()
    mse /= len(test_loader)
    results["TunedL1"] = {"mse": mse, "dead_features": dead_count}
    print(f"TunedL1: MSE={mse:.4f}, Dead={dead_count}")

    return results

# --- Sorting Benchmark ---
def benchmark_sort_scaling():
    print("\n⏱️ Benchmarking Sorting Overhead...")
    sizes = [1024, 4096, 16384, 32768, 65536, 131072]
    batch_size = 256
    results = {}
    for n in sizes:
        x = torch.randn(batch_size, n, device=device)
        start = time.time()
        for _ in range(10):
            torch.sort(x, dim=1)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = (time.time() - start) / 10 * 1000 # ms
        results[n] = elapsed
        print(f"Dim {n}: {elapsed:.2f} ms")
    return results

if __name__ == "__main__":
    sort_results = benchmark_sort_scaling()
    exp_results = run_experiment("fashion")
    
    final_output = {
        "sorting_overhead": sort_results,
        "modern_baselines": exp_results
    }
    
    with open("results/modern_baselines.json", "w") as f:
        json.dump(final_output, f, indent=2)
