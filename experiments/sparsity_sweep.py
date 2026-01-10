import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import json
import time

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256  # Larger batch for faster sweep
epochs = 3        # Fewer epochs for sweep speed
lr = 1e-3

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# --- Metrics ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

def get_sparsity(z, threshold=0.01):
    return (z < threshold).float().mean().item()

# --- Training Function ---
def train_model(mode, lam):
    print(f"Training {mode} with lambda={lam}...")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    final_mse = 0
    final_sparsity = 0
    
    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            if mode == "gini":
                loss = mse - lam * differentiable_gini(z).mean()
            else:
                loss = mse + lam * z.abs().mean()
                
            loss.backward()
            optimizer.step()
            
        # Record last epoch metrics
        with torch.no_grad():
            final_mse = mse.item()
            final_sparsity = get_sparsity(z)
            
    return final_mse, final_sparsity

# --- Sweep Execution ---
gini_lambdas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
l1_lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

results = {"gini": [], "l1": []}

for lam in gini_lambdas:
    mse, sparsity = train_model("gini", lam)
    results["gini"].append({"lambda": lam, "mse": mse, "sparsity": sparsity})

for lam in l1_lambdas:
    mse, sparsity = train_model("l1", lam)
    results["l1"].append({"lambda": lam, "mse": mse, "sparsity": sparsity})

# --- Save Results ---
with open("results/sweep_data.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot Pareto Frontier ---
plt.figure(figsize=(10, 7))

gini_mse = [r["mse"] for r in results["gini"]]
gini_sparsity = [r["sparsity"] for r in results["gini"]]
l1_mse = [r["mse"] for r in results["l1"]]
l1_sparsity = [r["sparsity"] for r in results["l1"]]

plt.scatter(gini_sparsity, gini_mse, label="Differentiable Gini", color="blue", s=100, marker='o')
plt.plot(gini_sparsity, gini_mse, color="blue", linestyle='--', alpha=0.5)

plt.scatter(l1_sparsity, l1_mse, label="L1 Baseline", color="orange", s=100, marker='s')
plt.plot(l1_sparsity, l1_mse, color="orange", linestyle='--', alpha=0.5)

plt.xlabel("Sparsity Ratio (L0)", fontsize=12)
plt.ylabel("Reconstruction MSE", fontsize=12)
plt.title("Sparsity-Reconstruction Tradeoff (MNIST)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/mnist_tradeoff_curve.png", dpi=150)

print("\n📈 Sweep complete! Tradeoff curve saved to results/mnist_tradeoff_curve.png")
