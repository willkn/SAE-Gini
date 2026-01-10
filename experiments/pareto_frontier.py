import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import json

# --- Gini Formula ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

# --- SAE Model ---
class SAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
    
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

def train_and_eval(mode, lam, device, train_loader, test_loader, d_in, d_latent):
    model = SAE(d_in, d_latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Unit norm decoder constraint
    def normalize_decoder():
        with torch.no_grad():
            model.decoder.weight.data /= (model.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)

    for epoch in range(10):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            if mode == "gini":
                reg = -lam * differentiable_gini(z).mean()
            else:
                reg = lam * z.abs().mean()
            loss = mse + reg
            loss.backward()
            optimizer.step()
            normalize_decoder()
            
    model.eval()
    total_mse = 0
    total_l0 = 0
    total_effective_l0 = 0 # < 1e-3
    total_samples = 0
    
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            total_mse += criterion(recon, batch).item() * batch.size(0)
            total_l0 += (z > 0).float().sum().item()
            total_effective_l0 += (z > 1e-3).float().sum().item()
            total_samples += batch.size(0)
            
    return total_mse / total_samples, total_l0 / (total_samples * d_latent), total_effective_l0 / (total_samples * d_latent)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_in = 784
    d_latent = 1024
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    gini_lambdas = [0.01, 0.05, 0.1, 0.2, 0.4]
    l1_lambdas = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    results = {"gini": [], "l1": []}
    
    print("Sweeping Gini...")
    for lam in gini_lambdas:
        mse, l0, eff_l0 = train_and_eval("gini", lam, device, train_loader, test_loader, d_in, d_latent)
        results["gini"].append({"lambda": lam, "mse": mse, "l0": l0, "eff_l0": eff_l0})
        print(f"Gini lambda {lam}: MSE={mse:.4f}, L0={l0:.4f}, EffL0={eff_l0:.4f}")
        
    print("Sweeping L1...")
    for lam in l1_lambdas:
        mse, l0, eff_l0 = train_and_eval("l1", lam, device, train_loader, test_loader, d_in, d_latent)
        results["l1"].append({"lambda": lam, "mse": mse, "l0": l0, "eff_l0": eff_l0})
        print(f"L1 lambda {lam}: MSE={mse:.4f}, L0={l0:.4f}, EffL0={eff_l0:.4f}")
        
    with open("results/pareto_frontier.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    g_mse = [r["mse"] for r in results["gini"]]
    g_l0 = [r["eff_l0"] for r in results["gini"]] # Use effective L0 for Gini plot
    plt.plot(g_l0, g_mse, 'o-', label="Gini (Effective L0 < 1e-3)")
    
    l_mse = [r["mse"] for r in results["l1"]]
    l_l0 = [r["l0"] for r in results["l1"]]
    plt.plot(l_l0, l_mse, 's-', label="L1 (Actual L0)")
    
    plt.xlabel("Sparsity (L0 or Effective L0)")
    plt.ylabel("MSE (Reconstruction Error)")
    plt.title("Pareto Frontier: MSE vs. Sparsity")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/pareto_frontier_new.png")
    print("Created results/pareto_frontier_new.png")
