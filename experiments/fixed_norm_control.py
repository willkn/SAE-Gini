import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
lr = 1e-3
epochs = 10

# Target sparsity ~97%
LAMBDA_GINI = 0.2
LAMBDA_L1 = 0.05

def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

class Autoencoder(nn.Module):
    def __init__(self, mode="gini", fixed_norm=False):
        super().__init__()
        self.mode = mode
        self.fixed_norm = fixed_norm
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
        
        if self.fixed_norm:
            # Rescale each sample to have a fixed average magnitude per active neuron
            # mimicking Gini's lack of magnitude penalty
            z_norm = torch.norm(z, p=2, dim=1, keepdim=True) + 1e-8
            z = z / z_norm * 5.0 # 5.0 is a typical scale for Gini activations found in previous viz
            
        recon = self.decoder(z)
        return recon, z

def train_and_eval(mode, fixed_norm):
    print(f"Training {mode} (Fixed Norm: {fixed_norm})...")
    model = Autoencoder(mode=mode, fixed_norm=fixed_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            if mode == "gini":
                loss = mse - LAMBDA_GINI * differentiable_gini(z).mean()
            else:
                loss = mse + LAMBDA_L1 * z.abs().mean()
                
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    mses = []
    sparsities = []
    with torch.no_grad():
        for b, _ in test_loader:
            b = b.to(device)
            r, z = model(b)
            mses.append(criterion(r, b).item())
            sparsities.append((z < 0.01).float().mean().item())
            
    return np.mean(mses), np.mean(sparsities)

if __name__ == "__main__":
    # Gini vs L1 vs Fixed-Norm L1
    results = {}
    
    # 1. Gini
    mse, sp = train_and_eval("gini", False)
    results["gini"] = {"mse": mse, "sparsity": sp}
    
    # 2. Standard L1
    mse, sp = train_and_eval("l1", False)
    results["l1"] = {"mse": mse, "sparsity": sp}
    
    # 3. Fixed-Norm L1
    mse, sp = train_and_eval("l1", True)
    results["l1_fixed_norm"] = {"mse": mse, "sparsity": sp}
    
    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: MSE={v['mse']:.4f}, Sparsity={v['sparsity']:.4f}")
        
    with open("results/fixed_norm_control.json", "w") as f:
        json.dump(results, f, indent=2)
