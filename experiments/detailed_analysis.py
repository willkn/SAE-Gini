import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import numpy as np

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

class SAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
    
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

def run_timing_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_in = 784
    d_latent = 1024
    batch_size = 256
    model = SAE(d_in, d_latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randn(batch_size, d_in).to(device)
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        recon, z = model(batch)
        loss = torch.nn.functional.mse_loss(batch, recon)
        loss.backward()
        optimizer.step()
    
    # Timing Gini
    start = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        recon, z = model(batch)
        mse = torch.nn.functional.mse_loss(batch, recon)
        gini = differentiable_gini(z).mean()
        loss = mse - 0.1 * gini
        loss.backward()
        optimizer.step()
    gini_time = (time.time() - start) / 100
    
    # Timing L1
    start = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        recon, z = model(batch)
        mse = torch.nn.functional.mse_loss(batch, recon)
        l1 = z.abs().mean()
        loss = mse + 0.01 * l1
        loss.backward()
        optimizer.step()
    l1_time = (time.time() - start) / 100
    
    return {"gini_step_ms": gini_time * 1000, "l1_step_ms": l1_time * 1000}

def run_sensitivity_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_in = 784
    d_latent = 1024
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    l2_penalties = [0, 1e-6, 1e-5, 1e-4]
    results = []
    
    for gamma in l2_penalties:
        model = SAE(d_in, d_latent).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(3): # Short run for trend
            model.train()
            for batch, _ in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, z = model(batch)
                mse = criterion(recon, batch)
                gini = differentiable_gini(z).mean()
                l2 = (z**2).mean()
                loss = mse - 0.1 * gini + gamma * l2
                loss.backward()
                optimizer.step()
        
        model.eval()
        mses = []
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                mses.append(criterion(recon, batch).item())
        
        results.append({"gamma": gamma, "mse": np.mean(mses)})
        print(f"Gamma {gamma}: MSE={np.mean(mses):.4f}")
        
    return results

if __name__ == "__main__":
    timing = run_timing_test()
    sensitivity = run_sensitivity_test()
    final_output = {"timing": timing, "sensitivity": sensitivity}
    with open("results/further_analysis.json", "w") as f:
        json.dump(final_output, f, indent=2)
