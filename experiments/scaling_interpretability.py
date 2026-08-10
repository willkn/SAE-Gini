import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 512
latent_dims = [64, 256, 1024] # Scaling test
batch_size = 256
epochs = 5
lr = 1e-3

# --- Gini Function ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

# --- AE with Scaling and Monitoring ---
class Autoencoder(nn.Module):
    def __init__(self, l_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, l_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(l_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def run_scaling_and_viz(l_dim):
    print(f"\nRunning Scalability & Interpretability Test (Latent Dim = {l_dim})")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    
    model = Autoencoder(l_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    grad_norms = []
    
    for epoch in range(epochs):
        model.train()
        for i, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            gini = differentiable_gini(z).mean()
            
            # Maximize inequality (Minimize -Gini)
            loss = mse - 0.2 * gini
            loss.backward()
            
            # Track gradient norm of the encoder's last layer to check for sort instability
            with torch.no_grad():
                grad_norm = model.encoder[-2].weight.grad.norm().item()
                grad_norms.append(grad_norm)
            
            optimizer.step()
            
    # Visualize Interpretability Artifacts
    model.eval()
    with torch.no_grad():
        batch, _ = next(iter(train_loader))
        batch = batch.to(device)
        _, z = model(batch)
        
        # 1. Activation Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(z.cpu().flatten().numpy(), bins=100, density=True, color='purple', alpha=0.7)
        plt.yscale('log')
        plt.title(f"Activation Density (Log Scale) | Latent Dim = {l_dim}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"results/hist_dim_{l_dim}.png")
        plt.close()
        
        # 2. Filter Visualization (if dim is small or just first 64)
        if l_dim >= 64:
            # Look at weights of the first layer reshaped
            # We want to see if they look like "filters"
            weights = model.encoder[0].weight.data.cpu() # [512, 784]
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(weights[i].reshape(28, 28), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"Learned Filters (Latent Dim = {l_dim})")
            plt.savefig(f"results/filters_dim_{l_dim}.png")
            plt.close()

    return grad_norms

if __name__ == "__main__":
    scaling_data = {}
    plt.figure(figsize=(10, 6))
    for d in latent_dims:
        norms = run_scaling_and_viz(d)
        scaling_data[d] = {
            "mean_grad_norm": np.mean(norms),
            "std_grad_norm": np.std(norms)
        }
        plt.plot(norms[-200:], label=f"Dim {d}") # Plot last 200 steps for stability check
    
    plt.title("Gradient Stability (Last 200 Mini-batches)")
    plt.xlabel("Step")
    plt.ylabel("Grad Norm (Encoder Last Layer)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/gradient_stability.png")
    plt.close()

    # Save statistics
    with open("results/scaling_stability.json", "w") as f:
        json.dump(scaling_data, f, indent=2)
        
    print("\nScaling & Interpretability Test Complete!")
