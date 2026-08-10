import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
lr = 1e-3
epochs = 10

# Hyperparameters
LAMBDA_GINI = 0.2
K = 2 # ~97% sparsity

def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

class HybridAutoencoder(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k
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

    def forward(self, x, use_topk=True):
        z = self.encoder(x)
        
        if use_topk:
            # Apply Hard Top-K
            topk_z = torch.zeros_like(z)
            values, indices = torch.topk(z, self.k, dim=1)
            topk_z.scatter_(1, indices, values)
            z_for_decoder = topk_z
        else:
            z_for_decoder = z
            
        recon = self.decoder(z_for_decoder)
        return recon, z

def train_and_eval_hybrid(dataset_name="mnist"):
    print(f"Training Gini -> Top-k Hybrid on {dataset_name}...")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = HybridAutoencoder(k=K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass with Top-k masking
            recon, z = model(batch, use_topk=True)
            
            # Reconstruction Loss (only through top-k neurons)
            mse = criterion(recon, batch)
            
            # Gini Loss (through ALL latent gradients)
            # This guides the distribution toward inequality, 
            # effectively "voting" which neurons should get to be in the Top-k.
            gini = differentiable_gini(z).mean()
            
            loss = mse - LAMBDA_GINI * gini
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    mses = []
    with torch.no_grad():
        for b, _ in test_loader:
            b = b.to(device)
            r, _ = model(b, use_topk=True)
            mses.append(criterion(r, b).item())
            
    test_mse = np.mean(mses)
    print(f"Hybrid {dataset_name} MSE: {test_mse:.4f}")
    return test_mse

if __name__ == "__main__":
    results = {
        "mnist": train_and_eval_hybrid("mnist"),
        "fmnist": train_and_eval_hybrid("fmnist")
    }
    
    with open("results/hybrid_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
