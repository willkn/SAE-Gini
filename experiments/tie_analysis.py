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
epochs = 5

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

def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

def check_ties(z, tol=1e-6):
    # z: [batch, latent]
    # Check for ties in absolute values per sample
    z_abs = torch.abs(z)
    batch_size, n = z_abs.shape
    tie_counts = []
    for i in range(batch_size):
        row = z_abs[i]
        # Sort to make tie detection easier
        row_sorted, _ = torch.sort(row)
        diffs = torch.diff(row_sorted)
        num_ties = (diffs < tol).sum().item()
        tie_counts.append(num_ties)
    return np.mean(tie_counts), np.max(tie_counts)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    stats = []
    
    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            gini = differentiable_gini(z).mean()
            loss = mse - 0.2 * gini
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                avg_ties, max_ties = check_ties(z)
                stats.append({"avg_ties": float(avg_ties), "max_ties": int(max_ties)})
                
    final_avg = np.mean([s["avg_ties"] for s in stats])
    print(f"Average ties per sample: {final_avg:.2f} / {latent_dim}")
    
    with open("results/tie_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)
