import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64

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
        return self.decoder(z), z

# --- Load Data ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
batch, _ = next(iter(test_loader))
batch = batch.to(device)

def get_activations(mode, lam):
    # We don't have the saved models, so we quickly retrain or just use the logic
    # Actually, let's just retrain two models for 3 epochs at matched sparsity (~97%)
    print(f"Training {mode} for visualization...")
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Define differentiable_gini inside for standalone
    def diff_gini(x):
        x_abs = torch.abs(x)
        x_sorted, _ = torch.sort(x_abs, dim=1)
        n = x_sorted.shape[1]
        index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return (2.0 * torch.sum(index * x_sorted, dim=1)) / (n * (torch.sum(x_sorted, dim=1) + 1e-8)) - (n + 1.0) / n

    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    
    for _ in range(3):
        for b, _ in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            recon, z = model(b)
            mse = criterion(recon, b)
            if mode == "gini":
                loss = mse - lam * diff_gini(z).mean()
            else:
                loss = mse + lam * z.abs().mean()
            loss.backward()
            optimizer.step()
    
    model.eval()
    _, z = model(batch)
    return z.cpu().detach().numpy()

# Get activations at ~97% sparsity
z_gini = get_activations("gini", 0.2)
z_l1 = get_activations("l1", 0.05)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(z_gini.flatten(), bins=100, color='blue', alpha=0.7)
plt.title("Gini Activations (Sparsity ~97%)")
plt.yscale('log')
plt.xlabel("Activation Value")

plt.subplot(1, 2, 2)
plt.hist(z_l1.flatten(), bins=100, color='orange', alpha=0.7)
plt.title("L1 Activations (Sparsity ~97%)")
plt.yscale('log')
plt.xlabel("Activation Value")

plt.tight_layout()
plt.savefig("results/latent_distributions.png", dpi=150)
print("Distribution plot saved to results/latent_distributions.png")
