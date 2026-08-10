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
latent_dim = 1024
batch_size = 256
lr = 1e-3
epochs = 10

class JumpReLU(nn.Module):
    def __init__(self, d_in, d_latent, threshold=0.1):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.threshold = nn.Parameter(torch.ones(d_latent) * threshold)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x):
        pre_act = self.encoder(x)
        # Heaviside mask
        mask = (pre_act > self.threshold).float()
        # Straight-through gradient: grad_out is passed to pre_act directly
        z = pre_act * mask + (pre_act - pre_act.detach())
        recon = self.decoder(z)
        return recon, z

def train_jumprelu(target_sparsity=0.9):
    print(f"Training JumpReLU for target sparsity {target_sparsity}...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = JumpReLU(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # λ is for the L0 penalty in JumpReLU
    lam = 0.001
    
    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            # L0 penalty: lambda * dead_count
            l0_loss = lam * (z > 0).float().mean()
            loss = mse + l0_loss
            loss.backward()
            optimizer.step()
            
    model.eval()
    mses = []
    dead_features = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            mses.append(criterion(recon, batch).item())
            dead_features.append((z == 0).float().mean().item())
            
    return np.mean(mses), np.mean(dead_features)

if __name__ == "__main__":
    mse, l0 = train_jumprelu(0.9)
    print(f"JumpReLU Result: MSE={mse:.4f}, Sparsity={l0:.4f}")
    with open("results/jumprelu_baseline.json", "w") as f:
        json.dump({"mse": mse, "sparsity": l0}, f)
