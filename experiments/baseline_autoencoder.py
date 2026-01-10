# Baseline Autoencoder with L1 regularization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple synthetic dataset (same as pilot)
X = torch.randn(1000, 20)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        return self.decoder(z)

model = SimpleAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# L1 regularization on encoder weights
l1_lambda = 1e-4

for epoch in range(5):
    epoch_loss = 0.0
    for batch, in loader:
        optimizer.zero_grad()
        recon = model(batch)
        mse = criterion(recon, batch)
        l1_norm = sum(p.abs().sum() for p in model.encoder.parameters())
        loss = mse + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/5] Loss: {epoch_loss/len(loader):.4f}")

# Save a small checkpoint (optional)
torch.save(model.state_dict(), "baseline_autoencoder.pt")
