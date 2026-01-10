# experiments/train_pilot.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the autoencoder architecture
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_size),
            nn.ReLU()  # ReLU activation for positive encodings
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()  # Assuming input is normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Define the differentiable Gini coefficient approximation (FIXED VERSION)
def differentiable_gini(x, epsilon=1e-8):
    """
    Compute Gini coefficient using rank-based formula.
    Gini = 0 means perfect equality (all activations same)
    Gini → 1 means perfect inequality (one activation dominates)
    
    For sparsity, we WANT high Gini (unequal distribution = sparse)
    So we minimize NEGATIVE Gini, or just use Gini as-is in loss.
    """
    # Work with absolute values
    x_abs = torch.abs(x)  # Shape: (batch, features)
    
    # Sort along feature dimension
    x_sorted, _ = torch.sort(x_abs, dim=1)  # ascending order
    
    n = x_sorted.shape[1]  # number of features
    
    # Index ranks (1, 2, 3, ..., n)
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)  # Shape: (n,)
    
    # Gini formula: G = (2 * Σ(i * x_i)) / (n * Σ(x)) - (n+1)/n
    # where x_i are sorted values and i are ranks
    sum_weighted = torch.sum(index * x_sorted, dim=1)  # Σ(i * x_i) for each batch
    sum_total = torch.sum(x_sorted, dim=1) + epsilon  # Σ(x) for each batch, avoid div by 0
    
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    
    return gini  # Shape: (batch,)

# Generate synthetic data
input_size = 784  # Example: MNIST-like data
encoding_size = 32
batch_size = 64
num_epochs = 10

# Create a synthetic dataset
train_data = torch.rand((1000, input_size))
train_labels = torch.randint(0, 10, (1000,)) # Placeholder for labels if needed
test_data = torch.rand((200, input_size))
test_labels = torch.randint(0, 10, (200,)) # Placeholder for labels if needed

train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
model = SparseAutoencoder(input_size, encoding_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
reconstruction_loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        # Forward pass
        decoded, encoded = model(data)
        reconstruction_loss = reconstruction_loss_fn(decoded, data)

        # Sparsity loss using differentiable Gini coefficient
        gini_raw = differentiable_gini(encoded)  # raw Gini per sample
        gini_loss = torch.mean(gini_raw)  # mean over batch
        
        # INCREASED from 0.001 to 0.1 to make Gini contribution visible
        lambda_sparse = 0.1  # tunable hyperparameter to control sparsity
        total_loss = reconstruction_loss + lambda_sparse * gini_loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            # Enhanced logging with raw Gini values for debugging
            print(f'[DENSE] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {total_loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                  f'Gini Loss: {gini_loss.item():.4f}, Raw Gini (mean): {gini_raw.mean().item():.4f}, '
                  f'Weighted Gini: {(lambda_sparse * gini_loss).item():.4f}')

# Evaluation (basic example)
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for data, _ in test_loader:
        decoded, encoded = model(data)
        reconstruction_loss = reconstruction_loss_fn(decoded, data)
        gini_loss = torch.mean(differentiable_gini(encoded))
        lambda_sparse = 0.1  # Match training value
        total_loss = reconstruction_loss + lambda_sparse * gini_loss
        test_loss += total_loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
