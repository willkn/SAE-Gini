import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

# --- Setup & Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 128
epochs = 5
lr = 1e-3

# Fixed coefficients for fair comparison (may need tuning)
lambda_gini = 0.5  # Weight for Gini maximisation
lambda_l1 = 1e-4   # Weight for L1 regularization

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Model Architecture ---
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

# --- Utility Functions ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

def get_sparsity(z, threshold=0.01):
    # Percentage of activations below threshold
    return (z < threshold).float().mean().item()

# --- Training / Evaluation Loops ---
def run_benchmark(mode="gini"):
    print(f"\nRunning Benchmark: {mode.upper()}...")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    logs = {"mse": [], "sparsity": [], "gini": []}
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_mse = 0
        epoch_sparsity = 0
        epoch_gini = 0
        
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            if mode == "gini":
                # Maximizing Gini (Sparsity)
                current_gini = differentiable_gini(z).mean()
                loss = mse - lambda_gini * current_gini
            elif mode == "l1":
                # L1 Sparsity
                l1_penalty = z.abs().mean()
                loss = mse + lambda_l1 * l1_penalty
            else:
                loss = mse
                
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_mse += mse.item()
            epoch_sparsity += get_sparsity(z)
            epoch_gini += differentiable_gini(z).mean().item()
            
        avg_mse = epoch_mse / len(train_loader)
        avg_sparsity = epoch_sparsity / len(train_loader)
        avg_gini = epoch_gini / len(train_loader)
        
        logs["mse"].append(avg_mse)
        logs["sparsity"].append(avg_sparsity)
        logs["gini"].append(avg_gini)
        
        print(f"Epoch {epoch+1}/{epochs} | MSE: {avg_mse:.4f} | Sparsity: {avg_sparsity:.2%} | Gini: {avg_gini:.4f}")
        
    duration = time.time() - start_time
    print(f"{mode.upper()} Complete in {duration:.2f}s")
    return model, logs

# --- Execution ---
gini_model, gini_logs = run_benchmark(mode="gini")
l1_model, l1_logs = run_benchmark(mode="l1")

# --- Plotting Results ---
plt.figure(figsize=(15, 5))

# MSE Comparison
plt.subplot(1, 3, 1)
plt.plot(gini_logs["mse"], label="Gini (Maximized)", marker='o')
plt.plot(l1_logs["mse"], label="L1 (Minimized)", marker='s')
plt.title("Reconstruction MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)

# Sparsity Comparison
plt.subplot(1, 3, 2)
plt.plot(gini_logs["sparsity"], label="Gini (Maximized)", marker='o')
plt.plot(l1_logs["sparsity"], label="L1 (Minimized)", marker='s')
plt.title(f"Sparsity (% < 0.01)")
plt.xlabel("Epoch")
plt.ylabel("Sparsity Ratio")
plt.legend()
plt.grid(True, alpha=0.3)

# Gini Comparison
plt.subplot(1, 3, 3)
plt.plot(gini_logs["gini"], label="Gini (Maximized)", marker='o')
plt.plot(l1_logs["gini"], label="L1 (Minimized)", marker='s')
plt.title("Gini Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Gini Value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plot_path = "results/mnist_benchmark_comparison.png"
plt.savefig(plot_path, dpi=150)
print(f"\nBenchmark comparison saved to {plot_path}")

# --- Sample Reconstruction Visualisation ---
def save_recons(model, name):
    model.eval()
    with torch.no_grad():
        batch, _ = next(iter(test_loader))
        batch = batch.to(device)
        recon, _ = model(batch)
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axes[0, i].imshow(batch[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
        plt.suptitle(f"Reconstructions: {name}")
        plt.savefig(f"results/mnist_recons_{name}.png")

save_recons(gini_model, "gini")
save_recons(l1_model, "l1")
print(f"Sample reconstructions saved to results/mnist_recons_*.png")
