import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import time

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 3072 # 32x32x3 for CIFAR-10
latent_dim = 2048 # Greater than input for SAE sparsity testing
batch_size = 128
lr = 5e-4
epochs = 15

# --- Gini Formula with Stability ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

# --- SAE with Stability and Feature Revival ---
class RobustSAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
        # Tie decoder weights to encoder transposed is a common trick, but let's keep them separate for now
        # to allow 'shouting' magnitudes. 
    
    def forward(self, x):
        z_pre = self.encoder(x)
        z = torch.relu(z_pre)
        # Apply latent normalization for stability (Addressing reviewer concern)
        # z = z / (z.norm(dim=1, keepdim=True) + 1e-8) # Optional: Hard norm
        recon = self.decoder(z)
        return recon, z

def train_robust_gini(dataset="cifar10"):
    print(f"Training Robust Gini SAE on {dataset}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = RobustSAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Feature Usage Tracking
    feature_counts = torch.zeros(latent_dim, device=device)
    
    for epoch in range(epochs):
        model.train()
        epoch_mse = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            # 1. Gini Sparsity
            gini = differentiable_gini(z).mean()
            
            # 2. Latent L2 Penalty (Numerical Stability)
            l2_reg = 1e-5 * z.pow(2).mean()
            
            # 3. Decoder Weight Norm (Prevent scale drift)
            # Standard SAE practice: keep decoder unit-norm
            # We'll allow scale but penalize extreme drift
            weight_reg = 1e-4 * model.decoder.weight.pow(2).mean()
            
            loss = mse - 0.2 * gini + l2_reg + weight_reg
            loss.backward()
            
            # Track feature usage
            with torch.no_grad():
                feature_counts += (z > 0).float().sum(dim=0)
            
            optimizer.step()
            epoch_mse += mse.item()
            
        # Feature Revival (Ghost Grads simplified)
        # If a feature hasn't fired in 1000 batches, we'll give it a tiny 'push' 
        # based on the residual of the batch.
        if (epoch + 1) % 5 == 0:
            dead_mask = (feature_counts == 0)
            num_dead = dead_mask.sum().item()
            if num_dead > 0:
                print(f"  [Epoch {epoch}] Reviving {num_dead} dead features...")
                # Simplified revival: re-initialize encoder weights for dead features to match current residuals
                # This is a crude approximation of resampling.
                with torch.no_grad():
                    # Get a batch for residuals
                    batch, _ = next(iter(train_loader))
                    batch = batch.to(device)
                    recon, _ = model(batch)
                    residual = batch - recon
                    
                    # Randomly sample residuals for dead neurons
                    for d_idx in torch.where(dead_mask)[0]:
                        sample_idx = torch.randint(0, batch.size(0), (1,)).item()
                        res_sample = residual[sample_idx]
                        model.encoder.weight[d_idx] = res_sample / (res_sample.norm() + 1e-8)
                        model.decoder.weight[:, d_idx] = res_sample / (res_sample.norm() + 1e-8)
                
            feature_counts.zero_()

        print(f"  Epoch {epoch} | MSE: {epoch_mse/len(train_loader):.4f}")

    # Final Eval
    model.eval()
    all_mses = []
    all_dead = []
    all_relative_sparsity = []
    
    with torch.no_grad():
        activated_mask = torch.zeros(latent_dim, device=device)
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon, z = model(batch)
            all_mses.append(criterion(recon, batch).item())
            
            activated_mask += (z > 0).float().sum(dim=0)
            
            max_vals = z.max(dim=1, keepdim=True)[0] + 1e-8
            all_relative_sparsity.append((z < 0.01 * max_vals).float().mean().item())
            
        dead_final = (activated_mask == 0).sum().item()
        
    res = {
        "mse": np.mean(all_mses),
        "dead_features": dead_final,
        "relative_sparsity": np.mean(all_relative_sparsity)
    }
    return res

if __name__ == "__main__":
    cifar_results = train_robust_gini("cifar10")
    print(f"\nCIFAR-10 Robust Gini Results:")
    print(json.dumps(cifar_results, indent=2))
    with open("results/robust_gini_cifar10.json", "w") as f:
        json.dump(cifar_results, f, indent=2)
