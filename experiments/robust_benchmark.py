import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import json
import os

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
ae_lr = 1e-3
clf_lr = 1e-3
ae_epochs = 10  # More epochs for robustness
clf_epochs = 10
seeds = [42, 43, 44]  # 3 seeds for statistical robustness

# Target Sparsity Settings (found from previous sweeps to be ~97%)
LAMBDAS = {
    "gini": 0.2,
    "l1": 0.05,
    "topk": 2 # k=2 -> 2/64 = 3.125% dense -> 96.875% sparse
}

# --- Model Definitions ---
class Autoencoder(nn.Module):
    def __init__(self, mode="gini", k=2):
        super().__init__()
        self.mode = mode
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

    def forward(self, x):
        z = self.encoder(x)
        
        if self.mode == "topk":
            topk_z = torch.zeros_like(z)
            values, indices = torch.topk(z, self.k, dim=1)
            topk_z.scatter_(1, indices, values)
            z = topk_z
            
        recon = self.decoder(z)
        return recon, z

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Metrics ---
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
    # Measured as percentage of activations below threshold per sample, averaged over batch
    return (z < threshold).float().mean().item()

# --- Workflow Functions ---
def get_data(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "fmnist":
        train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_run(dataset_name, mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader, test_loader = get_data(dataset_name)
    
    model = Autoencoder(mode=mode, k=LAMBDAS["topk"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ae_lr)
    criterion = nn.MSELoss()
    
    # 1. Train AE
    for epoch in range(ae_epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            if mode == "gini":
                loss = mse - LAMBDAS["gini"] * differentiable_gini(z).mean()
            elif mode == "l1":
                loss = mse + LAMBDAS["l1"] * z.abs().mean()
            else: # topk
                loss = mse
                
            loss.backward()
            optimizer.step()
            
    # 2. Extract Features & Evaluate
    model.eval()
    def extract(loader):
        zs, ys = [], []
        mses, sparsities = [], []
        with torch.no_grad():
            for b, y in loader:
                b = b.to(device)
                r, z = model(b)
                zs.append(z.cpu())
                ys.append(y)
                mses.append(criterion(r, b).item())
                sparsities.append(get_sparsity(z))
        return torch.cat(zs), torch.cat(ys), np.mean(mses), np.mean(sparsities)

    z_train, y_train, _, _ = extract(train_loader)
    z_test, y_test, test_mse, test_sparsity = extract(test_loader)
    
    # 3. Linear Probe
    clf = LinearClassifier(latent_dim).to(device)
    clf_opt = optim.Adam(clf.parameters(), lr=clf_lr)
    clf_crit = nn.CrossEntropyLoss()
    
    train_feat_loader = DataLoader(TensorDataset(z_train, y_train), batch_size=batch_size, shuffle=True)
    for _ in range(clf_epochs):
        clf.train()
        for zb, yb in train_feat_loader:
            zb, yb = zb.to(device), yb.to(device)
            clf_opt.zero_grad()
            loss = clf_crit(clf(zb), yb)
            loss.backward()
            clf_opt.step()
            
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for zb, yb in DataLoader(TensorDataset(z_test, y_test), batch_size=batch_size):
            zb, yb = zb.to(device), yb.to(device)
            outputs = clf(zb)
            _, pred = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            
    return {
        "accuracy": 100 * correct / total,
        "mse": test_mse,
        "sparsity": test_sparsity
    }

# --- Main Execution ---
if __name__ == "__main__":
    datasets_to_run = ["mnist", "fmnist"]
    modes = ["l1", "gini", "topk"]
    all_results = {}

    for ds in datasets_to_run:
        all_results[ds] = {}
        for mode in modes:
            print(f"\n🚀 Running {ds} | {mode}...")
            runs = []
            for seed in seeds:
                res = train_one_run(ds, mode, seed)
                runs.append(res)
                print(f"  Seed {seed}: Acc={res['accuracy']:.2f}, MSE={res['mse']:.4f}, Sparsity={res['sparsity']:.4f}")
            
            # Aggregate
            all_results[ds][mode] = {
                "acc_mean": np.mean([r['accuracy'] for r in runs]),
                "acc_std": np.std([r['accuracy'] for r in runs]),
                "mse_mean": np.mean([r['mse'] for r in runs]),
                "mse_std": np.std([r['mse'] for r in runs]),
                "sparsity_mean": np.mean([r['sparsity'] for r in runs]),
                "sparsity_std": np.std([r['sparsity'] for r in runs])
            }

    with open("results/robust_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ Robust Benchmarking Complete!")
