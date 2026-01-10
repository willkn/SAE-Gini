import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
latent_dim = 1024

class simpleSAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
    def forward(self, x):
        z = torch.relu(self.encoder(x))
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

def calculate_feature_absorption(model, dataloader):
    model.eval()
    activations = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device).view(batch.size(0), -1)
            _, z = model(batch)
            activations.append(z.cpu())
    
    activations = torch.cat(activations, dim=0)
    corr = torch.corrcoef(activations.T.float())
    off_diag = corr - torch.diag(torch.diag(corr))
    mean_abs_corr = off_diag.abs().nanmean().item()
    
    kurtosis = ((activations - activations.mean(0))**4).mean(0) / (activations.std(0)**4 + 1e-8)
    avg_kurtosis = kurtosis.nanmean().item()
    
    return {"mean_abs_correlation": mean_abs_corr, "avg_kurtosis": avg_kurtosis}

def steering_test(model, dataloader):
    model.eval()
    batch, _ = next(iter(dataloader))
    batch = batch.to(device).view(batch.size(0), -1)
    
    with torch.no_grad():
        recon_orig, z = model(batch)
        sample_idx = 0
        active_features = torch.where(z[sample_idx] > 0)[0]
        if len(active_features) == 0:
            return {"steering_impact": 0.0}
            
        target_f = active_features[0]
        z_ablated = z.clone()
        z_ablated[sample_idx, target_f] = 0
        recon_ablated = model.decoder(z_ablated)
        
        diff = (recon_orig[sample_idx] - recon_ablated[sample_idx]).norm()
        impact = diff.item() / (recon_orig[sample_idx].norm().item() + 1e-8)
        
    return {"steering_impact": impact}

def quick_train(mode="gini"):
    m = simpleSAE(784, 1024).to(device)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    tr_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    for _ in range(2):
        for b, _ in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            r, z = m(b)
            mse = torch.nn.functional.mse_loss(r, b)
            if mode == "gini":
                loss = mse - 0.1 * differentiable_gini(z).mean()
            else:
                loss = mse + 0.01 * z.abs().mean()
            loss.backward()
            opt.step()
    return m

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    print("Benchmarking Interpretability...")
    gini_m = quick_train("gini")
    l1_m = quick_train("l1")
    
    gini_metrics = calculate_feature_absorption(gini_m, test_loader)
    gini_metrics.update(steering_test(gini_m, test_loader))
    
    l1_metrics = calculate_feature_absorption(l1_m, test_loader)
    l1_metrics.update(steering_test(l1_m, test_loader))
    
    results = {
        "gini": gini_metrics,
        "l1": l1_metrics
    }
    
    print(json.dumps(results, indent=2))
    with open("results/interpretability_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
