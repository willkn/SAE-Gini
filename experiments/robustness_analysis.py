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

def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

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
        h = self.encoder(x)
        recon = self.decoder(h)
        return recon, h

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def train_sae(mode):
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    
    for epoch in range(5):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            if mode == "gini":
                loss = mse - 0.2 * differentiable_gini(z).mean()
            else:
                loss = mse + 0.05 * z.abs().mean()
            loss.backward()
            optimizer.step()
    return model

def train_probe(sae, train_loader):
    sae.eval()
    probe = nn.Linear(latent_dim, 10).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            with torch.no_grad():
                _, z = sae(batch)
            optimizer.zero_grad()
            outputs = probe(z)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return probe

def test_robustness(sae, probe, test_loader, epsilon):
    sae.eval()
    probe.eval()
    correct = 0
    adv_correct = 0
    criterion = nn.CrossEntropyLoss()
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        _, z = sae(data)
        output = probe(z)
        loss = criterion(output, target)
        
        sae.zero_grad()
        probe.zero_grad()
        loss.backward()
        
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        with torch.no_grad():
            _, z_orig = sae(data)
            output_orig = probe(z_orig)
            init_pred = output_orig.max(1, keepdim=True)[1]
            correct += init_pred.eq(target.view_as(init_pred)).sum().item()
            
            _, z_adv = sae(perturbed_data)
            output_adv = probe(z_adv)
            final_pred = output_adv.max(1, keepdim=True)[1]
            adv_correct += final_pred.eq(target.view_as(final_pred)).sum().item()
            
    return correct / len(test_loader.dataset), adv_correct / len(test_loader.dataset)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)
    
    print("Training models...")
    gini_sae = train_sae("gini")
    l1_sae = train_sae("l1")
    
    print("Training probes...")
    gini_probe = train_probe(gini_sae, train_loader)
    l1_probe = train_probe(l1_sae, train_loader)
    
    epsilons = [0, 0.05, 0.1, 0.2]
    results = {"gini": {}, "l1": {}}
    
    for eps in epsilons:
        orig, adv = test_robustness(gini_sae, gini_probe, test_loader, eps)
        results["gini"][eps] = adv
        orig_l1, adv_l1 = test_robustness(l1_sae, l1_probe, test_loader, eps)
        results["l1"][eps] = adv_l1
        print(f"Eps {eps} | Gini: {adv:.4f} | L1: {adv_l1:.4f}")
        
    with open("results/robustness_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
