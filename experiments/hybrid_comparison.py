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
K = 2

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
    def __init__(self, mode="hybrid"):
        super().__init__()
        self.mode = mode
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
        topk_z = torch.zeros_like(z)
        values, indices = torch.topk(z, K, dim=1)
        topk_z.scatter_(1, indices, values)
        recon = self.decoder(topk_z)
        return recon, z, topk_z

def train_model(mode):
    model = Autoencoder(mode=mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z, topk_z = model(batch)
            mse = criterion(recon, batch)
            if mode == "hybrid":
                loss = mse - 0.2 * differentiable_gini(z).mean()
            else:
                loss = mse # Pure Top-k
            loss.backward()
            optimizer.step()
    return model

def eval_accuracy(sae, train_loader, test_loader):
    sae.eval()
    probe = nn.Linear(latent_dim, 10).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            with torch.no_grad():
                _, _, topk_z = sae(batch)
            optimizer.zero_grad()
            outputs = probe(topk_z)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    correct = 0
    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            _, _, topk_z = sae(batch)
            outputs = probe(topk_z)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(test_loader.dataset)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)
    
    print("Training Hybrid...")
    hybrid_sae = train_model("hybrid")
    hybrid_acc = eval_accuracy(hybrid_sae, train_loader, test_loader)
    
    print("Training Pure Top-k...")
    topk_sae = train_model("topk")
    topk_acc = eval_accuracy(topk_sae, train_loader, test_loader)
    
    print(f"Hybrid Accuracy: {hybrid_acc:.4f} | Pure Top-k Accuracy: {topk_acc:.4f}")
    
    results = {"hybrid_acc": hybrid_acc, "topk_acc": topk_acc}
    with open("results/hybrid_vs_topk.json", "w") as f:
        json.dump(results, f, indent=2)
