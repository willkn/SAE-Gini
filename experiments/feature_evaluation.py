import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
ae_lr = 1e-3
clf_lr = 1e-3
ae_epochs = 5
clf_epochs = 10

# Matched Sparsity Lambdas (from sweep)
GINI_LAMBDA = 0.2
L1_LAMBDA = 0.05

# --- Model Definitions ---
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

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Gini Function ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

# --- Helper: Train AE ---
def train_ae(mode, lam):
    print(f"\n--- Training AE ({mode}, lambda={lam}) ---")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=ae_lr)
    criterion = nn.MSELoss()
    
    for epoch in range(ae_epochs):
        model.train()
        total_mse = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            mse = criterion(recon, batch)
            
            if mode == "gini":
                # Maximize Gini (Minimize -Gini)
                sparsity_loss = -differentiable_gini(z).mean()
            else:
                sparsity_loss = z.abs().mean()
                
            loss = mse + lam * sparsity_loss
            loss.backward()
            optimizer.step()
            total_mse += mse.item()
            
        print(f"Epoch {epoch+1}/{ae_epochs}, MSE: {total_mse/len(train_loader):.4f}")
        
    return model

# --- Helper: Evaluate Classification ---
def evaluate_features(ae_model, name):
    print(f"\n--- Evaluating Features ({name}) ---")
    ae_model.eval()
    
    # 1. Extract Latent Representations
    def extract_latents(loader):
        zs = []
        labels = []
        with torch.no_grad():
            for batch, y in loader:
                batch = batch.to(device)
                _, z = ae_model(batch)
                zs.append(z.cpu())
                labels.append(y)
        return torch.cat(zs), torch.cat(labels)

    z_train, y_train = extract_latents(train_loader)
    z_test, y_test = extract_latents(test_loader)
    
    train_feat_loader = DataLoader(TensorDataset(z_train, y_train), batch_size=batch_size, shuffle=True)
    test_feat_loader = DataLoader(TensorDataset(z_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 2. Train Linear Classifier
    classifier = LinearClassifier(latent_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=clf_lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(clf_epochs):
        classifier.train()
        for z_batch, y_batch in train_feat_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = classifier(z_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    # 3. Test
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for z_batch, y_batch in test_feat_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            outputs = classifier(z_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main ---
if __name__ == "__main__":
    # 1. Train Gini Model
    gini_ae = train_ae("gini", GINI_LAMBDA)
    gini_acc = evaluate_features(gini_ae, "Gini-Sparse")
    
    # 2. Train L1 Model
    l1_ae = train_ae("l1", L1_LAMBDA)
    l1_acc = evaluate_features(l1_ae, "L1-Sparse")
    
    # 3. Report
    results = {
        "gini_accuracy": gini_acc,
        "l1_accuracy": l1_acc,
        "gini_lambda": GINI_LAMBDA,
        "l1_lambda": L1_LAMBDA
    }
    
    with open("results/feature_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Evaluation complete!")
    print(f"Gini Accuracy: {gini_acc:.2f}%")
    print(f"L1 Accuracy: {l1_acc:.2f}%")
