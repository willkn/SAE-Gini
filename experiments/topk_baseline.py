import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 256
epochs = 5
lr = 1e-3
K = 2  # Matches ~97% sparsity (2/64 = 0.031)

# --- Model Definitions ---
class TopKAutoencoder(nn.Module):
    def __init__(self, k):
        super().__init__()
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
        
        # Apply Top-K
        topk_z = torch.zeros_like(z)
        if self.k < z.shape[1]:
            values, indices = torch.topk(z, self.k, dim=1)
            topk_z.scatter_(1, indices, values)
        else:
            topk_z = z
            
        recon = self.decoder(topk_z)
        return recon, topk_z

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

# --- Training Top-K SAE ---
def train_topk():
    print(f"\n--- Training Top-K SAE (k={K}) ---")
    model = TopKAutoencoder(K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_mse = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_mse += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, MSE: {total_mse/len(train_loader):.4f}")
        
    return model

# --- Evaluation ---
def evaluate_features(ae_model):
    print(f"\n--- Evaluating Features (Top-K) ---")
    ae_model.eval()
    
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
    
    classifier = LinearClassifier(latent_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        classifier.train()
        for z_batch, y_batch in train_feat_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = classifier(z_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
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

if __name__ == "__main__":
    model = train_topk()
    acc = evaluate_features(model)
    
    with open("results/topk_evaluation.json", "w") as f:
        json.dump({"accuracy": acc, "k": K}, f, indent=2)
        
    print(f"Top-K (k={K}) Accuracy: {acc:.2f}%")
