import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json

# --- Gini Formula ---
def differentiable_gini(x, epsilon=1e-8):
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini

# --- SAE Model ---
class SAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
    
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

# --- Activation Extraction ---
def get_activations(model, tokenizer, texts, layer_idx=4):
    activations = []
    def hook_fn(module, input, output):
        # GPTNeo residual stream is often a tuple, or just a tensor
        if isinstance(output, tuple):
            activations.append(output[0].detach().cpu())
        else:
            activations.append(output.detach().cpu())

    # In GPTNeo, blocks are in model.transformer.h
    hook = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    
    hook.remove()
    # activations shape: [num_texts, batch=1, seq_len, hidden_size]
    # Flatten across seq_len
    all_acts = torch.cat(activations, dim=1).view(-1, model.config.hidden_size)
    return all_acts

# --- Main ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_id = "roneneldan/TinyStories-1M"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # TinyStories uses GPT2 tokenizer
    
    texts = [
        "Once upon a time there was a little girl called Lily.",
        "She lived in a small house near a big forest.",
        "One day she went for a walk and found a magic bird.",
        "The bird said hello and flew away.",
        "Lily was very happy and went home.",
        "The sun was shining and the birds were singing.",
        "He saw a big cat in the garden.",
        "The cat was black and white.",
        "It was a very nice day for a picnic.",
        "They ate apples and drank some water."
    ]
    
    print("Extracting activations...")
    acts = get_activations(model, tokenizer, texts)
    print(f"Total activation tokens: {acts.shape[0]}")
    
    d_in = model.config.hidden_size
    d_latent = d_in * 8 # 512 expansion
    
    # Train Gini SAE
    print("Training Gini SAE on transformer activations...")
    sae_gini = SAE(d_in, d_latent).to(device)
    optimizer = optim.Adam(sae_gini.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for i in range(500):
        # Mini-batches
        indices = torch.randperm(acts.shape[0])[:64]
        batch = acts[indices].to(device)
        
        optimizer.zero_grad()
        recon, z = sae_gini(batch)
        mse = criterion(recon, batch)
        gini_loss = -0.1 * differentiable_gini(z).mean()
        loss = mse + gini_loss
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i} | MSE: {mse.item():.4f} | Gini: {-gini_loss.item():.4f}")
            
    # Train L1 SAE for comparison
    print("Training L1 SAE on transformer activations...")
    sae_l1 = SAE(d_in, d_latent).to(device)
    optimizer = optim.Adam(sae_l1.parameters(), lr=1e-3)
    
    for i in range(500):
        indices = torch.randperm(acts.shape[0])[:64]
        batch = acts[indices].to(device)
        
        optimizer.zero_grad()
        recon, z = sae_l1(batch)
        mse = criterion(recon, batch)
        l1_loss = 0.01 * z.abs().mean()
        loss = mse + l1_loss
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i} | MSE: {mse.item():.4f} | L1: {l1_loss.item():.4f}")

    # Final Eval
    results = {}
    with torch.no_grad():
        batch = acts.to(device)
        
        # Gini
        recon, z = sae_gini(batch)
        mse_gini = criterion(recon, batch).item()
        sparsity_gini = (z < 0.01 * z.max(dim=1, keepdim=True)[0]).float().mean().item()
        
        # L1
        recon, z = sae_l1(batch)
        mse_l1 = criterion(recon, batch).item()
        sparsity_l1 = (z < 0.01 * z.max(dim=1, keepdim=True)[0]).float().mean().item()
        
    results = {
        "gini": {"mse": mse_gini, "sparsity": sparsity_gini},
        "l1": {"mse": mse_l1, "sparsity": sparsity_l1}
    }
    
    print("\nResults on Transformer Activations:")
    print(f"Gini: MSE={mse_gini:.4f}, RelSparsity={sparsity_gini:.4f}")
    print(f"L1:   MSE={mse_l1:.4f}, RelSparsity={sparsity_l1:.4f}")
    
    with open("results/transformer_sae_results.json", "w") as f:
        json.dump(results, f, indent=2)
