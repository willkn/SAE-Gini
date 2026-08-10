import torch
import torch.nn as nn
import time
import json

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 256
latent_dim = 64
batch_size = 1024  # Larger batch for better measurement
num_batches = 100

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

# --- Efficiency Test ---
def measure_time(name, func, data):
    # Warmup
    for _ in range(10):
        _ = func(data)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start = time.time()
    for _ in range(num_batches):
        _ = func(data)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_batches
    print(f"{name}: {avg_time*1000:.4f} ms per batch")
    return avg_time

if __name__ == "__main__":
    data = torch.randn(batch_size, latent_dim).to(device)
    
    # 1. L1
    l1_time = measure_time("L1 Regularization", lambda x: x.abs().mean(), data)
    
    # 2. Gini
    gini_time = measure_time("Differentiable Gini", lambda x: differentiable_gini(x).mean(), data)
    
    # 3. Top-K
    def topk_func(x):
        values, indices = torch.topk(x, 2, dim=1)
        z = torch.zeros_like(x)
        z.scatter_(1, indices, values)
        return z
    topk_time = measure_time("Top-K Operator (k=2)", topk_func, data)
    
    results = {
        "l1_ms": l1_time * 1000,
        "gini_ms": gini_time * 1000,
        "topk_ms": topk_time * 1000
    }
    
    with open("results/efficiency_audit.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEfficiency Audit Saved to results/efficiency_audit.json")
