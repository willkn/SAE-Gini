import os
import json
import matplotlib.pyplot as plt
import glob

def parse_log(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if 'Loss:' in line and 'Batch' in line:
                try:
                    # Extract fields
                    # Format: [TYPE] Epoch [x/y], Batch [a/b], Loss: v, Reconstruction Loss: r, Gini Loss: g, ...
                    parts = line.split(',')
                    recon = float(parts[3].split(':')[1].strip())
                    gini = float(parts[4].split(':')[1].strip())
                    data.append({'recon': recon, 'gini': gini})
                except Exception:
                    pass
    return data

dense_logs = sorted(glob.glob('results/dense_run_*.log'), key=os.path.getmtime, reverse=True)
sparse_logs = sorted(glob.glob('results/sparse_run_*.log'), key=os.path.getmtime, reverse=True)

dense_data = parse_log(dense_logs[0]) if dense_logs else []
sparse_data = parse_log(sparse_logs[0]) if sparse_logs else []

plt.figure(figsize=(12, 5))

# Plot Reconstruction Loss
plt.subplot(1, 2, 1)
if dense_data:
    plt.plot([d['recon'] for d in dense_data], label='Minimizing Gini (Dense)', marker='o', markersize=2)
if sparse_data:
    plt.plot([d['recon'] for d in sparse_data], label='Maximizing Gini (Sparse)', marker='s', markersize=2)
plt.title('Reconstruction Loss (MSE)')
plt.xlabel('Batch/Step')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Gini Coefficient
plt.subplot(1, 2, 2)
if dense_data:
    plt.plot([d['gini'] for d in dense_data], label='Minimizing Gini (Dense)', marker='o', markersize=2)
if sparse_data:
    plt.plot([d['gini'] for d in sparse_data], label='Maximizing Gini (Sparse)', marker='s', markersize=2)
plt.title('Gini Coefficient (Inequality)')
plt.xlabel('Batch/Step')
plt.ylabel('Gini Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join('results', 'dense_vs_sparse_gini.png')
plt.savefig(plot_path, dpi=150)

summary = {
    'dense': {
        'final_recon': dense_data[-1]['recon'] if dense_data else None,
        'final_gini': dense_data[-1]['gini'] if dense_data else None,
    },
    'sparse': {
        'final_recon': sparse_data[-1]['recon'] if sparse_data else None,
        'final_gini': sparse_data[-1]['gini'] if sparse_data else None,
    },
    'plot_path': plot_path
}

with open(os.path.join('results', 'comparison_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Comparison complete. Plot saved to {plot_path}")
