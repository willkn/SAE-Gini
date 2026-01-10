import os
import json
import matplotlib.pyplot as plt
import torch

# Load the loss logs from the pilot and baseline runs (assumes they were printed to stdout and captured in results/*.log)

def parse_log(path):
    losses = []
    with open(path, 'r') as f:
        for line in f:
            if 'Loss:' in line:
                # Split on the first occurrence of "Loss:" and capture the first numeric token
                after = line.split('Loss:', 1)[1]
                # The token may be followed by a comma, whitespace, or end‑of‑line
                token = after.split(',')[0].strip().split()[0]
                try:
                    losses.append(float(token))
                except ValueError:
                    # If conversion fails, skip the line (malformed entry)
                    pass
    return losses

# Find the most recent pilot and baseline logs
import glob
pilot_logs = sorted(glob.glob('results/run_*.log'), key=os.path.getmtime, reverse=True)
baseline_logs = sorted(glob.glob('results/baseline_run_*.log'), key=os.path.getmtime, reverse=True)

pilot_log = os.path.basename(pilot_logs[0]) if pilot_logs else None
baseline_log = os.path.basename(baseline_logs[0]) if baseline_logs else None

pilot_losses = parse_log(os.path.join('results', pilot_log)) if pilot_log else []
baseline_losses = parse_log(os.path.join('results', baseline_log)) if baseline_log else []


if not pilot_losses and not baseline_losses:
    print("WARNING: No loss data found in either pilot or baseline logs!")
elif not pilot_losses:
    print("WARNING: No pilot loss data found!")
elif not baseline_losses:
    print("WARNING: No baseline loss data found!")

plt.figure(figsize=(10,6))
if pilot_losses:
    plt.plot(pilot_losses, label='Pilot (Gini)', marker='o', markersize=4, linewidth=2)
if baseline_losses:
    plt.plot(baseline_losses, label='Baseline (L1)', marker='s', markersize=4, linewidth=2)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Comparison: Gini vs L1 Regularization', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join('results', 'loss_comparison.png')
plt.savefig(plot_path, dpi=150)

# Write a short JSON summary for the human to read
summary = {
    'pilot_final_loss': pilot_losses[-1] if pilot_losses else None,
    'baseline_final_loss': baseline_losses[-1] if baseline_losses else None,
    'plot_path': plot_path
}
with open(os.path.join('results', 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Evaluation complete. Plot saved to', plot_path)
print('Summary written to results/summary.json')
