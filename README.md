# Gini Over L1: Scale-Invariant Sparsity for Sparse Autoencoders

[![Status](https://img.shields.io/badge/Status-Working%20Paper-orange.svg)]()
[![Field](https://img.shields.io/badge/Field-Mechanistic%20Interpretability-blue.svg)]()
[![Stage](https://img.shields.io/badge/Stage-Experimental-red.svg)]()

> **"Sparsity should be treated as a property of the distribution, not the scale."**

This repository contains the experimental code and findings for the working paper **"Gini Over L1: Scale-Invariant Sparsity for Sparse Autoencoders"** (January 2026).

---

## ⚠️ Important Disclaimer
**This is a research repository, not a production library.** 
The code provided here consists of experimental scripts used to generate the results in the paper. It is intended for researchers in Mechanistic Interpretability looking to replicate our findings or build upon the Gini Optimization objective.

**Status:** This is a **working paper**. While benchmarks on MNIST, Fashion-MNIST, and TinyStories-1M are stable, scaling to larger transformer residual streams is currently in progress.

---

## 🔬 Research Abstract

Sparse Autoencoders (SAEs) are critical for decoding the internal representations of AI models. However, the industry-standard **L1 Regularization** introduces **Magnitude Starvation**: by penalizing activation scale, it forces features to "whisper," blurring the lines between signal and noise.

We propose **Differentiable Gini Optimization**, a scale-invariant alternative that targets distributional inequality rather than absolute magnitude. By optimizing for the Gini Index, we allow semantic features to **"shout"** (maintain high-fidelity magnitudes) even at extreme sparsity levels (>97%).

### Key Findings:
*   **Magnitude Decoupling:** Gini-optimized features exhibit up to 17x higher magnitudes than L1 counterparts at matched sparsity.
*   **6.3x Steering Impact:** Proves that Gini-optimized features are more causally significant for model steering.
*   **Gradient Persistence:** Distributional pressure prevents the "gradient collapse" common in L1 models on high-entropy datasets (e.g., Fashion-MNIST).

---

## 📊 Key Results

### Shouting vs. Whispering
The following table compares the Gini SAE against a tuned L1 baseline on the Fashion-MNIST benchmark ($N=1024$):

| Metric | L1 Baseline | **Gini SAE (Ours)** |
| :--- | :--- | :--- |
| **Relative Sparsity** | 56.3% | **96.8%** |
| **Peak Activation** | ~4.0 | **~70.0** |
| **Steering Impact** | 0.019 | **0.121 (6.3x Increase)** |
| **Monosemanticity (Kurtosis)** | 4.44 | **7.50** |

---

## 📂 Repository Structure

This repository is organized by experiment suite:
*   `/mnist_fmnist/`: Scripts for the vision benchmarks and the Pareto frontier analysis (Figure 1).
*   `/tinystories/`: Feature extraction scripts for the TinyStories-1M transformer residual stream.
*   `/core/`: The core `GiniLoss` implementation and the **Feature Revival** (Ghost Grads) logic.
*   `/analysis/`: Jupyter notebooks for generating activation histograms and Lorenz curves.

---

## 🛠️ Replication

To replicate the results from Section 5 of the paper:

1. **Environment:**
   ```bash
   pip install torch torchvision numpy matplotlib
   ```

2. **Run F-MNIST Experiment:**
   ```bash
   python mnist_fmnist/train_gini_sae.py --dataset fmnist --lambda 0.05
   ```

3. **Run Transformer Feature Extraction:**
   ```bash
   python tinystories/extract_features.py --layer 4 --expansion 8
   ```

---

## 📐 The Gini Objective

We employ a rank-based, differentiable formulation of the Gini coefficient:

$$G(z) = \frac{2 \sum_{i=1}^n i \cdot z_{(i)}}{n \sum_{i=1}^n z_{(i)}} - \frac{n+1}{n}$$

Our implementation utilizes `torch.sort` to maintain a valid gradient path, ensuring that the learning signal updates the magnitudes of features based on their relative rank in the activation distribution.

---

## ✉️ Contact & Collaboration

I am an undergraduate researcher interested in the intersection of **Mechanistic Interpretability** and **AI Safety**. I am actively looking for feedback on this work, particularly regarding the scaling of Gini Optimization to billion-parameter models.

*   **Author:** William Knott
*   **Feedback:** [williamknott00@gmail.com]

---

## 📜 Citation

If you find this research or the Gini-Optimization objective useful for your work, please cite it as a working paper:

```bibtex
@article{knott2026gini,
  title={Gini Over L1: Scale-Invariant Sparsity for Sparse Autoencoders},
  author={Knott, William},
  journal={Working Paper},
  year={2026},
  url={https://github.com/willkn/Gini-SAE}
}
```
```
