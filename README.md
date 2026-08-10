# Gini Over L1: Scale-Invariant Sparsity for Sparse Autoencoders

Experimental code and results for the working paper "Gini Over L1:
Scale-Invariant Sparsity for Sparse Autoencoders" (January 2026).

This is a research repository, not a production library: the code here is the
set of experimental scripts used to generate the paper's results, intended for
replication or extension of the Gini-optimization objective. Benchmarks on
MNIST, Fashion-MNIST, and TinyStories-1M are stable; scaling to larger
transformer residual streams is in progress.

## Abstract

Sparse Autoencoders (SAEs) are used to decode the internal representations of
neural networks. The standard L1 regularizer penalizes activation scale
directly, which causes magnitude starvation: features are pushed toward small
magnitudes to satisfy the penalty, blurring the distinction between signal and
noise.

We propose differentiable Gini optimization: a scale-invariant sparsity
objective that targets distributional inequality (the Gini index) rather than
absolute magnitude. This lets semantic features keep high-fidelity magnitudes
even at extreme sparsity (>97%).

Key findings:
- Gini-optimized features reach up to 17x higher magnitudes than L1
  counterparts at matched sparsity.
- Gini-optimized features are 6.3x more causally significant for model
  steering.
- Distributional pressure avoids the gradient collapse that L1 models show on
  high-entropy datasets (e.g. Fashion-MNIST).

## Results

Gini SAE vs. a tuned L1 baseline on Fashion-MNIST (N=1024):

| Metric | L1 baseline | Gini SAE |
| --- | --- | --- |
| Relative sparsity | 56.3% | 96.8% |
| Peak activation | ~4.0 | ~70.0 |
| Steering impact | 0.019 | 0.121 (6.3x) |
| Monosemanticity (kurtosis) | 4.44 | 7.50 |

## The Gini objective

A rank-based, differentiable formulation of the Gini coefficient:

$$G(z) = \frac{2 \sum_{i=1}^n i \cdot z_{(i)}}{n \sum_{i=1}^n z_{(i)}} - \frac{n+1}{n}$$

The implementation uses `torch.sort` to keep a valid gradient path, so the
learning signal updates feature magnitudes based on their relative rank in the
activation distribution.

## Contact

William Knott — williamknott00@gmail.com

Feedback welcome, particularly on scaling Gini optimization to
billion-parameter models.

## Citation

```bibtex
@article{knott2026gini,
  title={Gini Over L1: Scale-Invariant Sparsity for Sparse Autoencoders},
  author={Knott, William},
  journal={Working Paper},
  year={2026},
  url={https://github.com/willkn/SAE-Gini}
}
```
