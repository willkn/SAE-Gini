"""
Tie-aware differentiable Gini index.

The original formulation (experiments/train_pilot_sparse.py,
experiments/tie_analysis.py) computes rank-based Gini by calling
torch.sort and using the 1..n integer positions in the sorted order as
weights. When many entries share the same value (as they do at the extreme
sparsity levels this project targets -- see results/tie_analysis.json,
where the average number of tied entries reaches ~59/64 by the end of
training), torch.sort still returns a valid gradient, but WHICH of the
tied entries gets rank 40 vs rank 55 is an arbitrary artifact of the sort
implementation. Two latent entries with identical values can end up with
different ranks purely by chance, and therefore receive different
gradients purely by chance -- see experiments/tie_gradient_analysis.py for
a direct measurement of this.

This module replaces the hard integer rank with a smooth, symmetric
"soft rank" computed from pairwise comparisons:

    rank_soft(x_i) = 1 + sum_{j != i} sigmoid((x_i - x_j) / tau)

For x_i > x_j this contributes ~1 (x_i ranked above x_j); for x_i < x_j it
contributes ~0; for x_i == x_j it contributes EXACTLY 0.5 regardless of i
or j, because sigmoid(0) = 0.5. So tied entries provably receive identical
soft ranks and therefore identical gradients (dGini/dz_i = dGini/dz_j
exactly whenever z_i = z_j), fixing the inconsistency by construction
rather than by empirical smoothing. As tau -> 0 this recovers the hard
rank (and hence the original torch.sort-based Gini) up to how ties are
broken, giving a one-parameter family that interpolates between the two.
"""
import torch


def differentiable_gini_hard(x, epsilon=1e-8):
    """Original torch.sort-based rank Gini (experiments/tie_analysis.py)."""
    x_abs = torch.abs(x)
    x_sorted, _ = torch.sort(x_abs, dim=1)
    n = x_sorted.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    sum_weighted = torch.sum(index * x_sorted, dim=1)
    sum_total = torch.sum(x_sorted, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini


def soft_rank(x, tau=0.01):
    """Smooth ascending rank (1..n) via pairwise sigmoid comparisons.
    x: [batch, n] -> returns [batch, n] soft ranks, differentiable, and
    exactly tie-symmetric (see module docstring).
    """
    # pairwise differences: diff[..., i, j] = x_i - x_j
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)  # [batch, n, n]
    comparisons = torch.sigmoid(diff / tau)  # ~1 if x_i > x_j, ~0.5 if tied
    # rank_i = 1 + sum_{j != i} comparisons[i, j]; the i==j term contributes
    # sigmoid(0)=0.5, so summing over all j and subtracting 0.5 is equivalent
    # and avoids an explicit diagonal mask.
    return comparisons.sum(dim=-1) - 0.5 + 1.0


def differentiable_gini_soft(x, tau=0.01, epsilon=1e-8):
    """Tie-aware Gini: same rank-weighted formula as differentiable_gini_hard,
    but using soft_rank in place of torch.sort's integer positions, and
    applied directly at each entry's original position (no sort needed)."""
    x_abs = torch.abs(x)
    n = x_abs.shape[1]
    ranks = soft_rank(x_abs, tau=tau)  # [batch, n], ~1..n
    sum_weighted = torch.sum(ranks * x_abs, dim=1)
    sum_total = torch.sum(x_abs, dim=1) + epsilon
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return gini
