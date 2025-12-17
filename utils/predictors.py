"""
Eigenvector prediction strategies for spectral annealing.

This module contains predictor functions used for warm-starting
eigenvalue computations in the spectral annealing algorithm.
"""

import cupy as cp


def diagonal_predictor(y_last, d_vec, delta_alpha, dtype):
    """
    Diagonal predictor: y_pred = D^{Δα/2} y_last / ||...||
    
    Replaces the trivial warm-start from the original code.
    
    Args:
        y_last: Previous eigenvector (cupy array)
        d_vec: Degree vector (cupy array)
        delta_alpha: Change in alpha parameter (float)
        dtype: Data type for computations
        
    Returns:
        Predicted eigenvector (normalized cupy array)
    """
    w = cp.power(cp.maximum(d_vec, cp.array(1e-12, dtype=dtype)), delta_alpha / 2.0)
    y = w * y_last
    norm = cp.linalg.norm(y)
    if norm < 1e-12:
        # Fallback to random initialization
        z = cp.random.randn(*y.shape).astype(dtype)
        return z / cp.linalg.norm(z)
    return y / norm


def secant_predictor(y_last, y_prev, dtype, gamma=1.0):
    """
    Secant predictor: y_pred = y_last + γ(y_last - y_prev)
    
    Uses a secant extrapolation based on the previous two eigenvectors.
    The damping parameter γ controls the contribution of the secant term.
    
    Args:
        y_last: Most recent eigenvector (cupy array)
        y_prev: Previous eigenvector (cupy array or None)
        dtype: Data type for computations
        gamma: Damping parameter in (0, 2), default is 1.0
        
    Returns:
        Predicted eigenvector (normalized cupy array)
    """
    if y_prev is None:
        return y_last
    # Sign alignment
    dot_prod = cp.dot(y_last, y_prev)
    if dot_prod < 0:
        y_prev = -y_prev
    s = y_last - y_prev
    s_norm = cp.linalg.norm(s)
    if s_norm < 1e-12:
        return y_last
    y_pred = y_last + gamma * s
    norm = cp.linalg.norm(y_pred)
    if norm < 1e-12:
        return y_last
    return y_pred / norm


def hybrid_predictor(y_last, y_prev, d_vec, delta_alpha, dtype):
    """
    Hybrid predictor combining diagonal and secant methods.
    
    Combines the diagonal predictor with secant extrapolation.
    
    Args:
        y_last: Most recent eigenvector (cupy array)
        y_prev: Previous eigenvector (cupy array or None)
        d_vec: Degree vector (cupy array)
        delta_alpha: Change in alpha parameter (float)
        dtype: Data type for computations
        
    Returns:
        Predicted eigenvector (normalized cupy array)
    """
    base = diagonal_predictor(y_last, d_vec, delta_alpha, dtype)
    if y_prev is None:
        return base

    dot_prod = cp.dot(y_last, y_prev)
    if dot_prod < 0:
        y_prev = -y_prev

    s = y_last - y_prev
    s_norm = cp.linalg.norm(s)
    if s_norm < 1e-12:
        return base

    gamma = 0.5
    y = base + gamma * s
    return y / cp.linalg.norm(y)


def subspace_predictor(y_last, y_prev, N_op, dtype):
    """
    Subspace predictor using Rayleigh-Ritz on span{y_last, y_prev}.
    
    Projects the operator onto the subspace spanned by the previous
    two eigenvectors and finds the dominant eigenvector in that subspace.
    
    Args:
        y_last: Most recent eigenvector (cupy array)
        y_prev: Previous eigenvector (cupy array or None)
        N_op: Linear operator (must have __call__ method implementing matvec)
        dtype: Data type for computations
        
    Returns:
        Predicted eigenvector (normalized cupy array)
    """
    if y_prev is None:
        return y_last

    dot_prod = cp.dot(y_last, y_prev)
    if dot_prod < 0:
        y_prev = -y_prev

    u1 = y_last
    u1_dot_prev = cp.dot(y_prev, u1)
    u2 = y_prev - u1_dot_prev * u1
    n2 = cp.linalg.norm(u2)
    if n2 < 1e-12:
        return u1

    u2 = u2 / n2
    # Stack vectors for projection
    U = cp.stack([u1, u2], axis=1)

    # Project N_op onto the subspace
    w1 = N_op(u1)
    w2 = N_op(u2)

    # Build 2x2 matrix H = U^T N U
    H = cp.array([
        [cp.dot(u1, w1), cp.dot(u1, w2)],
        [cp.dot(u2, w1), cp.dot(u2, w2)]
    ], dtype=dtype)

    # Eigenvalue decomposition of 2x2 matrix
    evals, vecs = cp.linalg.eigh(H)
    top_idx = cp.argmax(evals)
    top_vec = vecs[:, top_idx]
    y = U @ top_vec
    return y / cp.linalg.norm(y)

