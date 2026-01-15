#!/usr/bin/env python3
"""
DSN (Differentiable Spectral Normalization) Solver Wrapper

Integrates DSN solver with the Ising benchmark framework.
Compatible with main.py --solver DSN --dataset <dataset>

Based on IJCAI 2026 paper: Differentiable Spectral Normalization for Large-Scale Ising Optimization
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import logging
import time
from typing import Tuple, Union, Optional
from pathlib import Path
from datetime import datetime

# Configure Logger
logger = logging.getLogger("DSN")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =============================================================================
# Core DSN Solver (from dsn_solver_v3.py)
# =============================================================================

class DSNSolver:
    """
    Differentiable Spectral Normalization solver for MaxCut/Ising optimization.
    
    Key equations from the paper:
    - Normalized adjacency: N(W) = W^{-1/2} J W^{-1/2}
    - Spectral bound: B(W) = -1/2 * λ_max(N(W)) * Tr(W)
    - Hellmann-Feynman gradient: ∂B/∂w_i = (λ/2)[Tr(W)/w_i * u_i² - 1]
    """
    
    def __init__(
        self,
        coupling_matrix: Union[sp.spmatrix, np.ndarray],
        epsilon: float = 0.01,
        random_seed: Optional[int] = None,
        input_convention: str = 'auto'
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Convert to sparse CSR
        if sp.issparse(coupling_matrix):
            self.J_scipy = coupling_matrix.tocsr().astype(np.float64)
        else:
            self.J_scipy = sp.csr_matrix(coupling_matrix, dtype=np.float64)
        
        # Symmetrize and remove diagonal
        self.J_scipy = 0.5 * (self.J_scipy + self.J_scipy.T)
        self.J_scipy.setdiag(0)
        self.J_scipy.eliminate_zeros()
        
        self.n = self.J_scipy.shape[0]
        self.epsilon = epsilon
        
        # Handle sign convention
        if input_convention == 'ising':
            self.J_scipy = (-self.J_scipy).tocsr()
            logger.info("Input convention: Ising → MaxCut (flipped sign)")
        elif input_convention == 'maxcut':
            logger.info("Input convention: MaxCut (no sign change)")
        elif input_convention == 'auto':
            if self.J_scipy.nnz > 0 and np.median(self.J_scipy.data) < 0:
                self.J_scipy = (-self.J_scipy).tocsr()
                logger.warning("Auto-detected Ising convention, flipped sign.")
        
        self.W_scipy = self.J_scipy.copy()
        
        # Initialize weights
        self.w = np.ones(self.n, dtype=np.float64)
        self._cached_eigvec = None
        
        logger.info(f"DSN Solver: n={self.n}, edges={self.J_scipy.nnz//2}, ε={epsilon}")

    def _compute_leading_eigenpair(
        self, 
        w: np.ndarray, 
        tol: float = 1e-6, 
        maxiter: int = 300
    ) -> Tuple[float, np.ndarray]:
        """Compute leading eigenpair of N(W) = W^{-1/2} J W^{-1/2}"""
        w_inv_sqrt = 1.0 / np.sqrt(np.maximum(w, 1e-12))
        
        def matvec(v):
            u = v * w_inv_sqrt
            z = self.J_scipy.dot(u)
            return z * w_inv_sqrt
        
        N_op = LinearOperator(
            shape=(self.n, self.n),
            matvec=matvec,
            dtype=np.float64
        )
        
        v0 = self._cached_eigvec if self._cached_eigvec is not None else None
        
        eigenvalues, eigenvectors = eigsh(
            N_op, k=1, which='LA',
            tol=tol, maxiter=maxiter, v0=v0
        )
        
        lam = eigenvalues[0]
        u = eigenvectors[:, 0]
        u = u / (np.linalg.norm(u) + 1e-12)
        self._cached_eigvec = u.copy()
        
        return lam, u

    def compute_spectral_bound(self, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Compute B(W) = -1/2 * λ_max(N(W)) * Tr(W)"""
        lam, u = self._compute_leading_eigenpair(w)
        trace_W = np.sum(w)
        B = -0.5 * lam * trace_W
        return B, lam, u
    
    def compute_gradient(self, w: np.ndarray, lam: float, u: np.ndarray) -> np.ndarray:
        """Hellmann-Feynman gradient: g_i = (λ/2) * [Tr(W)/w_i * u_i² - 1]"""
        trace_W = np.sum(w)
        u_sq = u ** 2
        w_safe = np.maximum(w, 1e-12)
        grad = (lam / 2.0) * (trace_W * u_sq / w_safe - 1.0)
        return grad
    
    def project_to_domain(self, w: np.ndarray) -> np.ndarray:
        """Simple box projection: w_i ∈ [ε, ε⁻¹]"""
        return np.clip(w, self.epsilon, 1.0 / self.epsilon)
    
    def project_to_capped_simplex(self, w: np.ndarray, target_sum: float = None) -> np.ndarray:
        """
        Project onto intersection of box [ε, 1/ε]^n and hyperplane Σw_i = target_sum.
        Uses bisection to find the optimal shift τ.
        """
        if target_sum is None:
            target_sum = float(self.n)
        
        lo, hi = self.epsilon, 1.0 / self.epsilon
        
        def sum_after_shift(tau):
            return np.sum(np.clip(w - tau, lo, hi))
        
        tau_lo = np.min(w) - hi
        tau_hi = np.max(w) - lo
        
        sum_at_tau_lo = sum_after_shift(tau_lo)
        sum_at_tau_hi = sum_after_shift(tau_hi)
        
        if target_sum >= sum_at_tau_lo:
            return np.full_like(w, hi)
        if target_sum <= sum_at_tau_hi:
            return np.full_like(w, lo)
        
        for _ in range(50):
            tau_mid = (tau_lo + tau_hi) / 2
            sum_mid = sum_after_shift(tau_mid)
            
            if sum_mid > target_sum:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid
            
            if abs(sum_mid - target_sum) < 1e-12:
                break
        
        tau = (tau_lo + tau_hi) / 2
        return np.clip(w - tau, lo, hi)

    def normalize_trace(self, w: np.ndarray, target_trace: Optional[float] = None) -> np.ndarray:
        if target_trace is None:
            target_trace = self.n
        return w * (target_trace / np.sum(w))
    
    def get_cut_value(self, spins: np.ndarray) -> float:
        spins = spins.reshape(-1)
        interaction = float(spins.dot(self.W_scipy.dot(spins)))
        total_weight = float(self.W_scipy.sum())
        return 0.25 * (total_weight - interaction)
    
    def local_search_1flip(self, spins: np.ndarray, max_iter: int = None) -> Tuple[np.ndarray, float]:
        """
        1-flip local search: iteratively flip single vertices that improve cut.
        
        Args:
            spins: Initial spin configuration
            max_iter: Maximum iterations (default: 10 * n)
            
        Returns:
            improved_spins: Locally optimal configuration
            best_cut: Cut value after local search
        """
        if max_iter is None:
            max_iter = 10 * self.n
            
        spins = spins.copy().astype(np.float64)
        n = len(spins)
        
        # Precompute: Ws[i] = sum_j(W_ij * s_j)
        Ws = np.asarray(self.W_scipy.dot(spins)).flatten()
        
        best_cut = self.get_cut_value(spins)
        improved = True
        iteration = 0
        total_flips = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            # Compute flip gains: gain[i] = improvement if we flip spin i
            # Flipping s_i: new_cut - old_cut = 2 * s_i * Ws[i] / 4 = s_i * Ws[i] / 2
            gains = spins * Ws  # Positive = flipping improves cut
            
            # Find best flip
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain > 1e-9:  # Improvement found
                # Flip the spin
                old_spin = spins[best_idx]
                spins[best_idx] = -old_spin
                
                # Update Ws incrementally
                # Ws[j] changes by W[j, best_idx] * (new_s - old_s) = W[j, best_idx] * (-2 * old_spin)
                col = self.W_scipy.getcol(best_idx).toarray().flatten()
                Ws += col * (-2 * old_spin)
                
                best_cut += best_gain / 2  # Convert to actual cut change
                improved = True
                total_flips += 1
        
        return spins.astype(np.int8), best_cut
    
    def local_search_greedy(self, spins: np.ndarray, max_passes: int = 10) -> Tuple[np.ndarray, float]:
        """
        Greedy local search: multiple passes of 1-flip until convergence.
        """
        spins = spins.copy().astype(np.float64)
        best_cut = self.get_cut_value(spins)
        
        for pass_idx in range(max_passes):
            old_cut = best_cut
            spins, best_cut = self.local_search_1flip(spins, max_iter=self.n * 2)
            
            # Converged?
            if best_cut <= old_cut + 1e-9:
                break
        
        return spins.astype(np.int8), best_cut
    
    def round_eigenvector(self, u: np.ndarray, n_trials: int = 50, apply_local_search: bool = False) -> Tuple[np.ndarray, float]:
        """Multi-strategy rounding: sign, median, random hyperplane"""
        best_cut = -np.inf
        best_spins = None
        
        # Strategy 1: Sign rounding
        spins = np.sign(u)
        spins[spins == 0] = 1
        cut = self.get_cut_value(spins)
        if cut > best_cut:
            best_cut, best_spins = cut, spins.copy()
        
        # Strategy 2: Median rounding
        med = np.median(u)
        spins = np.sign(u - med)
        spins[spins == 0] = 1
        cut = self.get_cut_value(spins)
        if cut > best_cut:
            best_cut, best_spins = cut, spins.copy()
        
        # Strategy 3: Random hyperplane rounding
        u_std = np.std(u)
        for _ in range(n_trials):
            t = np.random.normal(0, u_std + 1e-6)
            spins = np.sign(u - t)
            spins[spins == 0] = 1
            cut = self.get_cut_value(spins)
            if cut > best_cut:
                best_cut, best_spins = cut, spins.copy()
        
        # Apply local search if requested
        if apply_local_search and best_spins is not None:
            best_spins, best_cut = self.local_search_greedy(best_spins)
        
        return best_spins, best_cut

    def solve(
        self,
        eta: float = 0.005,
        T: int = 300,
        init: str = 'identity',
        lr_decay: float = 0.998,
        patience: int = 20,
        grad_clip: float = 1.0,
        normalize_each_step: bool = True,
        verbose: bool = True,
        eval_frequency: int = 10,
        n_rounding: int = 50,
        local_search: bool = True,
        ls_frequency: int = 50
    ) -> Tuple[np.ndarray, float, float, dict]:
        """
        Main DSN optimization loop.
        
        Returns:
            best_spins: Best spin configuration found
            best_cut: Best cut value
            best_bound: Best spectral bound
            history: Training history dict
        """
        logger.info(f"DSN: T={T}, η={eta}, init={init}")
        t0 = time.time()
        
        # Initialize weights
        if init == 'degree':
            degrees = np.asarray(self.J_scipy.sum(axis=1)).reshape(-1) + 1e-6
            self.w = degrees / np.mean(degrees)
            self.w = self.project_to_capped_simplex(self.w, self.n)
        else:
            self.w = np.ones(self.n)
        
        self.w = self.normalize_trace(self.w)
        self.w = self.project_to_domain(self.w)
        self._cached_eigvec = None
        
        best_cut, best_spins = -np.inf, None
        best_bound, best_w = -np.inf, self.w.copy()
        no_improve_count = 0
        
        history = {'bound': [], 'eigenvalue': [], 'cut': [], 'time': []}
        current_eta = eta
        
        for t in range(T):
            B, lam, u = self.compute_spectral_bound(self.w)
            
            # Evaluate BEFORE update
            if t % eval_frequency == 0 or t == T - 1:
                # Apply local search periodically or at the end
                apply_ls = local_search and (t % ls_frequency == 0 or t == T - 1)
                spins, cut = self.round_eigenvector(u, n_trials=n_rounding, apply_local_search=apply_ls)
                
                if cut > best_cut:
                    best_cut = cut
                    best_spins = spins.copy()
                
                history['bound'].append(B)
                history['eigenvalue'].append(lam)
                history['cut'].append(cut)
                history['time'].append(time.time() - t0)
                
                if verbose and t % (eval_frequency * 4) == 0:
                    ls_marker = " [LS]" if apply_ls else ""
                    logger.info(f"t={t:03d}: B={B:.2f}, λ={lam:.4f}, Cut={cut:.0f}, Best={best_cut:.0f}{ls_marker}")
            
            # Track best bound with tolerance
            restored = False
            if B >= best_bound - 1e-9:
                if B > best_bound:
                    best_bound = B
                    best_w = self.w.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > patience:
                    self.w = best_w.copy()
                    self._cached_eigvec = None
                    current_eta *= 0.5
                    no_improve_count = 0
                    restored = True
                    if verbose:
                        logger.info(f"t={t}: Restored best w, η→{current_eta:.4f}")
            
            if restored:
                continue
            
            # Compute gradient
            grad = self.compute_gradient(self.w, lam, u)
            
            # Gradient clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > grad_clip * self.n:
                grad = grad * (grad_clip * self.n / grad_norm)
            
            # Gradient ascent + projection
            w_new = self.w + current_eta * grad
            
            if normalize_each_step:
                w_new = self.project_to_capped_simplex(w_new, target_sum=self.n)
            else:
                w_new = self.project_to_domain(w_new)
            
            self.w = w_new
            current_eta *= lr_decay
        
        # Final local search on best solution
        if local_search and best_spins is not None:
            cut_before_ls = best_cut
            best_spins, best_cut = self.local_search_greedy(best_spins, max_passes=20)
            if verbose and best_cut > cut_before_ls:
                logger.info(f"Final LS improved cut: {cut_before_ls:.0f} → {best_cut:.0f}")
        
        elapsed = time.time() - t0
        logger.info(f"DSN done in {elapsed:.2f}s | Cut={best_cut:.0f} | Bound={best_bound:.2f}")
        
        return best_spins, best_cut, best_bound, history


# =============================================================================
# GPU Version using CuPy (Optional)
# =============================================================================

class DSNSolverGPU:
    """GPU-accelerated DSN solver using CuPy."""
    
    def __init__(
        self,
        coupling_matrix: Union[sp.spmatrix, np.ndarray],
        epsilon: float = 0.01,
        random_seed: Optional[int] = None,
        input_convention: str = 'auto'
    ):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx_sparse
            import cupyx.scipy.sparse.linalg as cpx_splinalg
            self.cp = cp
            self.cpx_sparse = cpx_sparse
            self.cpx_splinalg = cpx_splinalg
        except ImportError:
            raise ImportError("CuPy required for GPU. Install: pip install cupy-cuda12x")
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Convert to sparse CSR
        if sp.issparse(coupling_matrix):
            J_scipy = coupling_matrix.tocsr().astype(np.float64)
        else:
            J_scipy = sp.csr_matrix(coupling_matrix, dtype=np.float64)
        
        J_scipy = 0.5 * (J_scipy + J_scipy.T)
        J_scipy.setdiag(0)
        J_scipy.eliminate_zeros()
        
        self.n = J_scipy.shape[0]
        self.epsilon = epsilon
        
        # Handle sign convention
        if input_convention == 'ising':
            J_scipy = (-J_scipy).tocsr()
        elif input_convention == 'auto':
            if J_scipy.nnz > 0 and np.median(J_scipy.data) < 0:
                J_scipy = (-J_scipy).tocsr()
        
        # Transfer to GPU
        self.J_gpu = cpx_sparse.csr_matrix(J_scipy.astype(cp.float32))
        self.W_scipy = J_scipy.copy()
        
        self.w = cp.ones(self.n, dtype=cp.float32)
        self._cached_eigvec = None
        
        logger.info(f"DSN GPU Solver: n={self.n}, edges={J_scipy.nnz//2}")

    def _compute_leading_eigenpair(self, w, tol=1e-5, maxiter=200):
        cp = self.cp
        w_inv_sqrt = 1.0 / cp.sqrt(cp.maximum(w, 1e-12))
        
        class NOperator(self.cpx_splinalg.LinearOperator):
            def __init__(inner_self, J_gpu, w_inv_sqrt):
                inner_self.J_gpu = J_gpu
                inner_self.w_inv_sqrt = w_inv_sqrt
                n = w_inv_sqrt.size
                super().__init__(dtype=cp.float32, shape=(n, n))
            
            def _matvec(inner_self, v):
                u = v * inner_self.w_inv_sqrt
                z = inner_self.J_gpu.dot(u)
                return z * inner_self.w_inv_sqrt
        
        N_op = NOperator(self.J_gpu, w_inv_sqrt)
        v0 = self._cached_eigvec if self._cached_eigvec is not None else None
        
        eigenvalues, eigenvectors = self.cpx_splinalg.eigsh(
            N_op, k=1, which='LA', tol=tol, maxiter=maxiter, v0=v0
        )
        
        lam = float(eigenvalues[0])
        u = eigenvectors[:, 0]
        u = u / (cp.linalg.norm(u) + 1e-12)
        self._cached_eigvec = u.copy()
        
        return lam, u

    def compute_spectral_bound(self, w):
        lam, u = self._compute_leading_eigenpair(w)
        trace_W = float(self.cp.sum(w))
        B = -0.5 * lam * trace_W
        return B, lam, u
    
    def compute_gradient(self, w, lam, u):
        cp = self.cp
        trace_W = cp.sum(w)
        u_sq = u ** 2
        w_safe = cp.maximum(w, 1e-12)
        return (lam / 2.0) * (trace_W * u_sq / w_safe - 1.0)
    
    def project_to_domain(self, w):
        return self.cp.clip(w, self.epsilon, 1.0 / self.epsilon)
    
    def project_to_capped_simplex(self, w, target_sum=None):
        cp = self.cp
        if target_sum is None:
            target_sum = float(self.n)
        
        lo, hi = self.epsilon, 1.0 / self.epsilon
        
        def sum_after_shift(tau):
            return float(cp.sum(cp.clip(w - tau, lo, hi)))
        
        tau_lo = float(cp.min(w)) - hi
        tau_hi = float(cp.max(w)) - lo
        
        if target_sum >= sum_after_shift(tau_lo):
            return cp.full_like(w, hi)
        if target_sum <= sum_after_shift(tau_hi):
            return cp.full_like(w, lo)
        
        for _ in range(50):
            tau_mid = (tau_lo + tau_hi) / 2
            sum_mid = sum_after_shift(tau_mid)
            if sum_mid > target_sum:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid
            if abs(sum_mid - target_sum) < 1e-9:
                break
        
        return cp.clip(w - (tau_lo + tau_hi) / 2, lo, hi)
    
    def normalize_trace(self, w):
        return w * (self.n / self.cp.sum(w))
    
    def get_cut_value(self, spins: np.ndarray) -> float:
        spins = spins.reshape(-1)
        interaction = float(spins.dot(self.W_scipy.dot(spins)))
        total_weight = float(self.W_scipy.sum())
        return 0.25 * (total_weight - interaction)
    
    def local_search_1flip(self, spins: np.ndarray, max_iter: int = None) -> Tuple[np.ndarray, float]:
        """1-flip local search (CPU-based, as it's inherently sequential)"""
        if max_iter is None:
            max_iter = 10 * self.n
            
        spins = spins.copy().astype(np.float64)
        Ws = np.asarray(self.W_scipy.dot(spins)).flatten()
        
        best_cut = self.get_cut_value(spins)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            gains = spins * Ws
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain > 1e-9:
                old_spin = spins[best_idx]
                spins[best_idx] = -old_spin
                col = self.W_scipy.getcol(best_idx).toarray().flatten()
                Ws += col * (-2 * old_spin)
                best_cut += best_gain / 2
                improved = True
        
        return spins.astype(np.int8), best_cut
    
    def local_search_greedy(self, spins: np.ndarray, max_passes: int = 10) -> Tuple[np.ndarray, float]:
        """Greedy local search: multiple passes of 1-flip"""
        spins = spins.copy().astype(np.float64)
        best_cut = self.get_cut_value(spins)
        
        for _ in range(max_passes):
            old_cut = best_cut
            spins, best_cut = self.local_search_1flip(spins, max_iter=self.n * 2)
            if best_cut <= old_cut + 1e-9:
                break
        
        return spins.astype(np.int8), best_cut
    
    def round_eigenvector(self, u, n_trials=50, apply_local_search=False):
        cp = self.cp
        u_cpu = cp.asnumpy(u)
        
        best_cut, best_spins = -np.inf, None
        
        # Sign rounding
        spins = np.sign(u_cpu)
        spins[spins == 0] = 1
        cut = self.get_cut_value(spins)
        if cut > best_cut:
            best_cut, best_spins = cut, spins.copy()
        
        # Random hyperplane
        u_std = np.std(u_cpu)
        for _ in range(n_trials):
            t = np.random.normal(0, u_std + 1e-6)
            spins = np.sign(u_cpu - t)
            spins[spins == 0] = 1
            cut = self.get_cut_value(spins)
            if cut > best_cut:
                best_cut, best_spins = cut, spins.copy()
        
        # Apply local search if requested
        if apply_local_search and best_spins is not None:
            best_spins, best_cut = self.local_search_greedy(best_spins)
        
        return best_spins, best_cut

    def solve(self, eta=0.01, T=200, verbose=True, eval_frequency=10, lr_decay=0.998, 
              n_rounding=50, local_search=True, ls_frequency=50):
        logger.info(f"DSN GPU: T={T}, η={eta}, LS={local_search}")
        t0 = time.time()
        cp = self.cp
        
        self.w = cp.ones(self.n, dtype=cp.float32)
        self.w = self.normalize_trace(self.w)
        self.w = self.project_to_domain(self.w)
        self._cached_eigvec = None
        
        best_cut, best_spins, best_bound = -np.inf, None, -np.inf
        best_w = self.w.copy()
        history = {'bound': [], 'cut': [], 'time': []}
        no_improve_count, patience = 0, 20
        current_eta = eta
        
        for t in range(T):
            B, lam, u = self.compute_spectral_bound(self.w)
            
            # Evaluate BEFORE update
            if t % eval_frequency == 0 or t == T - 1:
                # Apply local search periodically
                apply_ls = local_search and (t % ls_frequency == 0 or t == T - 1)
                spins, cut = self.round_eigenvector(u, n_trials=n_rounding, apply_local_search=apply_ls)
                if cut > best_cut:
                    best_cut, best_spins = cut, spins.copy()
                
                history['bound'].append(B)
                history['cut'].append(cut)
                history['time'].append(time.time() - t0)
                
                if verbose and t % (eval_frequency * 4) == 0:
                    ls_marker = " [LS]" if apply_ls else ""
                    logger.info(f"t={t:03d}: B={B:.2f}, Cut={cut:.0f}, Best={best_cut:.0f}{ls_marker}")
            
            # Track best bound
            restored = False
            if B >= best_bound - 1e-9:
                if B > best_bound:
                    best_bound = B
                    best_w = self.w.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > patience:
                    self.w = best_w.copy()
                    self._cached_eigvec = None
                    current_eta *= 0.5
                    no_improve_count = 0
                    restored = True
            
            if restored:
                continue
            
            grad = self.compute_gradient(self.w, lam, u)
            w_new = self.w + current_eta * grad
            w_new = self.project_to_capped_simplex(w_new, target_sum=self.n)
            self.w = w_new
            current_eta *= lr_decay
        
        # Final local search
        if local_search and best_spins is not None:
            cut_before_ls = best_cut
            best_spins, best_cut = self.local_search_greedy(best_spins, max_passes=20)
            if verbose and best_cut > cut_before_ls:
                logger.info(f"Final LS improved cut: {cut_before_ls:.0f} → {best_cut:.0f}")
        
        elapsed = time.time() - t0
        logger.info(f"DSN GPU done in {elapsed:.2f}s | Cut={best_cut:.0f}")
        
        return best_spins, best_cut, best_bound, history


# =============================================================================
# Wrapper for Benchmark Framework Integration
# =============================================================================

class DSNSolverWrapper:
    """
    Wrapper class to integrate DSN with the Ising benchmark pipeline.
    
    Compatible with main.py interface:
        solver = DSNSolverWrapper(instance_name, dataset, ...)
        solver.solve(J)
    """
    
    # Hyperparameter presets for different scenarios
    PRESETS = {
        'default': {
            'eta': 0.005,
            'T': 300,
            'lr_decay': 0.998,
            'patience': 20,
            'grad_clip': 1.0,
            'n_rounding': 50,
            'init': 'identity',
            'local_search': True,
            'ls_frequency': 50,
        },
        'fast': {
            'eta': 0.01,
            'T': 150,
            'lr_decay': 0.995,
            'patience': 15,
            'grad_clip': 1.0,
            'n_rounding': 30,
            'init': 'identity',
            'local_search': True,
            'ls_frequency': 75,  # Less frequent for speed
        },
        'thorough': {
            'eta': 0.003,
            'T': 500,
            'lr_decay': 0.999,
            'patience': 30,
            'grad_clip': 0.5,
            'n_rounding': 100,
            'init': 'degree',
            'local_search': True,
            'ls_frequency': 50,
        },
        'large_scale': {
            'eta': 0.002,
            'T': 400,
            'lr_decay': 0.999,
            'patience': 25,
            'grad_clip': 0.5,
            'n_rounding': 50,
            'init': 'degree',
            'local_search': True,
            'ls_frequency': 100,  # Less frequent for large graphs
        }
    }
    
    def __init__(
        self,
        instance_name: str,
        dataset: str,
        random_seed: int = None,
        gpu: bool = True,
        preset: str = 'default',
        input_convention: str = 'auto',
        run_name: str = None,
        # Override individual hyperparameters
        eta: float = None,
        T: int = None,
        lr_decay: float = None,
        patience: int = None,
        grad_clip: float = None,
        n_rounding: int = None,
        init: str = None,
        epsilon: float = 0.01,
        local_search: bool = None,
        ls_frequency: int = None,
        **kwargs
    ):
        """
        Initialize DSN solver wrapper.
        
        Args:
            instance_name: Name of the problem instance
            dataset: Dataset name (e.g., 'Gset', 'real_world')
            random_seed: Random seed for reproducibility
            gpu: Whether to use GPU (requires CuPy)
            preset: Hyperparameter preset ('default', 'fast', 'thorough', 'large_scale')
            input_convention: 'maxcut', 'ising', or 'auto'
            run_name: Custom name for results logging
            
            # Individual hyperparameter overrides:
            eta: Learning rate
            T: Number of iterations
            lr_decay: Learning rate decay factor
            patience: Early stopping patience
            grad_clip: Gradient clipping threshold
            n_rounding: Number of rounding trials
            init: Weight initialization ('identity' or 'degree')
            epsilon: Box constraint parameter
            local_search: Enable 1-flip local search refinement
            ls_frequency: How often to apply local search during optimization
        """
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.gpu = gpu
        self.input_convention = input_convention
        self.epsilon = epsilon
        
        # Auto-detect preset from run_name if not explicitly set
        if preset == 'default' and run_name:
            if 'fast' in run_name.lower():
                preset = 'fast'
            elif 'thorough' in run_name.lower():
                preset = 'thorough'
            elif 'large' in run_name.lower():
                preset = 'large_scale'
        
        # Load preset and override with explicit parameters
        params = self.PRESETS.get(preset, self.PRESETS['default']).copy()
        if eta is not None: params['eta'] = eta
        if T is not None: params['T'] = T
        if lr_decay is not None: params['lr_decay'] = lr_decay
        if patience is not None: params['patience'] = patience
        if grad_clip is not None: params['grad_clip'] = grad_clip
        if n_rounding is not None: params['n_rounding'] = n_rounding
        if init is not None: params['init'] = init
        if local_search is not None: params['local_search'] = local_search
        if ls_frequency is not None: params['ls_frequency'] = ls_frequency
        
        self.params = params
        self.solver_name = run_name if run_name else f"DSN_{preset}"
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Initialized {self.solver_name} | preset={preset} | GPU={gpu}")
        logger.info(f"Params: η={params['eta']}, T={params['T']}, init={params['init']}, LS={params['local_search']}")

    def solve(self, J: Union[np.ndarray, sp.spmatrix]):
        """
        Solve the MaxCut/Ising problem.
        
        Args:
            J: Adjacency/coupling matrix
        """
        start_time = time.time()
        
        if not sp.issparse(J):
            J = sp.csr_matrix(J)
        
        n = J.shape[0]
        
        # Choose CPU or GPU solver
        use_gpu = self.gpu
        if use_gpu:
            try:
                import cupy
                logger.info("Using GPU solver (CuPy)")
            except ImportError:
                logger.warning("CuPy not available, falling back to CPU")
                use_gpu = False
        
        try:
            if use_gpu:
                solver = DSNSolverGPU(
                    J, 
                    epsilon=self.epsilon,
                    random_seed=self.random_seed,
                    input_convention=self.input_convention
                )
                spins, cut, bound, history = solver.solve(
                    eta=self.params['eta'],
                    T=self.params['T'],
                    lr_decay=self.params['lr_decay'],
                    n_rounding=self.params['n_rounding'],
                    local_search=self.params['local_search'],
                    ls_frequency=self.params['ls_frequency'],
                    verbose=True
                )
            else:
                solver = DSNSolver(
                    J,
                    epsilon=self.epsilon,
                    random_seed=self.random_seed,
                    input_convention=self.input_convention
                )
                spins, cut, bound, history = solver.solve(
                    eta=self.params['eta'],
                    T=self.params['T'],
                    init=self.params['init'],
                    lr_decay=self.params['lr_decay'],
                    patience=self.params['patience'],
                    grad_clip=self.params['grad_clip'],
                    n_rounding=self.params['n_rounding'],
                    local_search=self.params['local_search'],
                    ls_frequency=self.params['ls_frequency'],
                    verbose=True
                )
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            raise
        
        time_taken = time.time() - start_time
        energy = -cut  # Ising energy convention
        
        self._store_results(
            energy=energy,
            spins=spins,
            time_taken=time_taken,
            cut=cut,
            bound=bound,
            history=history
        )
        
        return spins, cut

    def _store_results(
        self, 
        energy: float, 
        spins: np.ndarray, 
        time_taken: float, 
        cut: float,
        bound: float,
        history: dict
    ):
        """Store results to CSV file."""
        import pandas as pd
        
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': energy,
            'cut': cut,
            'bound': bound,
            'time': time_taken,
            'eta': self.params['eta'],
            'T': self.params['T'],
            'init': self.params['init'],
            'lr_decay': self.params['lr_decay'],
            'patience': self.params['patience'],
            'grad_clip': self.params['grad_clip'],
            'n_rounding': self.params['n_rounding'],
            'epsilon': self.epsilon,
            'gpu': self.gpu,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()
        
        new_row = pd.DataFrame([result_data])
        if not df.empty:
            duplicate_mask = (
                (df['instance_name'] == result_data['instance_name']) &
                (df['dataset'] == result_data['dataset']) &
                (df['seed'] == result_data['seed']) &
                (df['solver_name'] == result_data['solver_name'])
            )
            if duplicate_mask.any():
                df.loc[duplicate_mask, list(result_data.keys())] = list(result_data.values())
            else:
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to: {csv_file}")
        logger.info(f"Cut: {cut:.0f} | Bound: {bound:.2f} | Time: {time_taken:.2f}s")


# =============================================================================
# Convenience Functions for Direct Use
# =============================================================================

def solve_maxcut_dsn(
    J: Union[np.ndarray, sp.spmatrix],
    gpu: bool = True,
    preset: str = 'default',
    seed: int = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to solve MaxCut using DSN.
    
    Args:
        J: Adjacency matrix (scipy sparse or numpy)
        gpu: Use GPU if available
        preset: Hyperparameter preset
        seed: Random seed
        verbose: Print progress
    
    Returns:
        spins: Best spin configuration
        cut: Best cut value
    """
    wrapper = DSNSolverWrapper(
        instance_name='direct_call',
        dataset='direct',
        random_seed=seed,
        gpu=gpu,
        preset=preset
    )
    return wrapper.solve(J)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSN Solver Wrapper Test")
    print("=" * 60)
    
    # Generate test graph
    n = 200
    np.random.seed(42)
    
    # SBM graph
    p_in, p_out = 0.3, 0.05
    block_size = n // 2
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            block_i, block_j = i // block_size, j // block_size
            p = p_in if block_i == block_j else p_out
            if np.random.rand() < p:
                A[i, j] = A[j, i] = 1
    
    J = sp.csr_matrix(A)
    print(f"Test graph: n={n}, edges={J.nnz//2}")
    
    # Test wrapper
    wrapper = DSNSolverWrapper(
        instance_name='test_sbm.txt',
        dataset='test',
        random_seed=42,
        gpu=False,  # Use CPU for testing
        preset='default',
        input_convention='maxcut'
    )
    
    spins, cut = wrapper.solve(J)
    print(f"\nFinal result: Cut = {cut:.0f}")