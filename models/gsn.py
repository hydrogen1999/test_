import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator
import time
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DSN")


class DSNSolver:
    """
    Differentiable Spectral Normalization solver.
    Key equations from the paper:
    - Normalized adjacency: N(W) = W^{-1/2} J W^{-1/2}
    - Spectral bound: B(W) = -1/2 * λ_max(N(W)) * Tr(W)
    - Hellmann-Feynman gradient: ∂B/∂w_i = (λ/2)[Tr(W)/w_i * u_i² - 1]
    
    The bound B(W) is a lower bound on the Ising energy E(s).
    Maximizing B(W) tightens this bound.
    """
    
    def __init__(
        self,
        coupling_matrix: Union[sp.spmatrix, np.ndarray],
        epsilon: float = 0.01,
        random_seed: Optional[int] = None,
        input_convention: str = 'auto'
    ):
        """
        Initialize DSN solver.
        
        Args:
            coupling_matrix: Adjacency/coupling matrix J
            epsilon: Box constraint parameter, w_i ∈ [ε, ε⁻¹]
            random_seed: Random seed for reproducibility
            input_convention: 'maxcut' (J >= 0), 'ising' (J can be negative, 
                             will flip sign), or 'auto' (detect via median)
        """
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
            # Ising convention: J can be negative, flip to MaxCut (J >= 0)
            self.J_scipy = (-self.J_scipy).tocsr()
            logger.info("Input convention: Ising → MaxCut (flipped sign)")
        elif input_convention == 'maxcut':
            # Already MaxCut convention, no change needed
            logger.info("Input convention: MaxCut (no sign change)")
        elif input_convention == 'auto':
            # Auto-detect: flip if majority of weights are negative
            # WARNING: This is fragile for mixed-sign graphs!
            if self.J_scipy.nnz > 0 and np.median(self.J_scipy.data) < 0:
                self.J_scipy = (-self.J_scipy).tocsr()
                logger.warning("Auto-detected Ising convention (median < 0), flipped sign. "
                              "Consider using explicit input_convention parameter.")
        else:
            raise ValueError(f"Unknown input_convention: {input_convention}. "
                           f"Use 'maxcut', 'ising', or 'auto'.")
        
        # Store W = J for MaxCut evaluation
        self.W_scipy = self.J_scipy.copy()
        
        # Initialize diagonal weights
        self.w = np.ones(self.n, dtype=np.float64)
        
        # Cache for eigenvector warm-start
        self._cached_eigvec = None
        
        logger.info(f"DSN Solver: n={self.n}, edges={self.J_scipy.nnz//2}, ε={epsilon}")
    
    def _compute_leading_eigenpair(
        self, 
        w: np.ndarray, 
        tol: float = 1e-6,
        maxiter: int = 500
    ) -> Tuple[float, np.ndarray]:
        """
        Compute (λ_max, u) of N(W) = W^{-1/2} J W^{-1/2}.
        
        Uses operator form to avoid forming N explicitly.
        Warm-starts from previous eigenvector if available.
        """
        w_safe = np.maximum(w, 1e-12)
        w_inv_sqrt = 1.0 / np.sqrt(w_safe)
        
        def matvec(v):
            """N(W) v = W^{-1/2} J W^{-1/2} v"""
            u = v * w_inv_sqrt
            z = self.J_scipy.dot(u)
            return z * w_inv_sqrt
        
        N_op = LinearOperator(
            shape=(self.n, self.n), 
            matvec=matvec, 
            dtype=np.float64
        )
        
        # Prepare initial vector
        if self._cached_eigvec is not None:
            X0 = self._cached_eigvec.reshape(-1, 1)
        else:
            X0 = np.random.randn(self.n, 1)
            X0 /= np.linalg.norm(X0)
        
        # Try LOBPCG first (better for warm starts)
        try:
            eigenvalues, eigenvectors = lobpcg(
                N_op, X0, 
                largest=True, 
                tol=tol, 
                maxiter=maxiter,
                verbosityLevel=0
            )
            lam = eigenvalues[0]
            u = eigenvectors[:, 0]
        except Exception:
            # Fallback to eigsh
            eigenvalues, eigenvectors = eigsh(
                N_op, k=1, which='LA', 
                tol=tol, maxiter=maxiter,
                v0=X0.flatten()
            )
            lam = eigenvalues[0]
            u = eigenvectors[:, 0]
        
        # Normalize and cache
        u = u / (np.linalg.norm(u) + 1e-12)
        self._cached_eigvec = u.copy()
        
        return lam, u
    
    def compute_spectral_bound(self, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute spectral bound B(W) = -1/2 * λ_max(N(W)) * Tr(W).
        
        Returns:
            B: The spectral bound (lower bound on energy)
            lam: The largest eigenvalue λ_max
            u: The corresponding eigenvector
        """
        lam, u = self._compute_leading_eigenpair(w)
        trace_W = np.sum(w)
        B = -0.5 * lam * trace_W
        return B, lam, u
    
    def compute_gradient(
        self, 
        w: np.ndarray, 
        lam: float, 
        u: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hellmann-Feynman gradient (Theorem 4, Algorithm 1 line 5).
        
        Formula: g_i = (λ/2) * [Tr(W)/w_i * u_i² - 1]
        
        This is the gradient of B(W) w.r.t. w.
        """
        trace_W = np.sum(w)
        u_sq = u ** 2
        
        # Avoid division by zero
        w_safe = np.maximum(w, 1e-12)
        
        grad = (lam / 2.0) * (trace_W * u_sq / w_safe - 1.0)
        return grad
    
    def project_to_domain(self, w: np.ndarray) -> np.ndarray:
        """
        Project to box constraint: w_i ∈ [ε, ε⁻¹].
        """
        return np.clip(w, self.epsilon, 1.0 / self.epsilon)
    
    def project_to_capped_simplex(self, w: np.ndarray, target_sum: float = None) -> np.ndarray:
        """
        Project onto intersection of box [ε, 1/ε]^n and hyperplane Σw_i = target_sum.
        
        Finds τ via bisection such that:
            w_i' = clip(w_i - τ, ε, 1/ε)  and  Σw_i' = target_sum
        
        This is the proper projection for (box ∩ fixed-trace) constraint.
        """
        if target_sum is None:
            target_sum = float(self.n)
        
        lo, hi = self.epsilon, 1.0 / self.epsilon
        
        # Compute sum as function of shift τ
        def sum_after_shift(tau):
            return np.sum(np.clip(w - tau, lo, hi))
        
        # Find bounds for bisection
        tau_lo = np.min(w) - hi  # ensures at least one element at upper bound
        tau_hi = np.max(w) - lo  # ensures at least one element at lower bound
        
        # Check if target is achievable
        sum_at_tau_lo = sum_after_shift(tau_lo)
        sum_at_tau_hi = sum_after_shift(tau_hi)
        
        if target_sum >= sum_at_tau_lo:
            return np.full_like(w, hi)  # Target too high
        if target_sum <= sum_at_tau_hi:
            return np.full_like(w, lo)  # Target too low
        
        # Bisection to find τ
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
        """
        Normalize trace to prevent scale drift.
        
        By Lemma 3, the bound is scale-invariant: B(cW) = B(W).
        We normalize Tr(W) = n for numerical stability.
        """
        if target_trace is None:
            target_trace = self.n
        return w * (target_trace / np.sum(w))
    
    def get_cut_value(self, spins: np.ndarray) -> float:
        """
        Compute MaxCut value.
        
        Cut = 1/4 * Σ_{ij} w_ij * (1 - s_i*s_j)
        """
        spins = spins.reshape(-1)
        interaction = float(spins.dot(self.W_scipy.dot(spins)))
        total_weight = float(self.W_scipy.sum())
        return 0.25 * (total_weight - interaction)
    
    def round_eigenvector(
        self, 
        u: np.ndarray, 
        n_trials: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        Round eigenvector to discrete spins using multiple strategies.
        
        Strategies:
        1. Sign rounding: s = sign(u)
        2. Median threshold: s = sign(u - median(u))
        3. Random hyperplane: s = sign(u - t) for random t ~ N(0, σ)
        """
        best_cut = -np.inf
        best_spins = None
        
        # Strategy 1: Sign rounding
        spins = np.sign(u)
        spins[spins == 0] = 1
        cut = self.get_cut_value(spins)
        if cut > best_cut:
            best_cut = cut
            best_spins = spins.copy()
        
        # Strategy 2: Median threshold
        spins = np.sign(u - np.median(u))
        spins[spins == 0] = 1
        cut = self.get_cut_value(spins)
        if cut > best_cut:
            best_cut = cut
            best_spins = spins.copy()
        
        # Strategy 3: Random hyperplane rounding
        u_std = np.std(u)
        for _ in range(n_trials):
            threshold = np.random.normal(0, u_std + 1e-6)
            spins = np.sign(u - threshold)
            spins[spins == 0] = 1
            cut = self.get_cut_value(spins)
            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()
        
        return best_spins, best_cut
    
    def solve(
        self,
        eta: float = 0.01,
        T: int = 200,
        init: str = 'identity',
        normalize_each_step: bool = True,
        verbose: bool = True,
        eval_frequency: int = 10,
        grad_clip: float = 1.0,
        lr_decay: float = 0.995,
        patience: int = 30
    ) -> Tuple[np.ndarray, float, float, dict]:
        """
        Main optimization loop (Algorithm 1).
        
        Args:
            eta: Initial learning rate (step size)
            T: Number of iterations
            init: Initialization strategy ('identity', 'degree', 'random')
            normalize_each_step: Whether to normalize trace each iteration
            verbose: Print progress
            eval_frequency: How often to evaluate solution
            grad_clip: Gradient clipping threshold
            lr_decay: Learning rate decay per iteration
            patience: Early stopping patience
            
        Returns:
            best_spins: Best spin configuration found
            best_cut: Best cut value found
            best_bound: Best (tightest) bound found
            history: Dictionary with optimization history
        """
        logger.info(f"DSN: T={T}, η={eta}, init={init}")
        t0 = time.time()
        
        # Initialize W^(0)
        if init == 'identity':
            self.w = np.ones(self.n)
        elif init == 'degree':
            degrees = np.asarray(self.J_scipy.sum(axis=1)).flatten()
            self.w = np.maximum(degrees / np.mean(degrees), self.epsilon)
        elif init == 'random':
            self.w = np.random.uniform(0.8, 1.2, self.n)
        else:
            self.w = np.ones(self.n)
        
        # Normalize initial trace
        self.w = self.normalize_trace(self.w)
        self.w = self.project_to_domain(self.w)
        self._cached_eigvec = None
        
        # Track best solutions
        best_cut = -np.inf
        best_spins = None
        best_bound = -np.inf
        best_w = self.w.copy()
        
        # For early stopping
        no_improve_count = 0
        
        # History
        history = {
            'bound': [],
            'eigenvalue': [],
            'cut': [],
            'time': []
        }
        
        current_eta = eta
        
        for t in range(T):
            # Step 3-4: Compute leading eigenpair
            B, lam, u = self.compute_spectral_bound(self.w)
            
            # FIX 1: Evaluate/round BEFORE update so cut corresponds to current w^(t)
            if t % eval_frequency == 0 or t == T - 1:
                spins, cut = self.round_eigenvector(u)
                
                if cut > best_cut:
                    best_cut = cut
                    best_spins = spins.copy()
                
                history['bound'].append(B)
                history['eigenvalue'].append(lam)
                history['cut'].append(cut)
                history['time'].append(time.time() - t0)
                
                if verbose and t % (eval_frequency * 4) == 0:
                    logger.info(
                        f"t={t:03d}: B={B:.2f}, λ={lam:.4f}, "
                        f"Cut={cut:.0f}, Best={best_cut:.0f}"
                    )
            
            # Track best bound and restore if degrading
            # Use tolerance to handle numerical noise
            restored = False
            if B >= best_bound - 1e-9:
                if B > best_bound:
                    best_bound = B
                    best_w = self.w.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If bound is degrading badly, restore best weights
                if no_improve_count > patience:
                    self.w = best_w.copy()
                    self._cached_eigvec = None  # Reset warm-start cache
                    current_eta *= 0.5  # Reduce learning rate
                    no_improve_count = 0
                    restored = True
                    if verbose:
                        logger.info(f"t={t}: Restored best w, η→{current_eta:.4f}")
            
            # BUG FIX: If restored, eigenpair must match current w
            # Skip to next iteration to recompute eigenpair
            if restored:
                continue
            
            # Step 5: Compute Hellmann-Feynman gradient
            grad = self.compute_gradient(self.w, lam, u)
            
            # Gradient clipping for stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm > grad_clip * self.n:
                grad = grad * (grad_clip * self.n / grad_norm)
            
            # Step 6: Gradient ASCENT (maximize B)
            w_new = self.w + current_eta * grad
            
            # FIX 2: Use proper capped simplex projection for (box ∩ fixed-trace)
            if normalize_each_step:
                w_new = self.project_to_capped_simplex(w_new, target_sum=self.n)
            else:
                w_new = self.project_to_domain(w_new)
            
            self.w = w_new
            
            # Learning rate decay
            current_eta *= lr_decay
        
        elapsed = time.time() - t0
        logger.info(f"DSN done in {elapsed:.2f}s | Cut={best_cut:.0f} | Bound={best_bound:.2f}")
        
        return best_spins, best_cut, best_bound, history


class DSNSolverGPU:
    """
    GPU-accelerated DSN solver using CuPy.
    
    Provides significant speedup for large instances (n > 10000).
    """
    
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
            self.cp = cp
            self.cpx_sparse = cpx_sparse
        except ImportError:
            raise ImportError("CuPy required for GPU acceleration. Install with: pip install cupy-cuda12x")
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Convert to sparse CSR
        if sp.issparse(coupling_matrix):
            J_scipy = coupling_matrix.tocsr().astype(np.float64)
        else:
            J_scipy = sp.csr_matrix(coupling_matrix, dtype=np.float64)
        
        # Symmetrize and remove diagonal
        J_scipy = 0.5 * (J_scipy + J_scipy.T)
        J_scipy.setdiag(0)
        J_scipy.eliminate_zeros()
        
        self.n = J_scipy.shape[0]
        self.epsilon = epsilon
        
        # Handle sign convention
        if input_convention == 'ising':
            J_scipy = (-J_scipy).tocsr()
            logger.info("GPU: Input convention: Ising → MaxCut (flipped sign)")
        elif input_convention == 'maxcut':
            logger.info("GPU: Input convention: MaxCut (no sign change)")
        elif input_convention == 'auto':
            if J_scipy.nnz > 0 and np.median(J_scipy.data) < 0:
                J_scipy = (-J_scipy).tocsr()
                logger.warning("GPU: Auto-detected Ising convention, flipped sign.")
        else:
            raise ValueError(f"Unknown input_convention: {input_convention}")
        
        # Transfer to GPU
        self.J_gpu = cpx_sparse.csr_matrix(J_scipy.astype(cp.float32))
        self.W_scipy = J_scipy.copy()  # Keep CPU copy for evaluation
        
        # Initialize weights on GPU
        self.w = cp.ones(self.n, dtype=cp.float32)
        self._cached_eigvec = None
        
        logger.info(f"DSN GPU Solver: n={self.n}, edges={J_scipy.nnz//2}")
    
    def _compute_leading_eigenpair(self, w, tol=1e-5, maxiter=300):
        """GPU eigenpair computation using CuPy's eigsh."""
        import cupyx.scipy.sparse.linalg as cpx_splinalg
        
        cp = self.cp
        w_safe = cp.maximum(w, 1e-12)
        w_inv_sqrt = 1.0 / cp.sqrt(w_safe)
        
        class NOperator(cpx_splinalg.LinearOperator):
            def __init__(self, J_gpu, w_inv_sqrt):
                self.J_gpu = J_gpu
                self.w_inv_sqrt = w_inv_sqrt
                n = w_inv_sqrt.size
                super().__init__(dtype=cp.float32, shape=(n, n))
            
            def _matvec(self, v):
                u = v * self.w_inv_sqrt
                z = self.J_gpu.dot(u)
                return z * self.w_inv_sqrt
        
        N_op = NOperator(self.J_gpu, w_inv_sqrt)
        
        v0 = self._cached_eigvec if self._cached_eigvec is not None else None
        
        eigenvalues, eigenvectors, _ = cpx_splinalg.eigsh(
            N_op, k=1, which='LA',
            tol=tol, maxiter=maxiter, v0=v0
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
        grad = (lam / 2.0) * (trace_W * u_sq / w_safe - 1.0)
        return grad
    
    def project_to_domain(self, w):
        return self.cp.clip(w, self.epsilon, 1.0 / self.epsilon)
    
    def project_to_capped_simplex(self, w, target_sum=None):
        """GPU version of capped simplex projection via bisection."""
        cp = self.cp
        if target_sum is None:
            target_sum = float(self.n)
        
        lo, hi = self.epsilon, 1.0 / self.epsilon
        
        def sum_after_shift(tau):
            return float(cp.sum(cp.clip(w - tau, lo, hi)))
        
        tau_lo = float(cp.min(w)) - hi
        tau_hi = float(cp.max(w)) - lo
        
        sum_at_tau_lo = sum_after_shift(tau_lo)
        sum_at_tau_hi = sum_after_shift(tau_hi)
        
        if target_sum >= sum_at_tau_lo:
            return cp.full_like(w, hi)
        if target_sum <= sum_at_tau_hi:
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
        
        tau = (tau_lo + tau_hi) / 2
        return cp.clip(w - tau, lo, hi)
    
    def normalize_trace(self, w):
        return w * (self.n / self.cp.sum(w))
    
    def get_cut_value(self, spins: np.ndarray) -> float:
        spins = spins.reshape(-1)
        interaction = float(spins.dot(self.W_scipy.dot(spins)))
        total_weight = float(self.W_scipy.sum())
        return 0.25 * (total_weight - interaction)
    
    def round_eigenvector(self, u, n_trials=50):
        cp = self.cp
        u_cpu = cp.asnumpy(u)
        
        best_cut = -np.inf
        best_spins = None
        
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
        
        return best_spins, best_cut
    
    def solve(self, eta=0.01, T=200, verbose=True, eval_frequency=10, lr_decay=0.998):
        logger.info(f"DSN GPU: T={T}, η={eta}")
        t0 = time.time()
        cp = self.cp
        
        self.w = cp.ones(self.n, dtype=cp.float32)
        self.w = self.normalize_trace(self.w)
        self.w = self.project_to_domain(self.w)
        self._cached_eigvec = None
        
        best_cut, best_spins, best_bound = -np.inf, None, -np.inf
        best_w = self.w.copy()
        history = {'bound': [], 'cut': [], 'time': []}
        no_improve_count = 0
        patience = 20
        
        current_eta = eta
        
        for t in range(T):
            B, lam, u = self.compute_spectral_bound(self.w)
            
            # FIX 1: Evaluate/round before update so cut corresponds to current w^(t)
            if t % eval_frequency == 0 or t == T - 1:
                spins, cut = self.round_eigenvector(u)
                if cut > best_cut:
                    best_cut, best_spins = cut, spins.copy()
                
                history['bound'].append(B)
                history['cut'].append(cut)
                history['time'].append(time.time() - t0)
                
                if verbose and t % (eval_frequency * 4) == 0:
                    logger.info(f"t={t:03d}: B={B:.2f}, Cut={cut:.0f}, Best={best_cut:.0f}")
            
            # Track best bound with tolerance, restore if degrading
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
                    self._cached_eigvec = None  # Reset warm-start
                    current_eta *= 0.5
                    no_improve_count = 0
                    restored = True
                    if verbose:
                        logger.info(f"t={t}: Restored best w, η→{current_eta:.4f}")
            
            if restored:
                continue
            
            grad = self.compute_gradient(self.w, lam, u)
            
            # FIX 2: Use proper capped simplex projection
            w_new = self.w + current_eta * grad
            w_new = self.project_to_capped_simplex(w_new, target_sum=self.n)
            self.w = w_new
            
            current_eta *= lr_decay
        
        elapsed = time.time() - t0
        logger.info(f"DSN GPU done in {elapsed:.2f}s | Cut={best_cut:.0f}")
        
        return best_spins, best_cut, best_bound, history


# =============================================================================
# Test and Comparison
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSN Solver Test")
    print("=" * 60)
    
    # Generate test graph (Stochastic Block Model)
    n = 200
    p_in, p_out = 0.3, 0.05
    k = 2
    np.random.seed(42)
    
    # Create SBM adjacency
    block_size = n // k
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            block_i, block_j = i // block_size, j // block_size
            p = p_in if block_i == block_j else p_out
            if np.random.rand() < p:
                A[i, j] = A[j, i] = 1
    
    J = sp.csr_matrix(A)
    print(f"Graph: n={n}, edges={J.nnz//2}, p_in={p_in}, p_out={p_out}")
    
    # Test DSN solver (input is already MaxCut convention with positive weights)
    print("\n--- DSN Solver ---")
    solver = DSNSolver(J, epsilon=0.01, random_seed=42, input_convention='maxcut')
    
    # Baseline (W = I)
    solver_baseline = DSNSolver(J, epsilon=0.01, random_seed=42, input_convention='maxcut')
    B_baseline, lam_baseline, u_baseline = solver_baseline.compute_spectral_bound(
        np.ones(n)
    )
    spins_baseline, cut_baseline = solver_baseline.round_eigenvector(u_baseline)
    print(f"Baseline (W=I): Cut={cut_baseline:.0f}, B={B_baseline:.2f}")
    
    # DSN optimization
    best_spins, best_cut, best_bound, history = solver.solve(
        eta=0.005, T=300, verbose=True, patience=50, lr_decay=0.998
    )
    
    print("\n=== Results ===")
    print(f"Baseline: Cut={cut_baseline:.0f}, Bound={B_baseline:.2f}")
    print(f"DSN:      Cut={best_cut:.0f}, Bound={best_bound:.2f}")
    print(f"Improvement: +{best_cut - cut_baseline:.0f} cut, "
          f"Bound tightened by {(B_baseline - best_bound)/abs(B_baseline)*100:.1f}%")