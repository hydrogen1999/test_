import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import torch
import torch.nn as nn
import logging
import time
from typing import Tuple, Union, List

# Configure Logger
logger = logging.getLogger("SpectralResearch")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class SpectralObjectives:
    """
    Utilities for evaluating MaxCut objective.
    W must be the MaxCut adjacency matrix (weights >= 0, symmetric, zero diagonal).
    """
    @staticmethod
    def get_cut_value(W: sp.spmatrix, spins: np.ndarray) -> float:
        # Ensure spins is numpy array
        if isinstance(spins, torch.Tensor):
            spins = spins.detach().cpu().numpy()

        spins = spins.reshape(-1)

        # Explicitly cast to float to avoid sparse matrix return types or int overflows
        interaction = float(spins.dot(W.dot(spins)))
        total_weight = float(W.sum())

        # MaxCut = 1/4 * sum(w_ij * (1 - s_i*s_j))
        return 0.25 * (total_weight - interaction)


def _to_csr(A: Union[sp.spmatrix, np.ndarray]) -> sp.csr_matrix:
    if sp.issparse(A):
        return A.tocsr()
    return sp.csr_matrix(A)


def _symmetrize_gset(W: sp.csr_matrix) -> sp.csr_matrix:
    """
    Robust symmetrization logic for Gset and similar datasets.
    """
    W = W.tocsr().copy()
    W.setdiag(0)
    W.eliminate_zeros()

    upper = sp.triu(W, k=1)
    lower = sp.tril(W, k=-1)

    if lower.nnz == 0 and upper.nnz > 0:
        logger.info("  -> Detected Upper Triangular input. Symmetrizing...")
        W_sym = upper + upper.T
    elif upper.nnz == 0 and lower.nnz > 0:
        logger.info("  -> Detected Lower Triangular input. Symmetrizing...")
        W_sym = lower + lower.T
    else:
        # Check if it's already symmetric to avoid double counting
        if (W - W.T).nnz == 0:
             W_sym = W
        else:
             # General case
             logger.info("  -> Input has both U/L parts but not symmetric. Averaging...")
             W_sym = 0.5 * (W + W.T)

    W_sym = W_sym.tocsr()
    W_sym.setdiag(0)
    W_sym.eliminate_zeros()
    return W_sym


class ImprovedSpectralSolver(nn.Module):
    """
    Implementation of IJCAI 2026 methodology with Perron-Frobenius fix.
    - W (Adjacency) >= 0 is used for Cut Evaluation.
    - A = -W is used for Spectral/Gradient steps to target the correct eigenvector.
    """
    def __init__(
        self,
        adjacency_matrix: Union[sp.spmatrix, np.ndarray],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        A_in = _to_csr(adjacency_matrix)
        self.n = A_in.shape[0]
        self.device = device

        logger.info(f"Initializing Improved Solver on [{device}] for Graph N={self.n}")

        # 1) Robust Symmetrization
        W = _symmetrize_gset(A_in)

        # Handle Ising convention (negative weights) -> MaxCut convention (positive weights)
        if W.nnz > 0 and np.median(W.data) < 0:
            logger.info("  -> Detected mostly-negative weights (Ising formulation). Flipping signs to W >= 0...")
            W = W.copy()
            W.data *= -1
            W.eliminate_zeros()

        # Strict Validation: Do not use abs(W) as a band-aid.
        if W.nnz > 0 and W.data.min() < 0:
            # Print negative values for debugging
            neg_indices = np.where(W.data < 0)[0]
            logger.error(f"  -> Found {len(neg_indices)} negative edges. Example: {W.data[neg_indices[:5]]}")
            raise ValueError("W still has negative weights after sign fix. Check loader / conversion logic.")

        self.W_scipy: sp.csr_matrix = W

        # Immediate Diagnostic Printing
        logger.info("--- W Matrix Diagnostics (CHECK FOR CUT=73 BUG) ---")
        diag_val = float(W.diagonal().max()) if W.shape[0] > 0 else 0
        diff_nnz = (W - W.T).nnz
        w_sum = float(W.sum())
        logger.info(f"  > n: {self.n}")
        logger.info(f"  > nnz: {W.nnz}")
        logger.info(f"  > min/max weights: {W.data.min() if W.nnz > 0 else 0:.4f} / {W.data.max() if W.nnz > 0 else 0:.4f}")
        logger.info(f"  > sum(W): {w_sum:.2f}")
        logger.info(f"  > Symmetry diff nnz: {diff_nnz} (Should be 0)")
        logger.info(f"  > Diagonal max: {diag_val} (Should be 0)")

        if w_sum < self.n:
            logger.warning("WARNING: sum(W) is suspiciously low. Check if Data Loader parsed the file correctly!")

        # 2) Spectral operator uses A = -W (Perron-Frobenius fix)
        self.A_scipy: sp.csr_matrix = (-W).tocsr()

        # 3) Torch sparse for A (spectral operator) on GPU
        coo = self.A_scipy.tocoo()
        indices = np.vstack((coo.row, coo.col))
        values = coo.data.astype(np.float32)

        i = torch.as_tensor(indices, dtype=torch.long)
        v = torch.as_tensor(values, dtype=torch.float32)
        shape = coo.shape
        # Coalesce is crucial for some sparse operations
        self.A_torch = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=self.device).coalesce()

        # Learnable parameters (log_w)
        self.log_w = nn.Parameter(torch.zeros(self.n, device=self.device))

        # Cache
        self.register_buffer('cached_eigenvector', torch.randn(self.n, 1, device=self.device))
        self._normalize_cached_vector()

    def _normalize_cached_vector(self):
        with torch.no_grad():
            norm = torch.norm(self.cached_eigenvector) + 1e-8
            self.cached_eigenvector.data /= norm

    def _spectral_operator_mult(self, v: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
        """
        Computes: D_w^{-1/2} * A * D_w^{-1/2} * v
        """
        d_inv_sqrt = torch.rsqrt(w_vec + 1e-6).unsqueeze(1)
        u = v * d_inv_sqrt
        z = torch.sparse.mm(self.A_torch, u)
        out = z * d_inv_sqrt
        return out

    def _warm_start_power_iteration(self, w_vec: torch.Tensor, steps: int = 5) -> torch.Tensor:
        v = self.cached_eigenvector.detach().clone()
        for _ in range(steps):
            v = self._spectral_operator_mult(v, w_vec)
            v = v / (torch.norm(v) + 1e-8)
        return v

    def solve_sdp_proxy(self, n_rounding: int = 100) -> Tuple[np.ndarray, float]:
        """
        Initialize with Leading Eigenvector of A = -W (Normalized).
        """
        logger.info("--- Phase A: Running SDP Proxy (A = -W) ---")
        t0 = time.time()

        # Calculate degrees from W (positive)
        degrees = np.asarray(self.W_scipy.sum(axis=1)).reshape(-1)
        d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-8)

        def matvec(v):
            v = v * d_inv_sqrt
            v = self.A_scipy.dot(v) # A = -W
            v = v * d_inv_sqrt
            return v

        op = LinearOperator((self.n, self.n), matvec=matvec)

        # Find largest eigenvector of A (equivalent to smallest of Laplacian)
        vals, vecs = eigsh(op, k=1, which='LA', tol=1e-4)
        v_sdp = vecs[:, 0]

        best_cut = -np.inf
        best_spins = None

        # Improved Rounding: Random Thresholding
        # Instead of adding noise and taking sign(0), we randomly slide the threshold t.
        v_std = np.std(v_sdp)

        for _ in range(n_rounding):
            # Select threshold t from a normal distribution around 0
            t = np.random.normal(loc=0.0, scale=v_std + 1e-12)

            # Threshold the vector at t
            spins = np.sign(v_sdp - t)
            spins[spins == 0] = 1 # Handle zeros

            cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)
            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()

        # Update cache
        self.cached_eigenvector.data = torch.tensor(v_sdp, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self._normalize_cached_vector()

        logger.info(f"SDP Proxy Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut

    def solve_gradient_descent(
        self,
        lr: float = 0.05,
        steps: int = 100,
        warm_start_steps: int = 5
    ) -> Tuple[np.ndarray, float, List[float]]:

        logger.info(f"--- Phase B: Running Gradient Descent (Steps={steps}, LR={lr}) ---")
        optimizer = torch.optim.Adam([self.log_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        best_cut = -np.inf
        best_spins = None
        loss_history: List[float] = []
        t0 = time.time()

        for i in range(steps):
            optimizer.zero_grad()

            # Scale Stabilization: Clamp log_w
            # Prevent w_vec from becoming too large or too small, causing numerical errors (NaN/Inf)
            clamped_log_w = torch.clamp(self.log_w, -5.0, 5.0)
            w_vec = torch.exp(clamped_log_w)

            # Forward pass (Power method)
            v_approx = self._warm_start_power_iteration(w_vec, steps=warm_start_steps)
            self.cached_eigenvector.data = v_approx.data

            # Calculate Loss (Envelope Theorem)
            v_fixed = v_approx.detach()

            # Maximize Rayleigh Quotient => Minimize Negative
            numerator = (v_fixed.T @ self._spectral_operator_mult(v_fixed, w_vec)).squeeze()
            loss = -numerator

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(float(loss.item()))

            # Evaluation Interval
            if i % 10 == 0 or i == steps - 1:
                current_v_cpu = v_approx.detach().cpu().numpy().flatten()
                spins = np.sign(current_v_cpu)
                spins[spins == 0] = 1

                cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)

                if cut > best_cut:
                    best_cut = cut
                    best_spins = spins.copy()

                if i % 20 == 0:
                    logger.info(f"Iter {i:03d}: Loss {loss.item():.4f}, Cut {cut:.0f}")

        logger.info(f"Gradient Descent Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut, loss_history

    def solve_iterative(self, max_iter: int = 50) -> Tuple[np.ndarray, float]:
        """
        Iterative method using fixed-point updates (Baseline).
        Uses A = -W for spectral properties.
        """
        logger.info("--- Phase C: Running Iterative Method (A = -W) ---")
        t0 = time.time()

        w_diag = np.ones(self.n)
        best_cut = -np.inf
        best_spins = None

        for it in range(max_iter):
            # Operator D^-1/2 * A * D^-1/2
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(w_diag + 1e-8))
            Op = D_inv_sqrt @ self.A_scipy @ D_inv_sqrt

            # Find Leading Eigenvector
            vals, vecs = eigsh(Op, k=1, which='LA', tol=1e-3)
            v = vecs[:, 0]

            spins = np.sign(v)
            spins[spins == 0] = 1
            cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)

            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()
                logger.info(f"Iter {it}: New Best Cut {cut:.0f}")

            # Heuristic update rule
            amplitude = np.abs(v) + 1e-6
            w_diag = 0.9 * w_diag + 0.1 * (1.0 / amplitude)
            w_diag = w_diag / np.mean(w_diag)

        logger.info(f"Iterative Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut


class ImprovedSpectralSolverWrapper:
    """Wrapper class to integrate ImprovedSpectralSolver with the pipeline"""

    def __init__(
        self,
        instance_name: str,
        dataset: str,
        random_seed: int = None,
        variant: str = 'grad',  # 'grad', 'sdp', or 'iter'
        lr: float = 0.05,
        steps: int = 100,
        max_iter: int = 50,
        n_rounding: int = 100,
        gpu: bool = True,
        run_name: str = None,
        **kwargs
    ):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.variant = variant
        self.lr = lr
        self.steps = steps
        self.max_iter = max_iter
        self.n_rounding = n_rounding
        self.gpu = gpu

        self.solver_name = run_name if run_name else "improved_spectral"

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)

        print(f"Initialized {self.solver_name} solver. Results will be saved under '{self.solver_name}'.")
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  variant: {self.variant}")
        print(f"  lr: {self.lr}")
        print(f"  steps: {self.steps}")
        print(f"  gpu: {self.gpu}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        """Solve the MaxCut problem using Improved Spectral Solver"""
        start_time = time.time()

        if not sp.issparse(J):
            J = sp.csr_matrix(J)

        # Pass J (or adjacency) directly to Solver
        device = 'cuda' if torch.cuda.is_available() and self.gpu else 'cpu'

        try:
            solver = ImprovedSpectralSolver(J, device=device)
        except ValueError as e:
            logger.error(f"Solver Initialization Failed: {e}")
            return # Exit safely

        logger.info(f"[*] Running Improved Spectral Solver with method: {self.variant}")

        if self.variant == 'sdp':
            spins, cut = solver.solve_sdp_proxy(n_rounding=self.n_rounding)
        elif self.variant == 'iter':
            spins, cut = solver.solve_iterative(max_iter=self.max_iter)
        else:
            # Default pipeline: SDP init -> Gradient Descent
            solver.solve_sdp_proxy(n_rounding=10) # Quick init
            spins, cut, _ = solver.solve_gradient_descent(lr=self.lr, steps=self.steps)

        time_taken = time.time() - start_time

        energy = -cut # Energy is negative Cut for MaxCut
        self._store_results(energy=energy, spins=spins, time_taken=time_taken, cut=cut)

    def _store_results(self, energy: float, spins: np.ndarray, time_taken: float, cut: float):
        """Store results to CSV file using pandas"""
        import pandas as pd
        from pathlib import Path
        from datetime import datetime

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
            'time': time_taken,
            'variant': self.variant,
            'lr': self.lr,
            'steps': self.steps,
            'max_iter': self.max_iter,
            'n_rounding': self.n_rounding,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Check existing and append/update
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()

        new_row = pd.DataFrame([result_data])

        if not df.empty:
            # Simple deduplication based on seed/solver
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
        logger.info(f"Energy: {energy:.6f}")
        logger.info(f"Cut: {cut:.0f}")
        logger.info(f"Time: {time_taken:.4f}s")
        logger.info("-" * 40)
