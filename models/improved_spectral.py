import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import torch
import torch.nn as nn
import logging
import time
from typing import Tuple, Union, Optional, List

logger = logging.getLogger("SpectralResearch")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class SpectralObjectives:
    @staticmethod
    def get_cut_value(adj_matrix: sp.spmatrix, spins: np.ndarray) -> float:
        spins = spins.reshape(-1)
        interaction = spins.dot(adj_matrix.dot(spins))
        total_weight = adj_matrix.sum()
        return 0.25 * (total_weight - interaction)

class ImprovedSpectralSolver(nn.Module):
    def __init__(self, adjacency_matrix: Union[sp.spmatrix, np.ndarray], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.n = adjacency_matrix.shape[0]
        self.device = device

        logger.info(f"Initializing Improved Solver on [{device}] for Graph N={self.n}")

        if sp.issparse(adjacency_matrix):
            adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
            self.adj_scipy = adjacency_matrix

            coo = adjacency_matrix.tocoo()
            indices = np.vstack((coo.row, coo.col))
            values = coo.data

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape

            self.adj_torch = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=self.device)
        else:
            self.adj_scipy = sp.csr_matrix(adjacency_matrix)
            self.adj_torch = torch.FloatTensor(adjacency_matrix).to(device)

        self.log_w = nn.Parameter(torch.zeros(self.n, device=self.device))

        self.register_buffer('cached_eigenvector', torch.randn(self.n, 1, device=self.device))
        self._normalize_cached_vector()

    def _normalize_cached_vector(self):
        with torch.no_grad():
            norm = torch.norm(self.cached_eigenvector) + 1e-8
            self.cached_eigenvector.data /= norm

    def _spectral_operator_mult(self, v: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
        d_inv_sqrt = torch.rsqrt(w_vec + 1e-6).unsqueeze(1)

        u = v * d_inv_sqrt

        z = torch.sparse.mm(self.adj_torch, u)

        out = z * d_inv_sqrt
        return out

    def _warm_start_power_iteration(self, w_vec: torch.Tensor, steps: int = 5) -> torch.Tensor:
        v = self.cached_eigenvector.detach().clone()

        for _ in range(steps):
            v = self._spectral_operator_mult(v, w_vec)
            v = v / (torch.norm(v) + 1e-8)

        return v

    def solve_sdp_proxy(self, n_rounding: int = 100) -> Tuple[np.ndarray, float]:
        logger.info("--- Phase A: Running SDP Proxy (Signed Laplacian) ---")
        t0 = time.time()

        abs_degrees = np.abs(self.adj_scipy).sum(axis=1).A1
        d_inv_sqrt = 1.0 / np.sqrt(abs_degrees + 1e-8)

        def matvec(v):
            v = v * d_inv_sqrt
            v = self.adj_scipy.dot(v)
            v = v * d_inv_sqrt
            return v

        op = LinearOperator((self.n, self.n), matvec=matvec)

        vals, vecs = eigsh(op, k=1, which='LA', tol=1e-4)
        v_sdp = vecs[:, 0]

        best_cut = -np.inf
        best_spins = None

        for _ in range(n_rounding):
            random_noise = np.random.randn(self.n) * 0.01
            spins = np.sign(v_sdp + random_noise)
            spins[spins == 0] = 1

            cut = SpectralObjectives.get_cut_value(self.adj_scipy, spins)
            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()

        self.cached_eigenvector.data = torch.tensor(v_sdp, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self._normalize_cached_vector()

        logger.info(f"SDP Proxy Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut

    def solve_gradient_descent(self, lr: float = 0.05, steps: int = 100, warm_start_steps: int = 5) -> Tuple[np.ndarray, float, List[float]]:
        logger.info(f"--- Phase B: Running Gradient Descent (Steps={steps}, LR={lr}) ---")
        optimizer = torch.optim.Adam([self.log_w], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        best_cut = -np.inf
        best_spins = None
        loss_history = []
        t0 = time.time()

        for i in range(steps):
            optimizer.zero_grad()
            w_vec = torch.exp(self.log_w)

            v_approx = self._warm_start_power_iteration(w_vec, steps=warm_start_steps)

            self.cached_eigenvector.data = v_approx.data

            v_fixed = v_approx.detach()

            numerator = (v_fixed.T @ self._spectral_operator_mult(v_fixed, w_vec)).squeeze()

            loss = -numerator

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            if i % 10 == 0 or i == steps - 1:
                current_v_cpu = v_approx.detach().cpu().numpy().flatten()
                spins = np.sign(current_v_cpu)
                spins[spins == 0] = 1
                cut = SpectralObjectives.get_cut_value(self.adj_scipy, spins)

                if cut > best_cut:
                    best_cut = cut
                    best_spins = spins.copy()

                if i % 20 == 0:
                    logger.info(f"Iter {i:03d}: Loss {loss.item():.4f}, Cut {cut:.0f}")

        logger.info(f"Gradient Descent Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut, loss_history

    def solve_iterative(self, max_iter: int = 50) -> Tuple[np.ndarray, float]:
        logger.info("--- Phase C: Running Iterative Method ---")
        t0 = time.time()
        W = np.ones(self.n)
        best_cut = -np.inf
        best_spins = None

        for it in range(max_iter):
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(W + 1e-8))
            Op = D_inv_sqrt @ self.adj_scipy @ D_inv_sqrt

            vals, vecs = eigsh(Op, k=1, which='LA', tol=1e-3)
            v = vecs[:, 0]

            spins = np.sign(v)
            spins[spins == 0] = 1
            cut = SpectralObjectives.get_cut_value(self.adj_scipy, spins)

            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()

            amplitude = np.abs(v) + 1e-6
            W = 0.9 * W + 0.1 * (1.0 / amplitude)
            W = W / np.mean(W)

        logger.info(f"Iterative Time: {time.time()-t0:.2f}s | Best Cut: {best_cut:.0f}")
        return best_spins, best_cut
