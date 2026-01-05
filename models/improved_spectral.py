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
            # FIX: Symmetrize without halving weights (for Gset format)
            # Gset format stores edges once (upper triangular), need A + A.T not 0.5*(A + A.T)
            diff = (adjacency_matrix - adjacency_matrix.T)
            if diff.nnz > 0:
                logger.info("Graph is not symmetric. Symmetrizing (A = A + A.T)...")
                tri_u = sp.triu(adjacency_matrix, k=1)
                adj_sym = tri_u + tri_u.T
            else:
                adj_sym = adjacency_matrix

            # W: Positive adjacency for MaxCut (used for cut calculation)
            self.W_scipy = adj_sym
            # A: Negative matrix for spectral operations (to get alternating sign eigenvector)
            self.A_scipy = -adj_sym

            # PyTorch tensor uses A (negative) for spectral operations
            coo = self.A_scipy.tocoo()
            indices = np.vstack((coo.row, coo.col))
            values = coo.data

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape

            self.adj_torch = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=self.device)
        else:
            adj_sym = np.array(adjacency_matrix)
            self.W_scipy = sp.csr_matrix(adj_sym)
            self.A_scipy = sp.csr_matrix(-adj_sym)
            self.adj_torch = torch.FloatTensor(-adj_sym).to(device)

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

        # Use W (positive) for degree calculation
        abs_degrees = np.abs(self.W_scipy).sum(axis=1).A1
        d_inv_sqrt = 1.0 / np.sqrt(abs_degrees + 1e-8)

        def matvec(v):
            v = v * d_inv_sqrt
            # Use A (negative) for spectral operation to get alternating sign eigenvector
            v = self.A_scipy.dot(v)
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

            # Use W (positive) for cut calculation
            cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)
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
                # Use W (positive) for cut calculation
                cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)

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
            # Use A (negative) for spectral operation to get alternating sign eigenvector
            Op = D_inv_sqrt @ self.A_scipy @ D_inv_sqrt

            vals, vecs = eigsh(Op, k=1, which='LA', tol=1e-3)
            v = vecs[:, 0]

            spins = np.sign(v)
            spins[spins == 0] = 1
            # Use W (positive) for cut calculation
            cut = SpectralObjectives.get_cut_value(self.W_scipy, spins)

            if cut > best_cut:
                best_cut = cut
                best_spins = spins.copy()

            amplitude = np.abs(v) + 1e-6
            W = 0.9 * W + 0.1 * (1.0 / amplitude)
            W = W / np.mean(W)

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
        print(f"  random_seed: {self.random_seed}")
        print(f"  variant: {self.variant}")
        print(f"  lr: {self.lr}")
        print(f"  steps: {self.steps}")
        print(f"  max_iter: {self.max_iter}")
        print(f"  n_rounding: {self.n_rounding}")
        print(f"  gpu: {self.gpu}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        """Solve the MaxCut problem using Improved Spectral Solver"""
        import pandas as pd
        from pathlib import Path
        from datetime import datetime

        start_time = time.time()

        # Convert to sparse if needed
        if not sp.issparse(J):
            J = sp.csr_matrix(J)

        # Initialize solver
        device = 'cuda' if torch.cuda.is_available() and self.gpu else 'cpu'
        solver = ImprovedSpectralSolver(J, device=device)

        logger.info(f"[*] Running Improved Spectral Solver with method: {self.variant}")

        # Run appropriate method
        if self.variant == 'sdp':
            spins, cut = solver.solve_sdp_proxy(n_rounding=self.n_rounding)
        elif self.variant == 'iter':
            spins, cut = solver.solve_iterative(max_iter=self.max_iter)
        else:  # 'grad' or default
            # Warm start with SDP
            solver.solve_sdp_proxy(n_rounding=10)
            spins, cut, _ = solver.solve_gradient_descent(lr=self.lr, steps=self.steps)

        time_taken = time.time() - start_time

        # Convert cut to energy (energy = -cut for MaxCut)
        energy = -cut

        # Store results
        self._store_results(
            energy=energy,
            spins=spins,
            time_taken=time_taken,
            cut=cut
        )

    def _store_results(self, energy: float, spins: np.ndarray, time_taken: float, cut: float):
        """Store results to CSV file"""
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

        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()

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
                df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        else:
            df = pd.DataFrame([result_data])

        df.to_csv(csv_file, index=False, encoding='utf-8')

        logger.info(f"Results saved to: {csv_file}")
        logger.info(f"Energy: {energy:.6f}")
        logger.info(f"Cut: {cut:.0f}")
        logger.info(f"Time: {time_taken:.4f}s")
        logger.info("-" * 40)
