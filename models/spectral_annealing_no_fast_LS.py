import numpy as np
import time
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from math import log
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_splinalg

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class QOperator(cpx_splinalg.LinearOperator):
    def __init__(self, J_csr, D_vec, beta):
        self.J_csr = J_csr
        self.D_vec = D_vec
        self.beta = beta
        n = D_vec.size
        super().__init__(dtype=D_vec.dtype, shape=(n, n))

    def _matvec(self, x):
        return self.beta * self.D_vec * x - self.J_csr.dot(x)


class SOperator(cpx_splinalg.LinearOperator):
    def __init__(self, Q_op, D_inv_sqrt):
        self.Q_op = Q_op
        self.D_inv_sqrt = D_inv_sqrt
        n = D_inv_sqrt.size
        super().__init__(dtype=D_inv_sqrt.dtype, shape=(n, n))

    def _matvec(self, y):
        z = self.D_inv_sqrt * y
        w = self.Q_op._matvec(z)
        return self.D_inv_sqrt * w




class SpectralAnnealingV4:
    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 mode: int = 0, beta: float = None, alpha: float = None,
                 use_local_search: bool = True, max_local_iterations: int = 1000,
                 eigsh_tol: float = 1e-6, eigsh_maxiter: int = 1000, run_name: str = None, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.mode = mode
        self.beta = beta
        self.alpha = alpha
        self.use_local_search = use_local_search
        self.max_local_iterations = max_local_iterations
        self.eigsh_tol = float(eigsh_tol)  
        self.eigsh_maxiter = int(eigsh_maxiter)
        
        self.solver_name = run_name if run_name else "spectral_annealing"
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            
        print(f"Initialized {self.solver_name} solver (GPU sparse). Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  mode: {self.mode}")
        print(f"  beta: {self.beta}")
        print(f"  alpha: {self.alpha}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  max_local_iterations: {self.max_local_iterations}")
        print(f"  eigsh_tol: {self.eigsh_tol}")
        print(f"  eigsh_maxiter: {self.eigsh_maxiter}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    
    def solve(self, J: np.ndarray):
        """Solve the Ising problem"""
        start_time = time.time()
        
        J_csr = cpx_sparse.csr_matrix(J)  

        J_abs = J_csr.copy()
        J_abs.data = cp.abs(J_abs.data)

        deg_abs_original = J_abs @ cp.ones(J_csr.shape[1], dtype=cp.float64)

        max_deg_abs = cp.max(deg_abs_original)

        self._normalization_factor = float(max_deg_abs)

        J_csr = J_csr / max_deg_abs
        deg_abs_normalized = deg_abs_original / max_deg_abs

        if self.alpha is None:
            alpha_list = self._get_alpha_values(self.mode)
            best_energy, best_spins, best_alpha, best_beta, best_local_iterations = float("inf"), None, None, None, 0
            alpha_results = []  
            prev_v0_y = None
            
            for alpha in alpha_list:
                try:
                    alpha_start_time = time.time()
                    self.alpha = alpha
                    
                    D_vec = cp.power(deg_abs_normalized, float(alpha))

                    beta = self.beta if self.beta is not None else self._calculate_beta(deg_abs_normalized)
                    
                    # Solve eigenvalue problem
                    evals, evecs, evecs_y = self._solve_eigenvalue_problem(J_csr, D_vec, beta, k=1, v0=prev_v0_y)
                    prev_v0_y = evecs_y[:, 0]
                    
                    # Rounding
                    spins = cp.where(evecs[:, 0] >= 0, 1.0, -1.0)
                    
                    energy = self.compute_energy(spins, J_csr)
                    
                    local_iterations = 0
                    
                    if self.use_local_search:
                        J_csr_float64 = cpx_sparse.csr_matrix(J_csr.astype(cp.float64), dtype=cp.float64)
                        
                        refined_spins_batch, local_iterations = flip_refinement(
                            J_csr_float64, [spins], 
                            max_iterations=self.max_local_iterations
                        )
                        
                        refined_spins = cp.sign(refined_spins_batch[:, 0].astype(cp.float64))
                        refined_spins[refined_spins == 0] = 1.0
                        refined_energy = self.compute_energy(refined_spins, J_csr)
                        
                        if refined_energy < energy:
                            spins = refined_spins
                            energy = refined_energy
                    
                    alpha_time = time.time() - alpha_start_time
                    
                    alpha_results.append({
                        'instance_name': self.instance_name,
                        'dataset': self.dataset,
                        'seed': self.random_seed,
                        'solver_name': self.solver_name,
                        'alpha': alpha,
                        'beta': beta,
                        'energy': energy,
                        'time': alpha_time,
                        'mode': self.mode,
                        'use_local_search': self.use_local_search,
                        'local_search_iterations': local_iterations,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if energy < best_energy:
                        best_energy, best_spins, best_alpha, best_beta = energy, spins, alpha, beta
                        best_local_iterations = local_iterations
                        
                except Exception as e:
                    print(f"Alpha {alpha} failed: {e}")
                    continue

            total_time = time.time() - start_time
            
            self._store_general_results(best_energy, best_alpha, best_beta, total_time, best_local_iterations)
            self._store_timing_results(alpha_results)
            
        else:
            raise ValueError("Alpha is not set")

    def _get_alpha_values(self, mode: int) -> list:   
        if mode == 0:
            return [float(x) for x in cp.sqrt(cp.linspace(0, 1, 128))]
        else:
            return [float(x) for x in cp.linspace(0, 2, 201)]

    def _calculate_beta(self, deg_abs: "cp.ndarray") -> float:
        beta = float((1.0 + 1e-8) * cp.max(deg_abs ** (1.0 - self.alpha)))
        return beta

    def _solve_eigenvalue_problem(self, J_csr, D_vec, beta, k=1, v0=None):   
        n = D_vec.size

        D_work = cp.maximum(D_vec, 1e-12)
        D_inv_sqrt = cp.reciprocal(cp.sqrt(D_work))

        if not hasattr(self, "_Q_op") or not hasattr(self, "_S_op"):
            self._Q_op = QOperator(J_csr, D_vec, beta)
            self._S_op = SOperator(self._Q_op, D_inv_sqrt)
        else:
            self._Q_op.D_vec = D_vec
            self._Q_op.beta = beta
            self._S_op.D_inv_sqrt = D_inv_sqrt

        evals, evecs_y = cpx_splinalg.eigsh(
            self._S_op, k=k, which="SA",
            tol=self.eigsh_tol, maxiter=self.eigsh_maxiter, v0=v0
        )

        X = evecs_y * D_inv_sqrt[:, cp.newaxis]

        return evals[:k], X[:, :k], evecs_y[:, :k]


    def compute_energy(self, spins: "cp.ndarray", J_csr) -> float:
        if hasattr(J_csr, 'dot'):
            energy = -0.5 * cp.dot(spins, J_csr.dot(spins))
        else:
            energy = -0.5 * cp.einsum("i,ij,j->", spins, J_csr, spins)

        if hasattr(self, '_normalization_factor'):
            energy *= self._normalization_factor
        return float(energy)


    def _store_general_results(self, energy: float, alpha: float, beta: float, time_taken: float, local_search_iterations: int = 0):
        try:
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
                'time': time_taken,
                'local_search': self.use_local_search,
                'local_search_iterations': local_search_iterations,
                'alpha': alpha,
                'beta': beta,
                'mode': self.mode,
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
            print(f"General results saved to: {csv_file}")
        except Exception as e:
            print(f"Error saving general results: {e}")

    def _store_timing_results(self, alpha_results: list):
        if not alpha_results:
            return
            
        alphas_results_dir = Path(f"results/{self.dataset}/{self.solver_name}/alphas")
        alphas_results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '')
        csv_file = alphas_results_dir / f"{instance_base}_alphas.csv"
        
        df_new = pd.DataFrame(alpha_results)
        
        if csv_file.exists():
            df_existing = pd.read_csv(csv_file)
            
            duplicate_rows = []
            for _, new_row in df_new.iterrows():
                duplicate_mask = (
                    (df_existing['instance_name'] == new_row['instance_name']) &
                    (df_existing['dataset'] == new_row['dataset']) &
                    (df_existing['seed'] == new_row['seed']) &
                    (df_existing['solver_name'] == new_row['solver_name']) &
                    (df_existing['alpha'] == new_row['alpha'])
                )
                
                if duplicate_mask.any():
                    for col in df_new.columns:
                        df_existing.loc[duplicate_mask, col] = new_row[col]
                else:
                    duplicate_rows.append(new_row)
            
            if duplicate_rows:
                df_new_rows = pd.DataFrame(duplicate_rows)
                df = pd.concat([df_existing, df_new_rows], ignore_index=True)
            else:
                df = df_existing
        else:
            df = df_new
        
        df = df.sort_values(['seed', 'alpha']).reset_index(drop=True)
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Alpha results saved to: {csv_file} ({len(alpha_results)} alpha values)")




# ============================================================
# Unified CUDA kernel: spin updates on selected nodes
# ============================================================
_update_spins_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* indptr,
    const int* indices,
    const float* data,
    signed char* spins,
    int* improvement_counts)
{
    int lane = threadIdx.x;
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;  // node index in selected_nodes
    int s = blockIdx.y;                                   // solution id

    if (warp_id >= n_selected || s >= k) return;

    int node = selected_nodes[warp_id];
    int row_start = indptr[node];
    int row_end   = indptr[node+1];

    float local_sum = 0.0f;

    // Partial sum across warp
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        int j = indices[idx];
        int idx_n = s * n + j;
        local_sum += data[idx] * (float)spins[idx_n];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Lane 0 decides whether to flip
    if (lane == 0) {
        int idx_spin = s * n + node;
        signed char si = spins[idx_spin];
        float delta_E = 2.0f * (float)si * local_sum;
        if (delta_E < 0.0f) {
            spins[idx_spin] = -si;
            atomicAdd(&improvement_counts[s], 1);
        }
    }
}
''', "update_spins_kernel")



# ============================================================
# GPU Final Sweep (multi-round, reshuffled)
# ============================================================
def gpu_final_sweep(J_csr,
                    spins_flat: cp.ndarray,
                    n: int,
                    k: int,
                    node_fraction: float = 0.5,
                    max_sub_iters: int = 5,
                    num_rounds: int = 3):

    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(cp.float32)

    warp_size = 32
    warps_per_block = 4
    threads_per_block = (warp_size, warps_per_block)

    for round_idx in range(num_rounds):
        num_to_select = max(1, int(n * node_fraction))
        node_indices = cp.arange(n, dtype=cp.int32)
        node_indices = cp.random.permutation(node_indices)
        selected_nodes = node_indices[:num_to_select]

        num_warps_x = (num_to_select + warps_per_block - 1) // warps_per_block
        grid = (num_warps_x, k)
        improvement_counts = cp.zeros(k, dtype=cp.int32)

        for sub_iter in range(max_sub_iters):
            improvement_counts.fill(0)
            
            _update_spins_kernel(
                grid, threads_per_block,
                (num_to_select, n, k,
                 selected_nodes, indptr, indices, data,
                 spins_flat, improvement_counts)
            )
            
            if not cp.any(improvement_counts > 0):
                break

    return spins_flat



# ============================================================
# Local search refinement with random subset each iteration
# ============================================================
def flip_refinement(J_csr, spins_list, max_iterations=1000,
                    final_sweep_fraction=1.0, final_sweep_iters=25,
                    num_rounds=25, subset_fraction=0.9):

    Spins = cp.stack([sp.astype(cp.int8).copy() for sp in spins_list], axis=1)
    n, k = Spins.shape
    spins_flat = Spins.T.reshape(-1).copy()

    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(cp.float32)

    improvement_counts = cp.zeros(k, dtype=cp.int32)
    active_mask = cp.ones(k, dtype=cp.bool_)
    max_iters_used = 0

    warp_size = 32
    warps_per_block = 4
    threads_per_block = (warp_size, warps_per_block)

    for it in range(max_iterations):
        max_iters_used += 1
        improvement_counts.fill(0)

        num_selected = max(1, int(subset_fraction * n))
        selected_nodes = cp.random.permutation(n).astype(cp.int32)[:num_selected]

        num_warps_x = (num_selected + warps_per_block - 1) // warps_per_block
        grid = (num_warps_x, k)

        _update_spins_kernel(
            grid, threads_per_block,
            (num_selected, n, k,
             selected_nodes, indptr, indices, data,
             spins_flat, improvement_counts)
        )

        no_improvement = (improvement_counts == 0)
        active_mask &= ~no_improvement

        if not cp.any(active_mask):
            break

    spins_flat = gpu_final_sweep(
        J_csr, spins_flat, n, k,
        node_fraction=final_sweep_fraction,
        max_sub_iters=final_sweep_iters,
        num_rounds=num_rounds
    )

    Spins_final = spins_flat.reshape(k, n).T.astype(cp.int8)

    return Spins_final, max_iters_used