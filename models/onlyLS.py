import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# Unified CUDA kernel: spin updates on selected nodes (float32)
# ============================================================
_update_spins_kernel_float32 = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel_float32(
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
''', "update_spins_kernel_float32")

# ============================================================
# Unified CUDA kernel: spin updates on selected nodes (float64)
# ============================================================
_update_spins_kernel_float64 = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel_float64(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* indptr,
    const int* indices,
    const double* data,
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

    double local_sum = 0.0;

    // Partial sum across warp
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        int j = indices[idx];
        int idx_n = s * n + j;
        local_sum += data[idx] * (double)spins[idx_n];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Lane 0 decides whether to flip
    if (lane == 0) {
        int idx_spin = s * n + node;
        signed char si = spins[idx_spin];
        double delta_E = 2.0 * (double)si * local_sum;
        if (delta_E < 0.0) {
            spins[idx_spin] = -si;
            atomicAdd(&improvement_counts[s], 1);
        }
    }
}
''', "update_spins_kernel_float64")


# ============================================================
# GPU Final Sweep (multi-round, reshuffled)
# ============================================================
def gpu_final_sweep(J_csr,
                    spins_flat: cp.ndarray,
                    n: int,
                    k: int,
                    node_fraction: float = 0.5,
                    max_sub_iters: int = 5,
                    num_rounds: int = 3,
                    dtype=cp.float32):
    
    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(dtype)
    
    # Select appropriate kernel based on dtype
    if dtype == cp.float32:
        kernel = _update_spins_kernel_float32
    elif dtype == cp.float64:
        kernel = _update_spins_kernel_float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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
            
            kernel(
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
                    num_rounds=25, subset_fraction=0.9, dtype=cp.float32):

    Spins = cp.stack([sp.astype(cp.int8).copy() for sp in spins_list], axis=1)
    n, k = Spins.shape
    spins_flat = Spins.T.reshape(-1).copy()

    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(dtype)
    
    # Select appropriate kernel based on dtype
    if dtype == cp.float32:
        kernel = _update_spins_kernel_float32
    elif dtype == cp.float64:
        kernel = _update_spins_kernel_float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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

        kernel(
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
        num_rounds=num_rounds,
        dtype=dtype
    )

    Spins_final = spins_flat.reshape(k, n).T.astype(cp.int8)

    return Spins_final, max_iters_used


class OnlyLS:
    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 n_random_solutions: int = 128, max_local_iterations: int = 1000,
                 precision: int = 32, run_name: str = None, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.n_random_solutions = n_random_solutions
        self.max_local_iterations = max_local_iterations
        
        # Set precision (32 or 64 bits)
        if precision == 32:
            self.dtype = cp.float32
        elif precision == 64:
            self.dtype = cp.float64
        else:
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.precision = precision
        
        self.solver_name = run_name if run_name else "onlyLS"
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            self.random_seed = random_seed
            
        print(f"Initialized {self.solver_name} solver (GPU local search only). Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  n_random_solutions: {self.n_random_solutions}")
        print(f"  max_local_iterations: {self.max_local_iterations}")
        print(f"  precision: {self.precision} bits ({self.dtype})")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    
    def solve(self, J: np.ndarray):
        start_time = time.time()

        n = J.shape[0]

        # Convert J to the specified precision
        J_csr = cpx_sparse.csr_matrix(J.astype(self.dtype), dtype=self.dtype)

        J_abs = J_csr.copy()
        J_abs.data = cp.abs(J_abs.data)

        deg_abs_original = J_abs @ cp.ones(J_csr.shape[1], dtype=self.dtype)

        max_deg_abs = cp.max(deg_abs_original)

        self._normalization_factor = float(max_deg_abs)

        J_csr = J_csr / max_deg_abs

        # Generate random solutions
        print(f"Generating {self.n_random_solutions} random solutions...")
        if self.random_seed is not None:
            cp.random.seed(self.random_seed)
        
        # Generate all random solutions at once
        random_spins_list = []
        for i in range(self.n_random_solutions):
            spins = cp.random.choice([-1.0, 1.0], size=n).astype(self.dtype)
            random_spins_list.append(spins)
        
        print(f"Applying local search to {self.n_random_solutions} solutions...")
        
        # Apply local search to all solutions
        J_csr_precision = cpx_sparse.csr_matrix(J_csr.astype(self.dtype), dtype=self.dtype)
        
        refined_spins_batch, local_iterations = flip_refinement(
            J_csr_precision, random_spins_list, 
            max_iterations=self.max_local_iterations,
            dtype=self.dtype
        )
        
        # Find best solution
        best_energy = float("inf")
        best_spins = None
        
        for i in range(self.n_random_solutions):
            spins = cp.sign(refined_spins_batch[:, i].astype(self.dtype))
            spins[spins == 0] = 1.0
            energy = self.compute_energy(spins, J_csr)
            
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()

        total_time = time.time() - start_time
        
        self._store_results(best_energy, total_time, local_iterations)
        
        print(f"Best energy: {best_energy:.6f}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Local search iterations: {local_iterations}")
        print("-" * 40)

    def compute_energy(self, spins: "cp.ndarray", J_csr) -> float:
        if hasattr(J_csr, 'dot'):
            energy = -0.5 * cp.dot(spins, J_csr.dot(spins))
        else:
            energy = -0.5 * cp.einsum("i,ij,j->", spins, J_csr, spins)

        if hasattr(self, '_normalization_factor'):
            energy *= self._normalization_factor
        return float(energy)

    def _store_results(self, energy: float, time_taken: float, local_search_iterations: int = 0):
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
                'local_search': True,
                'local_search_iterations': local_search_iterations,
                'n_random_solutions': self.n_random_solutions,
                'max_local_iterations': self.max_local_iterations,
                'precision': self.precision,
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
            print(f"Results saved to: {csv_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

