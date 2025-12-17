from math import log
import numpy as np
import time
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_splinalg

from utils.predictors import (
    diagonal_predictor,
    secant_predictor,
    hybrid_predictor,
    subspace_predictor
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# V2 OPERATORS: N_alpha and S_rho for spectral-radius-based shift
# ============================================================

class NAlphaOperator(cpx_splinalg.LinearOperator):
    """
    NEW: Operator N^α(v) = D^{-1/2} A (D^{-1/2} v)
    Replaces the old QOperator + SOperator approach.
    """
    def __init__(self, J_csr, D_vec, dtype):
        self.J_csr = J_csr
        self.D_vec = D_vec
        self.D_inv_sqrt = cp.reciprocal(cp.sqrt(cp.maximum(D_vec, cp.array(1e-12, dtype=dtype))))
        n = D_vec.size
        super().__init__(dtype=dtype, shape=(n, n))

    def _matvec(self, v):
        z = self.D_inv_sqrt * v
        w = self.J_csr.dot(z)
        return self.D_inv_sqrt * w


class SRhoOperator(cpx_splinalg.LinearOperator):
    """
    Operator S_ρ(v) = ρv - N^α(v)
    This is SPD when ρ > λ₁(N^α), enabling efficient Lanczos.
    """
    def __init__(self, N_op, rho):
        self.N_op = N_op
        self.rho = rho
        n = N_op.shape[0]
        super().__init__(dtype=N_op.dtype, shape=(n, n))

    def _matvec(self, v):
        return self.rho * v - self.N_op(v)


# ============================================================
# V2 SHIFT STRATEGY: Spectral-radius-based shift
# ============================================================

def compute_shift_rho(lambda1_prev, gershgorin_bound, eps=1e-3):
    """
    NEW: Compute shift ρ_k = (1+eps) * max(lambda1_prev, gershgorin_bound)
    """
    if lambda1_prev is None or lambda1_prev <= 0:
        base = gershgorin_bound
    else:
        base = max(lambda1_prev, gershgorin_bound)

    rho = (1.0 + eps) * base
    if rho <= 0:
        rho = (1.0 + eps) * max(gershgorin_bound, 1.0)
    return rho


# ============================================================
# LEGACY OPERATORS (kept for reference, not used in V2)
# ============================================================

# class QOperator(cpx_splinalg.LinearOperator):
#     """
#     OLD IMPLEMENTATION: Kept for reference only, not used in V2
#     """
#     def __init__(self, J_csr, D_vec, beta):
#         self.J_csr = J_csr
#         self.D_vec = D_vec
#         self.beta = beta
#         n = D_vec.size
#         super().__init__(dtype=D_vec.dtype, shape=(n, n))

#     def _matvec(self, x):
#         return self.beta * self.D_vec * x - self.J_csr.dot(x)


# class SOperator(cpx_splinalg.LinearOperator):
#     """
#     OLD IMPLEMENTATION: Kept for reference only, not used in V2
#     """
#     def __init__(self, Q_op, D_inv_sqrt):
#         self.Q_op = Q_op
#         self.D_inv_sqrt = D_inv_sqrt
#         n = D_inv_sqrt.size
#         super().__init__(dtype=D_inv_sqrt.dtype, shape=(n, n))

#     def _matvec(self, y):
#         z = self.D_inv_sqrt * y
#         w = self.Q_op._matvec(z)
#         return self.D_inv_sqrt * w


# ============================================================
# SPECTRAL ANNEALING V2: Main solver class
# ============================================================

class SpectralAnnealingV2:
    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 mode: int = 0, beta: float = None, alpha: float = None,
                 use_local_search: bool = True, max_local_iterations: int = 1000,
                 eigsh_tol: float = 1e-6, eigsh_maxiter: int = 1000, run_name: str = None,
                 disable_warm_start: bool = False, precision: int = 32,
                 pred_mode: str = "hybrid", pred_tol: float = 1e-3, shift_eps: float = 1e-3,
                 **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.mode = mode
        self.beta = beta  # Not used in V2, kept for compatibility
        self.alpha = alpha
        self.use_local_search = use_local_search
        self.max_local_iterations = max_local_iterations
        self.eigsh_tol = float(eigsh_tol)
        self.eigsh_maxiter = int(eigsh_maxiter)
        self.disable_warm_start = disable_warm_start
        
        # V2 NEW PARAMETERS
        self.pred_mode = pred_mode  # "diagonal", "secant", "hybrid", "subspace"
        self.pred_tol = float(pred_tol)  # Residual tolerance for predictor validation
        self.shift_eps = float(shift_eps)  # Epsilon for shift calculation
        
        # Set precision (32 or 64 bits)
        if precision == 32:
            self.dtype = cp.float32
        elif precision == 64:
            self.dtype = cp.float64
        else:
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.precision = precision
        
        self.solver_name = run_name if run_name else "spectral_annealing_v2"
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            self.random_seed = random_seed
            
        print(f"Initialized {self.solver_name} solver (GPU sparse, V2 with predictors). Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  mode: {self.mode}")
        print(f"  beta: {self.beta} (not used in V2)")
        print(f"  alpha: {self.alpha}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  max_local_iterations: {self.max_local_iterations}")
        print(f"  eigsh_tol: {self.eigsh_tol}")
        print(f"  eigsh_maxiter: {self.eigsh_maxiter}")
        print(f"  disable_warm_start: {self.disable_warm_start}")
        print(f"  precision: {self.precision} bits ({self.dtype})")
        print(f"  pred_mode: {self.pred_mode} (NEW)")
        print(f"  pred_tol: {self.pred_tol} (NEW)")
        print(f"  shift_eps: {self.shift_eps} (NEW)")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        start_time = time.time()

        n = J.shape[0]
        print(f"\n[DEBUG solve] Starting solve for instance: {self.instance_name}")
        print(f"  [Input] J.shape={J.shape}, J.dtype={J.dtype}")
        print(f"  [Input] n={n}, precision={self.precision} bits ({self.dtype})")

        # Convert J to the specified precision
        J_csr = cpx_sparse.csr_matrix(J.astype(self.dtype), dtype=self.dtype)

        J_abs = J_csr.copy()
        J_abs.data = cp.abs(J_abs.data)

        deg_abs_original = J_abs @ cp.ones(J_csr.shape[1], dtype=self.dtype)

        max_deg_abs = cp.max(deg_abs_original)

        self._normalization_factor = float(max_deg_abs)
        print(f"  [Normalization] max_deg_abs={self._normalization_factor:.6f}")

        J_csr = J_csr / max_deg_abs
        deg_abs_normalized = deg_abs_original / max_deg_abs
        print(f"  [Normalization] deg_abs_normalized: min={float(cp.min(deg_abs_normalized)):.6f}, max={float(cp.max(deg_abs_normalized)):.6f}, mean={float(cp.mean(deg_abs_normalized)):.6f}")

        if self.alpha is None:
            # Ensure same random seed for fair comparison
            if self.random_seed is not None:
                cp.random.seed(self.random_seed)
            
            alpha_list = self._get_alpha_values(self.mode)
            print(f"  [Alpha schedule] mode={self.mode}, num_alphas={len(alpha_list)}")
            print(f"    First 5 alphas: {[f'{a:.6f}' for a in alpha_list[:5]]}")
            print(f"    Last 5 alphas: {[f'{a:.6f}' for a in alpha_list[-5:]]}")
            
            best_energy, best_spins, best_alpha, best_beta, best_local_iterations = float("inf"), None, None, None, 0
            alpha_results = []
            
            unique_solutions = {}
            alpha_to_solution_hash = {}
            
            # V2: Track previous eigenvectors and eigenvalues for predictors
            y_last = None
            y_prev = None
            lambda1_prev = None
            alpha_prev = None
            
            for alpha_idx, alpha in enumerate(alpha_list):
                print(f"\n[DEBUG solve] Alpha iteration {alpha_idx+1}/{len(alpha_list)}: alpha={alpha:.6f}")
                try:
                    alpha_start_time = time.time()
                    self.alpha = alpha
                    
                    D_vec = cp.power(deg_abs_normalized, float(alpha))
                    print(f"  [D_vec] shape={D_vec.shape}, min={float(cp.min(D_vec)):.6f}, max={float(cp.max(D_vec)):.6f}, sum={float(cp.sum(D_vec)):.6f}")

                    # OLD IMPLEMENTATION :
                    # beta = self.beta if self.beta is not None else self._calculate_beta(deg_abs_normalized)
                    # v0_input = None if self.disable_warm_start else prev_v0_y
                    # evals, evecs, evecs_y, iter_count = self._solve_eigenvalue_problem(J_csr, D_vec, beta, k=1, v0=v0_input)
                    
                    # V2 with predictors and spectral-radius shift
                    print(f"  [Tracking] y_last={'None' if y_last is None else f'shape={y_last.shape}, norm={cp.linalg.norm(y_last):.6f}'}")
                    print(f"  [Tracking] y_prev={'None' if y_prev is None else f'shape={y_prev.shape}, norm={cp.linalg.norm(y_prev):.6f}'}")
                    print(f"  [Tracking] lambda1_prev={lambda1_prev}, alpha_prev={alpha_prev}")
                    
                    evals, evecs, evecs_y, iter_count, lambda1 = self._solve_eigenvalue_problem_v2(
                        J_csr, D_vec, alpha_idx, alpha, alpha_prev,
                        y_last, y_prev, lambda1_prev, deg_abs_normalized
                    )
                    
                    print(f"  [Eigenvalue result] lambda1={lambda1:.6f}, eigsh_iterations={iter_count}")
                    print(f"    evals.shape={evals.shape}, evecs.shape={evecs.shape}, evecs_y.shape={evecs_y.shape}")
                    print(f"    evecs.norm={cp.linalg.norm(evecs[:, 0]):.6f}, evecs_y.norm={cp.linalg.norm(evecs_y[:, 0]):.6f}")

                    degree_scale = float(cp.sum(D_vec))
                    normalization_factor = float(self._normalization_factor)
                    bound_rescaled = -0.5 * lambda1 * degree_scale * normalization_factor

                    print(f"  [Bound calculation] degree_scale_real={degree_scale:.6f}")
                    print(f"    lambda1={lambda1:.6f}, bound_rescaled={bound_rescaled:.6f}")


                    # V2: Update tracking variables for next iteration
                    if not self.disable_warm_start:
                        # Guardamos el anterior eigenvector como y_prev
                        y_prev = y_last

                        # Nuevo eigenvector en el espacio y (el que usa N^α)
                        y_new = evecs_y[:, 0]

                        # Normalización explícita para que el warm start esté siempre en la esfera unidad
                        y_new_norm = cp.linalg.norm(y_new)
                        print(f"  [Tracking update] Before normalization: y_new.norm={y_new_norm:.6f}")
                        if y_new_norm > 0:
                            y_new = y_new / y_new_norm

                        # Actualizamos y_last con el vector normalizado
                        y_last = y_new

                        # Actualizamos info escalar
                        lambda1_prev = lambda1
                        alpha_prev = alpha
                        print(f"  [Tracking update] After update:")
                        print(f"    y_last.norm={cp.linalg.norm(y_last):.6f}, y_prev={'None' if y_prev is None else f'norm={cp.linalg.norm(y_prev):.6f}'}")
                        print(f"    lambda1_prev={lambda1_prev:.6f}, alpha_prev={alpha_prev:.6f}")

                    
                    spins = cp.where(evecs[:, 0] >= 0, cp.array(1.0, dtype=self.dtype), cp.array(-1.0, dtype=self.dtype))
                    num_pos = int(cp.sum(spins > 0))
                    num_neg = int(cp.sum(spins < 0))
                    print(f"  [Spins] num_pos={num_pos}, num_neg={num_neg}, total={n}")
                    
                    # Compute energy before local search (raw energy)
                    raw_energy = self.compute_energy(spins, J_csr)
                    # Check for invalid energy values
                    if np.isinf(raw_energy) or np.isnan(raw_energy):
                        print(f"Warning: Invalid energy {raw_energy} for alpha={alpha}, skipping this iteration")
                        continue
                    energy = raw_energy  # Will be updated after local search if applied
                    print(f"  [Energy] raw_energy={raw_energy:.6f}, bound={bound_rescaled:.6f}, gap={raw_energy - bound_rescaled:.6f}")
                    
                    local_iterations = 0
                    
                    if self.use_local_search:
                        spins_hash = self._compute_spin_signature(spins)
                        
                        if spins_hash not in unique_solutions:
                            unique_solutions[spins_hash] = (spins.copy(), [len(alpha_results)])
                        else:
                            unique_solutions[spins_hash][1].append(len(alpha_results))
                        
                        alpha_to_solution_hash[len(alpha_results)] = spins_hash
                    
                    alpha_time = time.time() - alpha_start_time
                    
                    alpha_results.append({
                        'instance_name': self.instance_name,
                        'dataset': self.dataset,
                        'seed': self.random_seed,
                        'solver_name': self.solver_name,
                        'alpha': alpha,
                        'beta': None,  # Not used in V2
                        'energy': energy,
                        'raw_energy': raw_energy,
                        'time': alpha_time,
                        'mode': self.mode,
                        'use_local_search': self.use_local_search,
                        'local_search_iterations': local_iterations,
                        'eigsh_iterations': int(iter_count),
                        'warm_start': not self.disable_warm_start,
                        'alpha_index': alpha_idx,
                        'precision': self.precision,
                        'lambda1': lambda1,
                        'degree_scale': degree_scale,
                        'energy_bound': bound_rescaled,
                        'pred_mode': self.pred_mode,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if energy < best_energy and not (np.isinf(energy) or np.isnan(energy)):
                        old_best = best_energy
                        best_energy, best_spins, best_alpha, best_beta = energy, spins.copy(), alpha, None
                        best_local_iterations = local_iterations
                        print(f"  [Best update] New best energy: {old_best:.6f} -> {best_energy:.6f} (alpha={alpha:.6f})")
                        
                except Exception as e:
                    print(f"Alpha {alpha} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if self.use_local_search and unique_solutions:
                print(f"Applying local search to {len(unique_solutions)} unique solutions (out of {len(alpha_list)} total)")
                solution_hashes = list(unique_solutions.keys())
                spins_list = [spins for spins, _ in unique_solutions.values()]
                
                improved_solutions = {}
                
                if len(spins_list) > 0:
                    # Use the specified precision for local search
                    J_csr_precision = cpx_sparse.csr_matrix(J_csr.astype(self.dtype), dtype=self.dtype)
                    
                    refined_spins_batch, local_iterations = flip_refinement(
                        J_csr_precision, spins_list,
                        max_iterations=self.max_local_iterations,
                        dtype=self.dtype
                    )
                    
                    for i, solution_hash in enumerate(solution_hashes):
                        original_spins, alpha_indices = unique_solutions[solution_hash]
                        original_energy = self.compute_energy(original_spins, J_csr)
                        
                        improved_spins = cp.sign(refined_spins_batch[:, i].astype(self.dtype))
                        improved_spins[improved_spins == 0] = 1.0
                        new_energy = self.compute_energy(improved_spins, J_csr)
                        if new_energy > original_energy:
                            improved_solutions[solution_hash] = (original_spins, original_energy, local_iterations)
                        else:
                            improved_solutions[solution_hash] = (improved_spins, new_energy, local_iterations)
                
                for idx_a, result in enumerate(alpha_results):
                    if idx_a in alpha_to_solution_hash:
                        solution_hash = alpha_to_solution_hash[idx_a]
                        if solution_hash in improved_solutions:
                            improved_spins, improved_energy, local_iterations = improved_solutions[solution_hash]
                            result["energy"] = float(improved_energy)
                            result["local_search_iterations"] = local_iterations
                            
                            if result["energy"] < best_energy:
                                best_energy = result["energy"]
                                best_spins = improved_spins.copy()
                                best_alpha = result["alpha"]
                                best_beta = result["beta"]
                                best_local_iterations = local_iterations

            total_time = time.time() - start_time
            
            # Check if we have valid results
            if best_energy == float("inf") or best_spins is None:
                if alpha_results:
                    # Use the first valid result if available
                    for result in alpha_results:
                        if result.get('energy') is not None and not (np.isinf(result['energy']) or np.isnan(result['energy'])):
                            best_energy = result['energy']
                            best_alpha = result['alpha']
                            best_beta = result.get('beta')
                            print(f"Warning: No valid best solution found, using first valid result with energy={best_energy}")
                            break
                else:
                    print("Warning: No valid results found for any alpha value!")
            
            if best_alpha is not None:
                self._store_general_results(best_energy, best_alpha, best_beta, total_time, best_local_iterations)
            else:
                print("Warning: Cannot store general results - no valid alpha found")
            
            self._store_timing_results(alpha_results)
            
        else:
            raise ValueError("Alpha is not set")

    def _get_alpha_values(self, mode: int) -> list:
        if mode == 0:
            return [float(x) for x in cp.sqrt(cp.linspace(0, 1, 128))]
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _compute_spin_signature(self, spins: cp.ndarray) -> bytes:
        bits = (spins > 0).astype(cp.uint8)
        packed = cp.packbits(bits)
        return packed.tobytes()

    def _calculate_beta(self, deg_abs: "cp.ndarray") -> float:
        """
        OLD IMPLEMENTATION: Kept for reference only, not used in V2
        """
        beta = float((1.0 + 1e-8) * cp.max(deg_abs ** (1.0 - self.alpha)))
        return beta

    def _solve_eigenvalue_problem_v2(self, J_csr, D_vec, alpha_idx, alpha, alpha_prev,
                                      y_last, y_prev, lambda1_prev, deg_abs_normalized):
        """
        V2 eigenvalue solver with predictors and spectral-radius shift.
        
        Replaces the old _solve_eigenvalue_problem method which used:
        - QOperator and SOperator
        - Simple warm-start with prev_v0_y
        - Beta-based shift
        
        New approach:
        - NAlphaOperator: N^α(v) = D^{-1/2} A (D^{-1/2} v)
        - SRhoOperator: S_ρ(v) = ρv - N^α(v) (SPD when ρ > λ₁)
        - Predictor-based initialization (diagonal, secant, hybrid, subspace)
        - Residual check for predictor validation
        - Spectral-radius-based shift: ρ_k = (1+ε) max{λ₁, Gershgorin}
        """
        n = D_vec.size
        print(f"\n    [DEBUG _solve_eigenvalue_problem_v2] alpha_idx={alpha_idx}, alpha={alpha:.6f}")

        # Step 1: Build N^α operator
        N_op = NAlphaOperator(J_csr, D_vec, self.dtype)
        print(f"      [Step 1] N_op: shape={N_op.shape}, dtype={N_op.dtype}")

        # Step 2: Compute Gershgorin bound
        # Gershgorin bound: max_i (d_i^{1-α}) for normalized degrees
        gershgorin_bound = float(cp.max(cp.power(deg_abs_normalized, 1.0 - alpha)))
        print(f"      [Step 2] Gershgorin bound={gershgorin_bound:.6f}")

        # Step 3: Compute spectral-radius-based shift
        rho_k = compute_shift_rho(lambda1_prev, gershgorin_bound, eps=self.shift_eps)
        print(f"      [Step 3] rho_k={rho_k:.6f} (shift_eps={self.shift_eps}, lambda1_prev={lambda1_prev})")

        # Step 4: Build SPD operator S_ρ = ρI - N^α
        S_op = SRhoOperator(N_op, rho_k)
        print(f"      [Step 4] S_op: shape={S_op.shape}, dtype={S_op.dtype}")

        # Step 5: Predictor-based initialization
        if alpha_idx == 0 or self.disable_warm_start or y_last is None:
            # Random initialization
            y_init = cp.random.randn(n).astype(self.dtype)
            y_init = y_init / cp.linalg.norm(y_init)
            print(f"      [Step 5] Random initialization: y_init.norm={cp.linalg.norm(y_init):.6f}")
        else:
            # Compute delta_alpha for diagonal predictor
            delta_alpha = alpha - alpha_prev if alpha_prev is not None else 0.0
            print(f"      [Step 5] Using predictor: mode={self.pred_mode}, delta_alpha={delta_alpha:.6f}")

            # Select predictor based on mode
            if self.pred_mode == "diagonal":
                y_init = diagonal_predictor(y_last, deg_abs_normalized, delta_alpha, self.dtype)
            elif self.pred_mode == "secant":
                y_init = secant_predictor(y_last, y_prev, self.dtype)
            elif self.pred_mode == "hybrid":
                y_init = hybrid_predictor(y_last, y_prev, deg_abs_normalized, delta_alpha, self.dtype)
            elif self.pred_mode == "subspace":
                y_init = subspace_predictor(y_last, y_prev, N_op, self.dtype)
            else:
                raise ValueError(f"Unknown predictor mode: {self.pred_mode}")

            print(f"      [Step 5] After predictor: y_init.norm={cp.linalg.norm(y_init):.6f}")

            # Step 6: Residual check
            v = N_op(y_init)
            lambda_est = float(cp.dot(y_init, v))
            r = float(cp.linalg.norm(v - lambda_est * y_init))
            print(f"      [Step 6] Residual check: lambda_est={lambda_est:.6f}, residual={r:.6f}, pred_tol={self.pred_tol}")

            if r > self.pred_tol:
                y_init = y_last
                print(f"      [Step 6] Predictor quality poor, using y_last instead")

        # Step 7: Solve eigenvalue problem on SPD operator S_ρ
        try:
            evals, evecs_y, iter_count = cpx_splinalg.eigsh(
                S_op, k=1, which="SA",
                tol=self.eigsh_tol, maxiter=self.eigsh_maxiter, v0=y_init
            )
            print(f"      [Step 7] eigsh succeeded with v0: iter_count={iter_count}")
        except Exception as e:
            print(f"Warning: eigsh failed for alpha={alpha}, rho_k={rho_k}, error: {e}")
            # Fallback: try without initial vector
            try:
                evals, evecs_y, iter_count = cpx_splinalg.eigsh(
                    S_op, k=1, which="SA",
                    tol=self.eigsh_tol, maxiter=self.eigsh_maxiter
                )
                print(f"      [Step 7] eigsh succeeded without v0 (fallback): iter_count={iter_count}")
            except Exception as e2:
                print(f"Error: eigsh failed even without initial vector: {e2}")
                raise

        if hasattr(iter_count, 'item'):
            iter_count = int(iter_count.item())
        else:
            iter_count = int(iter_count)

        # Step 8: Recover eigenvalue of N^α from eigenvalue of S_ρ
        # If S_ρ v = μ v, then N^α v = (ρ - μ) v
        # So λ₁(N^α) = ρ - μ_min(S_ρ)
        mu_min = float(evals[0])
        lambda1 = rho_k - mu_min
        print(f"      [Step 8] mu_min={mu_min:.6f}, rho_k={rho_k:.6f}, lambda1={lambda1:.6f}")
        print(f"      [Step 8] evals={evals}, evecs_y.norm={cp.linalg.norm(evecs_y[:, 0]):.6f}")

        # ------------------------------------------------------------
        # SIGN CONSISTENCY 
        # ------------------------------------------------------------
        if y_last is not None:
            # Align sign so eigenvectors evolve smoothly along α
            dot_before = float(cp.dot(evecs_y[:, 0], y_last))
            if dot_before < 0:
                evecs_y[:, 0] = -evecs_y[:, 0]
                print(f"      [Sign consistency] Flipped sign (dot_before={dot_before:.6f})")
            else:
                print(f"      [Sign consistency] Sign OK (dot={dot_before:.6f})")

        norm_y = cp.linalg.norm(evecs_y[:, 0])
        if norm_y > 0:
            evecs_y[:, 0] /= norm_y
        print(f"      [Sign consistency] After normalization: evecs_y.norm={cp.linalg.norm(evecs_y[:, 0]):.6f}")
        
        # Step 9: Transform eigenvector back to original space
        # The eigenvector evecs_y is in the transformed space (y-space)
        # We need to apply D^{-1/2} to get back to original space (x-space)
        D_work = cp.maximum(D_vec, cp.array(1e-12, dtype=self.dtype))
        D_inv_sqrt = cp.reciprocal(cp.sqrt(D_work))

        X = evecs_y * D_inv_sqrt[:, cp.newaxis]
        
        if X.ndim == 1:
            X = X[:, cp.newaxis]
        
        print(f"      [Step 9] Transform to original space: X.shape={X.shape}, X.norm={cp.linalg.norm(X[:, 0]):.6f}")
        
        # Return: evals (1,), evecs in original space (n, 1), evecs_y in transformed space (n, 1), iter_count, lambda1
        evals_return = cp.array([lambda1], dtype=self.dtype)
        print(f"      [Return] lambda1={lambda1:.6f}, iter_count={iter_count}")
        return evals_return, X, evecs_y, iter_count, lambda1

    def compute_energy(self, spins: "cp.ndarray", J_csr) -> float:
        """
        UNCHANGED: Energy computation remains the same as original
        """
        if hasattr(J_csr, 'dot'):
            energy = -0.5 * cp.dot(spins, J_csr.dot(spins))
        else:
            energy = -0.5 * cp.einsum("i,ij,j->", spins, J_csr, spins)

        if hasattr(self, '_normalization_factor'):
            energy *= self._normalization_factor
        return float(energy)

    def _store_general_results(self, energy: float, alpha: float, beta: float, time_taken: float, local_search_iterations: int = 0):
        """
        UNCHANGED: Result storage remains the same as original
        """
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
            print(f"General results saved to: {csv_file}")
        except Exception as e:
            print(f"Error saving general results: {e}")

    def _store_timing_results(self, alpha_results: list):
        """
        UNCHANGED: Timing result storage remains the same as original
        """
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
# GPU KERNELS: Unchanged from original implementation
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
# LOCAL SEARCH FUNCTIONS: Unchanged from original implementation
# ============================================================

def gpu_final_sweep(J_csr,
                    spins_flat: cp.ndarray,
                    n: int,
                    k: int,
                    node_fraction: float = 0.5,
                    max_sub_iters: int = 5,
                    num_rounds: int = 3,
                    dtype=cp.float32):
    """
    UNCHANGED: GPU final sweep remains the same as original
    """
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


def flip_refinement(J_csr, spins_list, max_iterations=1000,
                    final_sweep_fraction=1.0, final_sweep_iters=25,
                    num_rounds=25, subset_fraction=0.9, dtype=cp.float32):
    """
    UNCHANGED: Local search refinement remains the same as original
    """
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

