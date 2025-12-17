"""
SDP Solver using MOSEK and CVXPY for Ising problems
Implements Shor's SDP relaxation (level 1 of Lasserre hierarchy)
"""

import numpy as np
import time
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import cvxpy as cp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SDPSolverMosek:
    """
    SDP Solver for Ising problems using MOSEK and CVXPY.
    Implements Shor's SDP relaxation with random hyperplane rounding.
    """

    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 verbose: bool = False, num_rounding_trials: int = 10,
                 run_name: str = None, **kwargs):
        """
        Initialize the SDP solver.
        
        Parameters:
        -----------
        instance_name : str
            Name of the instance file
        dataset : str
            Name of the dataset
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool, default False
            Whether to print MOSEK solver output
        num_rounding_trials : int, default 10
            Number of random hyperplane rounding trials (best result is kept)
        run_name : str, optional
            Custom name for results storage
        """
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.verbose = verbose
        self.num_rounding_trials = num_rounding_trials
        
        self.solver_name = run_name if run_name else "sdp"
        
        print(f"Initialized {self.solver_name} solver (SDP with MOSEK). Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  verbose: {self.verbose}")
        print(f"  num_rounding_trials: {self.num_rounding_trials}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J):
        """
        Solve the Ising problem using SDP relaxation.
        
        Assumes fields are absorbed in the bias node of J matrix.
        Treats the problem as pure quadratic Ising (no explicit linear terms).
        
        Parameters:
        -----------
        J : np.ndarray or csr_matrix
            Coupling matrix (fields absorbed in bias node)
            
        Returns:
        --------
        energy : float
            Energy of the rounded solution
        spins : np.ndarray
            Binary solution in {-1, +1}
        time_taken : float
            Total time taken
        """
        start_time = time.time()
        
        # Convert to sparse for normalization
        if sp.issparse(J):
            J_csr = J.tocsr()
        else:
            J_csr = sp.csr_matrix(J)
        
        n = J_csr.shape[0]
        
        # Normalize based on degrees to avoid numerical issues
        J_abs = J_csr.copy()
        J_abs.data = np.abs(J_abs.data)
        
        deg_abs_original = J_abs @ np.ones(n, dtype=np.float64)
        max_deg_abs = np.max(deg_abs_original) if deg_abs_original.size > 0 else 1.0

        
        self._normalization_factor = float(max_deg_abs)
        J_normalized = J_csr / max_deg_abs
        
        # Convert to dense for CVXPY. NOTE: POSSIBLE BOTTLENECK IN BIG INSTANCES
        J_dense = J_normalized.toarray() if sp.issparse(J_normalized) else J_normalized
        
        # Build SDP: X is n x n matrix
        # Objective: -sum_{i<j} J_ij * X_ij
        # Constraints: X_ii = 1 for all i
        
        # Variable SDP: Gram matrix
        X = cp.Variable((n, n), PSD=True)
        
        # Objective function: minimize -sum(J_ij * X_ij)
        objective = cp.Minimize(-cp.sum(cp.multiply(J_dense, X)))
        
        # Constraints: diagonal elements must be 1
        constraints = [cp.diag(X) == 1]
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.MOSEK, verbose=self.verbose)
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise RuntimeError(f"SDP solver failed with status: {problem.status}")
            
            X_opt = X.value
            
            if X_opt is None:
                raise RuntimeError("SDP solver did not return a solution")
            
            print(f"SDP solved successfully. Objective value: {problem.value:.6f}")
            
        except Exception as e:
            print(f"Error solving SDP: {e}")
            raise
        
        # Project X_opt to PSD explicitly to avoid numerical issues
        # MOSEK may return matrices that are not exactly PSD (eigenvalues ≈ -1e-8)
        X_opt = self._project_to_psd(X_opt)
        
        # Factorize X 
        # X = V @ V^T
        eigvals, eigvecs = np.linalg.eigh(X_opt)
        eigvals_positive = np.maximum(eigvals, 0)
        V = eigvecs @ np.diag(np.sqrt(eigvals_positive))
        
        # Hyperplane Rounding
        best_spins = None
        best_energy = float('inf')
        
        for trial in range(self.num_rounding_trials):
            trial_seed = (self.random_seed + trial) if self.random_seed is not None else None
            trial_rng = np.random.default_rng(trial_seed)
            
            spins = self._round_solution(V, trial_rng)
            # Compute energy with normalized J, then denormalize
            energy_normalized = self._compute_energy(spins, J_dense)
            energy = energy_normalized * self._normalization_factor
            
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()
        
        total_time = time.time() - start_time
        
        print(f"Best energy after {self.num_rounding_trials} rounding trials: {best_energy:.6f}")
        print(f"Total time: {total_time:.4f}s")
        
        # Store results (denormalize SDP objective for consistency)
        sdp_objective_denormalized = problem.value * self._normalization_factor if problem.value is not None else None
        
        self._store_results(
            energy=best_energy,
            spins=best_spins,
            time_taken=total_time,
            sdp_objective=sdp_objective_denormalized
        )
        
        return best_energy, best_spins, total_time

    def _project_to_psd(self, X):
        """
        Project matrix to positive semidefinite cone.
        
        Parameters:
        -----------
        X : np.ndarray
            Matrix to project (may have small negative eigenvalues)
            
        Returns:
        --------
        X_psd : np.ndarray
            Projected PSD matrix
        """
        # Use eigendecomposition for symmetric matrices (faster than SVD)
        eigvals, eigvecs = np.linalg.eigh(X)
        
        # Clip negative eigenvalues to small positive value
        eigvals = np.maximum(eigvals, 1e-12)
        
        # Reconstruct PSD matrix
        X_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return X_psd

    def _round_solution(self, V, rng):
        """
        Round the SDP solution to binary spins using random hyperplane rounding.
        
        Parameters:
        -----------
        V : np.ndarray
            Factorized embedding matrix (n x d) where X = V @ V^T
        rng : np.random.Generator
            Random number generator for this trial
            
        Returns:
        --------
        spins : np.ndarray
            Binary solution in {-1, +1}
        """
        # Random hyperplane: sample random vector using provided RNG (reproducible)
        r = rng.normal(size=V.shape[1])
        r = r / np.linalg.norm(r)  # Normalize
        
        # Project and take sign
        projections = V @ r
        spins = np.sign(projections)
        spins[spins == 0] = 1  # Handle zeros
        
        return spins.astype(np.int8)

    def _compute_energy(self, spins, J):
        """
        Compute Ising energy: -0.5 * spins^T @ J @ spins
        
        Parameters:
        -----------
        spins : np.ndarray
            Binary spins in {-1, +1}
        J : np.ndarray
            Coupling matrix (normalized)
            
        Returns:
        --------
        energy : float
            Ising energy (normalized)
        """
        # Quadratic term: -0.5 * spins^T @ J @ spins
        if sp.issparse(J):
            energy = -0.5 * float(spins @ (J @ spins))
        else:
            energy = -0.5 * float(np.dot(spins, np.dot(J, spins)))
        
        return energy

    def _store_results(self, energy: float, spins: np.ndarray, time_taken: float, sdp_objective: float):
        """
        Store results to CSV file.
        
        Parameters:
        -----------
        energy : float
            Final energy
        spins : np.ndarray
            Final spins
        time_taken : float
            Time taken
        sdp_objective : float
            SDP objective value
        """
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '').replace('.bin', '').replace('.gz', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': energy,
            'time': time_taken,
            'sdp_objective': sdp_objective,
            'num_rounding_trials': self.num_rounding_trials,
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
        print(f"Energy: {energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)

