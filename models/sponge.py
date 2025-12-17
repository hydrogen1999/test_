"""
SPONGE Solver for Ising problems using SigNet
Implements SPONGE clustering on signed graphs derived from Ising coupling matrices
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from signet.cluster import Cluster
except ImportError:
    raise ImportError(
        "SigNet is not installed. Install it with: pip install git+https://github.com/alan-turing-institute/SigNet.git"
    )


class SpongeSolver:
    """
    SPONGE solver wrapper for Ising problems, using the SigNet implementation.
    
    Converts J (Ising couplings) into (A+, A-) for signed graph.
    Applies SPONGE or SPONGE_sym via SigNet.
    Converts clusters to spins in {-1, +1} and computes Ising energy.
    Saves results in CSV with the same format as other solvers.
    """

    def __init__(self,
                 instance_name: str,
                 dataset: str,
                 random_seed: int = None,
                 tau_p: float = 1.0,
                 tau_n: float = 1.0,
                 use_sym: bool = True,
                 run_name: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        instance_name : str
            Instance file name
        dataset : str
            Dataset name
        random_seed : int, optional
            Random seed for reproducibility
        tau_p, tau_n : float
            SPONGE regularization parameters (see original paper).
        use_sym : bool, default True
            If True, uses SPONGE_sym (normalized symmetric version).
            If False, uses classic SPONGE.
        run_name : str, optional
            Solver name for results folders. Default 'sponge_sym' or 'sponge'.
        """
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.k = 2  # Fixed to 2 for Ising problems (binary spins)
        self.tau_p = tau_p
        self.tau_n = tau_n
        self.use_sym = use_sym

        base_name = "sponge_sym" if use_sym else "sponge"
        self.solver_name = run_name if run_name else base_name

        if random_seed is not None:
            np.random.seed(random_seed)

        print(f"Initialized {self.solver_name} solver (SPONGE via SigNet). "
              f"Results will be saved under '{self.solver_name}'.")
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  k (clusters): {self.k} (fixed for Ising)")
        print(f"  tau_p: {self.tau_p}")
        print(f"  tau_n: {self.tau_n}")
        print(f"  use_sym: {self.use_sym}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def _split_positive_negative(self, J):
        """
        From J (symmetric, diagonal ≈ 0), build Ap, An in CSR format.
        Ap_ij = max(J_ij, 0), An_ij = max(-J_ij, 0).
        
        Parameters
        ----------
        J : np.ndarray or scipy.sparse
            Coupling matrix
            
        Returns
        -------
        Ap : scipy.sparse.csr_matrix
            Positive part of J
        An : scipy.sparse.csr_matrix
            Negative part of J (absolute values)
        """
        if sp.issparse(J):
            J_csr = J.tocsr()
        else:
            J_csr = sp.csr_matrix(J)

        n = J_csr.shape[0]

        # Extract data
        data = J_csr.data
        indices = J_csr.indices
        indptr = J_csr.indptr

        # Positive: max(data, 0)
        data_pos = np.where(data > 0, data, 0.0)
        Ap = sp.csr_matrix((data_pos, indices, indptr), shape=(n, n))

        # Negative: max(-data, 0)
        data_neg = np.where(data < 0, -data, 0.0)
        An = sp.csr_matrix((data_neg, indices, indptr), shape=(n, n))

        # Explicitly remove diagonal (just in case)
        Ap.setdiag(0.0)
        An.setdiag(0.0)
        Ap.eliminate_zeros()
        An.eliminate_zeros()

        return Ap, An

    def solve(self, J):
        """
        Execute SPONGE on matrix J and return energy, spins, time.
        
        Parameters
        ----------
        J : np.ndarray or scipy.sparse
            Ising coupling matrix (fields already absorbed in bias node, if applicable).
        
        Returns
        -------
        energy : float
            Ising energy of the found solution.
        spins : np.ndarray, shape (n,)
            Spin vector in {-1, +1}.
        time_taken : float
            Total execution time (s).
        """
        start_time = time.time()

        # Split into positive and negative parts
        Ap, An = self._split_positive_negative(J)
        n = Ap.shape[0]

        # Build Cluster object from SigNet
        cluster_obj = Cluster((Ap, An))

        # Execute SPONGE or SPONGE_sym
        sponge_kwargs = {
            "k": self.k,
            "tau_p": self.tau_p,
            "tau_n": self.tau_n,
        }

        try:
            if self.use_sym:
                labels = cluster_obj.SPONGE_sym(**sponge_kwargs)
            else:
                labels = cluster_obj.SPONGE(**sponge_kwargs)
        except Exception as e:
            raise RuntimeError(f"SPONGE execution failed: {e}")

        labels = np.asarray(labels)
        if labels.shape[0] != n:
            raise RuntimeError(
                f"SPONGE returned {labels.shape[0]} labels but J has size {n}"
            )

        # Convert clusters to spins: k is fixed to 2 for Ising problems
        # Map labels to spins using majority/minority assignment
        unique_labels = np.unique(labels)
        base_spins = np.ones(n, dtype=np.int8)

        if len(unique_labels) == 2:
            # Map majority label to +1, minority label to -1 (more robust)
            counts = [(lab, np.sum(labels == lab)) for lab in unique_labels]
            counts.sort(key=lambda x: -x[1])  # Most frequent first
            majority_label = counts[0][0]
            minority_label = counts[1][0]
            base_spins[labels == minority_label] = -1
        else:
            # SPONGE collapsed to a single cluster -> all spins same sign
            # We'll test both orientations (+1 all, -1 all) and keep the best
            base_spins = np.ones(n, dtype=np.int8)  # All +1

        spins1 = base_spins
        spins2 = -base_spins  # Inverted orientation (physically equivalent)

        energy1 = self._compute_energy(spins1, J)
        energy2 = self._compute_energy(spins2, J)

        if energy1 <= energy2:
            best_spins = spins1
            best_energy = energy1
        else:
            best_spins = spins2
            best_energy = energy2

        time_taken = time.time() - start_time

        print(f"SPONGE finished. Best energy: {best_energy:.6f}")
        print(f"Total time: {time_taken:.4f}s")

        # Store results
        self._store_results(
            energy=best_energy,
            spins=best_spins,
            time_taken=time_taken,
        )

        return best_energy, best_spins, time_taken

    def _compute_energy(self, spins, J):
        """
        Ising energy: -0.5 * s^T J s
        (Assumes J already includes bias node if absorbed before)
        
        Parameters
        ----------
        spins : np.ndarray
            Spin vector in {-1, +1}
        J : np.ndarray or scipy.sparse
            Coupling matrix
            
        Returns
        -------
        energy : float
            Ising energy
        """
        if sp.issparse(J):
            return -0.5 * float(spins @ (J @ spins))
        else:
            return -0.5 * float(np.dot(spins, np.dot(J, spins)))

    def _store_results(self, energy: float, spins: np.ndarray, time_taken: float):
        """
        Store results in CSV, similar to other solvers in the pipeline.
        
        Parameters
        ----------
        energy : float
            Final energy
        spins : np.ndarray
            Final spins
        time_taken : float
            Time taken
        """
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        instance_base = (
            self.instance_name.replace(".txt", "")
            .replace(".bin", "")
            .replace(".gz", "")
        )
        csv_file = results_dir / f"{instance_base}.csv"

        result_data = {
            "instance_name": self.instance_name,
            "dataset": self.dataset,
            "seed": self.random_seed,
            "solver_name": self.solver_name,
            "energy": energy,
            "time": time_taken,
            "k": self.k,
            "tau_p": self.tau_p,
            "tau_n": self.tau_n,
            "use_sym": self.use_sym,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()

        if not df.empty:
            duplicate_mask = (
                (df["instance_name"] == result_data["instance_name"])
                & (df["dataset"] == result_data["dataset"])
                & (df["seed"] == result_data["seed"])
                & (df["solver_name"] == result_data["solver_name"])
            )

            if duplicate_mask.any():
                df.loc[duplicate_mask, list(result_data.keys())] = list(
                    result_data.values()
                )
            else:
                df = pd.concat(
                    [df, pd.DataFrame([result_data])], ignore_index=True
                )
        else:
            df = pd.DataFrame([result_data])

        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"Results saved to: {csv_file}")
        print(f"Energy: {energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)

