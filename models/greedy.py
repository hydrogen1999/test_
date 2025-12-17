import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import csr_matrix


class GreedySolver:
    def __init__(
        self,
        instance_name: str,
        dataset: str,
        random_seed: int = None,
        n_trials: int = 1000,
        max_sweeps: int = 1000,
        use_local_search: bool = False,
        run_name: str = None,
        **kwargs,
    ):

        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.n_trials = n_trials
        self.max_sweeps = max_sweeps
        self.use_local_search = False  # Greedy doesn't use additional local search

        self.solver_name = run_name if run_name else "greedy"

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        print(f"Initialized {self.solver_name} solver with n_trials={n_trials}, max_sweeps={max_sweeps}. Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  n_trials: {self.n_trials}")
        print(f"  max_sweeps: {self.max_sweeps}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        start_time = time.time()

        if isinstance(J, csr_matrix):
            J_cpu = J.astype(np.float64, copy=False)
        else:
            if sp.issparse(J):
                J = J.toarray()
            J_cpu = np.array(J, dtype=np.float64)

        n = J_cpu.shape[0]

        best_spins = None
        best_energy = float("inf")
        total_iterations = 0

        for trial in range(self.n_trials):
            spins = np.random.choice([-1, 1], size=n).astype(np.int8)

            for sweep in range(self.max_sweeps):
                flips = 0
                for i in range(n):
                    if isinstance(J_cpu, csr_matrix):
                        local_field = J_cpu[i].dot(spins)
                    else:
                        local_field = np.dot(J_cpu[i], spins)

                    delta_e = 2 * spins[i] * local_field

                    if delta_e < 0:
                        spins[i] *= -1
                        flips += 1

                if flips == 0:
                    break
                
                total_iterations += 1

            energy = self.compute_energy(J_cpu, spins) 

            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()

        time_taken = time.time() - start_time

        self._store_results(
            energy=best_energy,
            spins=best_spins,
            time_taken=time_taken,
            iterations=total_iterations
        )

    def compute_energy(self, J: np.ndarray, spins: np.ndarray) -> float:
        if isinstance(J, csr_matrix):
            return -0.5 * float(spins @ (J @ spins))
        else:
            return -0.5 * float(np.einsum("i,ij,j->", spins, J, spins))

    def _store_results(self, energy: float, spins: np.ndarray, time_taken: float, iterations: int):
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
            'n_trials': self.n_trials,
            'max_sweeps': self.max_sweeps,
            'iterations': iterations,
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
        print(f"Iterations: {iterations}")
        print(f"Trials: {self.n_trials}")
        print("-" * 40)