import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse import csr_matrix

try:
    from tabu import TabuSampler
    TABU_AVAILABLE = True
except ImportError:
    TABU_AVAILABLE = False
    TabuSampler = None


class TabuSearchSolver:
    """
    Wrapper for D-Wave's MST2 Tabu Search algorithm via dwave-tabu.
    """

    def __init__(
        self,
        instance_name: str,
        dataset: str,
        random_seed: int = None,
        num_reads: int = 100,
        timeout: int = 100,
        use_local_search: bool = False,
        run_name: str = None,
        **kwargs,
    ):

        if not TABU_AVAILABLE:
            raise ImportError(
                "dwave-tabu is not installed. Please install it with: pip install dwave-tabu"
            )

        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.num_reads = num_reads
        self.timeout = timeout
        self.use_local_search = False  
        
        self.solver_name = run_name if run_name else "tabu_search"


        self.sampler = TabuSampler()

        print(f"Initialized {self.solver_name} solver with num_reads={num_reads}, timeout={timeout}ms. Results will be saved under '{self.solver_name}'.")
        
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  num_reads: {self.num_reads}")
        print(f"  timeout: {self.timeout}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        n = J.shape[0]
        start_time = time.time()

        if sp.issparse(J):
            J_dense = J.toarray().astype(np.float64)

        try:
            h_dict = {i: 0.0 for i in range(n)}
            J_dict = {}
            
            for i in range(n):
                for j in range(i + 1, n):
                        J_dict[(i, j)] = -J_dense[i, j]


            response = self.sampler.sample_ising(
                h=h_dict,
                J=J_dict,
                num_reads=self.num_reads,
                timeout=self.timeout,
                seed=self.random_seed,
            )

            best_sample = None
            best_energy = float("inf")
            
            for sample, dwave_energy in response.data(["sample", "energy"]):
                spins = np.array([sample[i] for i in range(n)], dtype=np.int8)
                

                energy_val = dwave_energy
                
                if energy_val < best_energy:
                    best_energy = energy_val
                    best_sample = spins

            if best_sample is None:
                print("Warning: No valid solution found, using random fallback")
                best_sample = np.random.choice([-1, 1], size=n).astype(np.int8)
                best_energy = self.compute_energy(J_dense, best_sample)

        except Exception as e:
            print(f"Error during tabu search: {e}")
            best_sample = np.random.choice([-1, 1], size=n).astype(np.int8)
            best_energy = self.compute_energy(J_dense, best_sample)

        time_taken = time.time() - start_time

        self._store_results(
            energy=best_energy,
            spins=best_sample,
            time_taken=time_taken,
            iterations=self.num_reads
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
            'num_reads': self.num_reads,
            'timeout_ms': self.timeout,
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
        print(f"Num reads: {self.num_reads}")
        print(f"Timeout: {self.timeout}ms")
        print("-" * 40)