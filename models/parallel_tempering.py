import numpy as np
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from omnisolver.pt.sampler import PTSampler
    PT_AVAILABLE = True
except ImportError:
    PT_AVAILABLE = False
    PTSampler = None


class ParallelTemperingSolver:

    def __init__(self, instance_name: str, dataset: str, random_seed: Optional[int] = None,
                 use_local_search: bool = False, run_name: str = None,
                 num_replicas: int = 10, num_pt_steps: int = 1000, num_sweeps: int = 10,
                 beta_min: float = 0.1, beta_max: float = 5.0, **kwargs):

        if not PT_AVAILABLE:
            raise ImportError(
                "omnisolver-pt is not installed. Please install it with: pip install omnisolver-pt"
            )

        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.use_local_search = False  # PT doesn't use local search

        self.solver_name = run_name if run_name else "parallel_tempering"
        
        # PT-specific parameters
        self.num_replicas = num_replicas
        self.num_pt_steps = num_pt_steps
        self.num_sweeps = num_sweeps
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Initialize the sampler
        self.sampler = PTSampler()
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        print(f"Initialized PT solver with {self.num_replicas} replicas, {self.num_pt_steps} steps. Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  num_replicas: {self.num_replicas}")
        print(f"  num_pt_steps: {self.num_pt_steps}")
        print(f"  num_sweeps: {self.num_sweeps}")
        print(f"  beta_min: {self.beta_min}")
        print(f"  beta_max: {self.beta_max}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    @property
    def name(self) -> str:
        return "parallel_tempering"

    @property
    def supports_gpu(self) -> bool:
        return False

    def solve(self, J):
        """Solve Ising model using Parallel Tempering."""
        
        n = J.shape[0]
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        

        try:
            h_dict = {}
            J_dict = {}
            
            coo = J.tocoo()
            J_dict = {
                (int(i), int(j)): float(v)
                for i, j, v in zip(coo.row, coo.col, -coo.data)
                if i < j
            }
            start_time = time.time()
            response = self.sampler.sample_ising(
                h=h_dict,
                J=J_dict,
                num_replicas=self.num_replicas,
                num_pt_steps=self.num_pt_steps,
                num_sweeps=self.num_sweeps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
            )
            time_taken = time.time() - start_time
        
            best_sample, best_energy = None, float("inf")
            for sample, energy in response.data(["sample", "energy"]):
                if energy < best_energy:
                    best_energy = energy
                    best_sample = sample

            spins = np.array([best_sample[i] for i in range(n)], dtype=int)
            pt_energy = float(best_energy)

        except Exception as e:
            print(f"PT solver failed: {e}")

        self._store_results(pt_energy, spins, time_taken)
    
    def _store_results(self, pt_energy: float, spins: np.ndarray, time_taken: float):
        """Store results in CSV format"""
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': pt_energy,  
            'time': time_taken,
            'local_search': self.use_local_search,
            'num_replicas': self.num_replicas,
            'num_pt_steps': self.num_pt_steps,
            'num_sweeps': self.num_sweeps,
            'beta_min': self.beta_min,
            'beta_max': self.beta_max,
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
        print(f"Energy (PT): {pt_energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)