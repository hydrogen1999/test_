import numpy as np
import time
import scipy.sparse as sp
import simulated_bifurcation as sb
import torch
import pandas as pd
from pathlib import Path
import os
from datetime import datetime


class SimulatedBifurcationSolver:

    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 device: str = "cuda", use_local_search: bool = False,
                 run_name: str = None, agents: int = 1024,
                 mode: str = "ballistic", heated: bool = False, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.agents = agents
        self.device = "cuda"  
        self.mode = mode
        self.heated = heated
        self.use_local_search = use_local_search
        
        self.solver_name = run_name if run_name else "SB"
        
        print(f"Initialized SB solver with {self.agents} agents on CUDA (float64). Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  device: {self.device}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        print(f"  agents: {self.agents}")
        print(f"  mode: {self.mode}")
        print(f"  heated: {self.heated}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J):
        n = J.shape[0]
        
        if sp.issparse(J):
            J_torch = torch.tensor(J.toarray(), dtype=torch.float64, device="cuda")

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)



        try:
            start_time = time.time()
            value, solution = sb.minimize(
                -0.5 * J_torch,                      # Quadratic term
                torch.zeros(n, dtype=torch.float64, device="cuda"),  # No linear terms
                0.0,                                # No constant
                domain="spin",
                mode=self.mode,
                heated=self.heated,
                agents=self.agents,
                device="cuda",
                best_only=True
            )
            time_taken = time.time() - start_time

            if hasattr(value, 'cpu'):
                spins = value.cpu().numpy()
            elif hasattr(value, 'detach'):
                spins = value.detach().cpu().numpy()
            else:
                spins = np.array(value)

            if spins.ndim > 1:
                spins = spins.flatten()

            spins = np.sign(spins).astype(np.int8)
            spins[spins == 0] = 1  

            energy = float(solution.item()) if hasattr(solution, "item") else float(solution)

        except Exception as e:
            print(f"SB solver failed: {e}")
            return

        computed_energy = self.compute_energy(J, spins)
        
        self._store_results(energy, computed_energy, spins, time_taken)

    def compute_energy(self, J, spins):
        if sp.issparse(J):
            return -0.5 * spins.T @ J @ spins
        else:
            return -0.5 * np.dot(spins, np.dot(J, spins))
    
    def _store_results(self, sb_energy: float, computed_energy: float, spins: np.ndarray, time_taken: float):
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        instance_base = self.instance_name.replace('.txt', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': computed_energy, 
            'time': time_taken,
            'local_search': self.use_local_search,
            'agents': self.agents,
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
        print(f"Energy (computed): {computed_energy:.6f}")
        print(f"Energy (SB): {sb_energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)