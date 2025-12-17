import numpy as np
import time
import json
import os
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import dimod
from models.solvers import SimulatedAnnealingSolver
from dimod import BinaryQuadraticModel


class SASolver:

    def __init__(self, instance_name: str, dataset: str, random_seed: Optional[int] = None,
                 use_local_search: bool = True, run_name: str = None,
                 num_reads: int = 1000, num_sweeps: int = 1000, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.use_local_search = use_local_search

        self.solver_name = run_name if run_name else "SA"
        
        print(f"Initialized SA solver with {self.num_reads} reads. Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  num_reads: {self.num_reads}")
        print(f"  num_sweeps: {self.num_sweeps}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)


    @property
    def name(self) -> str:
        return "SA"

    @property
    def supports_gpu(self) -> bool:
        return False

    def solve(self, J):
        n = J.shape[0]
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        start_time = time.time()

        try:
            coo = J.tocoo()
            J_dict = {(int(i), int(j)): -float(v/2) 
                     for i, j, v in zip(coo.row, coo.col, coo.data)
                     if i != j}
            h = {}
            
            bqm = BinaryQuadraticModel.from_ising(h, J_dict)
            
            solver = SimulatedAnnealingSolver()
            
            params = {
                'num_reads': self.num_reads,
                'num_sweeps': self.num_sweeps,
            }
            if self.random_seed is not None:
                params['seed'] = self.random_seed

            solution = solver.solve_bqm(bqm, params)
            
            if isinstance(solution, dict):
                sampleset = solution['solution']
                solve_time = solution['time']
            else:
                sampleset = solution[0]['solution']
                solve_time = solution[0]['time']
            
            best_sample = sampleset.first.sample
            spins = np.array([best_sample[i] for i in range(n)], dtype=int)
            
            if sampleset.vartype == dimod.BINARY:
                spins = 2 * spins - 1
            
            sa_energy = float(sampleset.first.energy)

        except Exception as e:
            print(f"SA solver failed: {e}")
            return

        time_taken = time.time() - start_time

        computed_energy = self.compute_energy(J, spins)
        
        self._store_results(sa_energy, computed_energy, spins, time_taken)

    def compute_energy(self, J, spins):
        return -0.5 * spins.T @ J @ spins
    
    def _store_results(self, sa_energy: float, computed_energy: float, spins: np.ndarray, time_taken: float):
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
            'num_reads': self.num_reads,
            'num_sweeps': self.num_sweeps,
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
        print(f"Energy (SA): {sa_energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)
  