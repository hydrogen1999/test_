"""
Simple solver selector - handles dynamic importing of solver classes
"""
import importlib
import sys
from pathlib import Path
from typing import Type


class SolverSelector:
    
    def __init__(self):
        self.current_path = Path(__file__).parent.parent.resolve()  
        
        self.solvers = {
            # Simulated Annealing solvers
            'SA': ('models.simulated_annealing', 'SASolver'),
            
            # Simulated Bifurcation solver (unified)
            'SB': ('models.simulated_bifurcation', 'SimulatedBifurcationSolver'),
            
            # Spectral Annealing solvers
            'spectral_annealing': ('models.spectral_annealing', 'SpectralAnnealing'),
            'spectral_annealing_legacy': ('models.spectral_annealing', 'SpectralAnnealingLegacy'),
            'spectral_annealing_v2': ('models.spectral_annealing_v2', 'SpectralAnnealingV2'),
            'spectral_annealing_no_fast_eig': ('models.spectral_annealing_no_fast_eig', 'SpectralAnnealingV2'),
            'spectral_annealing_no_fast_LS_no_fast_eig': ('models.spectral_annealing_no_fast_LS_no_fast_eig', 'SpectralAnnealingV3'),
            'spectral_annealing_no_fast_LS': ('models.spectral_annealing_no_fast_LS', 'SpectralAnnealingV4'),
            'spectral_annealing_trevisan': ('models.spectral_annealing_trevisan', 'SpectralAnnealingTrevisan'),
            
            # GPU solvers
            'spec_adj': ('models.spec_adj', 'Alpha0GPU'),
            'spec_signed_lap': ('models.spec_signed_lap', 'Alpha1GPU'),
            
            # SDP solvers
            'sdp': ('models.sdp_mosek', 'SDPSolverMosek'),
            
            # SPONGE solver
            'sponge': ('models.sponge', 'SpongeSolver'),
            
            # QQA solver
            'qqa': ('models.qqa_solver', 'QQASolver'),
            
            # Exact solvers
            'gurobi': ('models.gurobi', 'GurobiSolver'),
            
            # Heuristic solvers
            'greedy': ('models.greedy', 'GreedySolver'),
            'tabu_search': ('models.tabu_search', 'TabuSearchSolver'),
            'parallel_tempering': ('models.parallel_tempering', 'ParallelTemperingSolver'),
            
            # iSCO solver
            'isco': ('models.isco_solver', 'ISCOSolver'),

            # Only LS
            'onlyLS': ('models.onlyLS', 'OnlyLS'),

            # Improved Spectral solver
            'improved_spectral': ('models.improved_spectral', 'ImprovedSpectralSolverWrapper'),

            # DSN Solver
            'DSN': ('models.dsn_solver', 'DSNSolverWrapper'),
            'DSN_fast': ('models.dsn_solver', 'DSNSolverWrapper'),
            'DSN_thorough': ('models.dsn_solver', 'DSNSolverWrapper'),
            'DSN_large': ('models.dsn_solver', 'DSNSolverWrapper'),
        }
    
    def get_solver_class(self, solver_name: str) -> Type:
        if solver_name not in self.solvers:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        module_path, class_name = self.solvers[solver_name]
        
        paths_to_add = []
        if str(self.current_path) not in sys.path:
            sys.path.insert(0, str(self.current_path))
            paths_to_add.append(str(self.current_path))
        
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                raise ValueError(f"Class {class_name} not found in module {module_path}")
            return getattr(module, class_name)
        
        except ImportError as e:
            raise ImportError(f"Failed to import solver {solver_name}: {e}")
        
        finally:
            for path in paths_to_add:
                if path in sys.path:
                    sys.path.remove(path)
    
    def list_available_solvers(self) -> list:
        return list(self.solvers.keys())
    
    def add_solver(self, name: str, module_path: str, class_name: str):
        self.solvers[name] = (module_path, class_name)


solver_selector = SolverSelector()