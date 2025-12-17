"""
Core execution logic for the spectral annealing pipeline
"""
from typing import Dict, Any
from .data_loader import DataLoader
from .selector import solver_selector


def run_solver(config: Dict[str, Any]) -> Dict[str, Any]:
    data_loader = DataLoader(config['dataset'])
    
    if 'instance' in config and config['instance']:
        instances = [config['instance']]
    else:
        instances = data_loader.list_instances()
    
    # Get solver class
    solver_class = solver_selector.get_solver_class(config['solver'])
    
    # Extract solver parameters from config (exclude global and execution params)
    solver_params = {k: v for k, v in config.items() 
                    if k not in ['solver', 'dataset', 'instance', 'num_seeds', 'random_seed', 'run_name']}
    
    successful_runs = 0
    total_runs = 0
    
    num_seeds = config.get('num_seeds', 1)
    base_seed = config.get('random_seed', 42)
    
    # Use run_name for results if available, otherwise use solver name
    result_name = config.get('run_name', config['solver'])
    
    for instance_name in instances:
        print(f"Processing instance: {instance_name}")
        
        J = data_loader.load_instance(instance_name)
        
        for seed_idx in range(num_seeds):
            current_seed = base_seed + seed_idx
            print(f"  Seed {seed_idx + 1}/{num_seeds} (seed={current_seed})")
            
            current_params = solver_params.copy()
            current_params['random_seed'] = current_seed
            
            solver = solver_class(
                instance_name=instance_name,
                dataset=config['dataset'],
                run_name=result_name,
                **current_params
            )

            try:
                solver.solve(J)
                successful_runs += 1
                
            except Exception as e:
                print(f"  Error processing {instance_name} with seed {current_seed}: {e}")
            
            total_runs += 1
    
    return {
        'solver': config['solver'],
        'run_name': result_name,
        'dataset': config['dataset'],
        'total_instances': len(instances),
        'total_runs': total_runs,
        'successful_runs': successful_runs
    }
