"""
Utility functions for argument parsing and configuration management
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple Spectral Annealing Pipeline")
    
    parser.add_argument('--solver', required=True,
                       help='Name of the solver to use')
    parser.add_argument('--dataset', required=True,
                       help='Name of the dataset to use')
    parser.add_argument('--instance', default=None,
                       help='Specific instance name (optional)')
    parser.add_argument('--config', default='default',
                       help='Config file name in configs/ directory (optional)')
    parser.add_argument('--runtime_analysis', nargs='?', const='runtime.prof', default=None,
                       help='Enable runtime profiling (optional output filename)')
    
    return parser.parse_args()


def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(f"configs/{config_name}.yml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    run_config = config.get(args.solver, {})
    
    if not run_config:
        print(f"Warning: No configuration found for run '{args.solver}' in config file")
        run_config = {}
    
    merged = {}
    
    for key, value in config.items():
        if not isinstance(value, dict):  
            merged[key] = value
    
    if 'solver_name' in run_config and 'params' in run_config:
        actual_solver = run_config['solver_name']
        solver_params = run_config['params']
        
        merged.update(solver_params)
        
        merged['solver'] = actual_solver
        merged['run_name'] = args.solver  
        
    else:
        merged.update(run_config)
        merged['solver'] = args.solver
        merged['run_name'] = args.solver  

    merged['dataset'] = args.dataset
    if args.instance:
        merged['instance'] = args.instance
    
    return merged
