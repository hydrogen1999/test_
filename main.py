#!/usr/bin/env python3
"""
Spectral Annealing Pipeline
"""
import sys
from utils.utils import parse_arguments, load_config, merge_args_with_config
from utils.core import run_solver
import cProfile


# Global variable to store parsed arguments for profiling
_parsed_args = None

def main():
    """Main execution function"""
    try:
        # Use global args if available, otherwise parse arguments
        args = _parsed_args if _parsed_args is not None else parse_arguments()
        
        # Load configuration file
        config = load_config(args.config)
        
        # Merge arguments with configuration
        final_config = merge_args_with_config(args, config)
        
        print(f"Running: {final_config.get('run_name', final_config['solver'])}")
        print(f"Solver: {final_config['solver']}")
        print(f"Dataset: {final_config['dataset']}")
        if 'instance' in final_config:
            print(f"Instance: {final_config['instance']}")
        print("=" * 50)
        
        # Run the solver
        results = run_solver(final_config)
        
        print("=" * 50)
        print("Execution Finished.")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.runtime_analysis is not None:
        # Use the provided filename or default to 'runtime.prof'
        prof_filename = args.runtime_analysis
        # Remove runtime_analysis from args before passing to main
        args.runtime_analysis = None
        # Store the parsed args globally for main() to use
        _parsed_args = args
        cProfile.run('sys.exit(main())', prof_filename)
    else:
        _parsed_args = args
        sys.exit(main())
