"""
QUBO Solver Implementation

This module provides implementations for solving Quadratic Unconstrained Binary Optimization (QUBO) problems
using various solvers including D-Wave quantum annealer, Simulated Annealing, and Gurobi optimizer.
"""

import copy
import json
import time
import traceback
from typing import Any, Dict, List, Union, Optional

from dwave.cloud import Client

import dimod
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.cloud.exceptions import SolverFailureError, ConfigFileError, SolverNotFoundError, SolverAuthenticationError
from neal import SimulatedAnnealingSampler
from dwave.embedding.chain_breaks import MinimizeEnergy, majority_vote, discard
from dwave.samplers import SteepestDescentSolver

from models.improved_spectral import ImprovedSpectralSolver
import torch

import gurobipy as gp

import utils.ising_util as ising_util

import networkx as nx

# Remove unused imports
# from dwave.cloud import Client
# from dimod import quicksum


def get_best_energy(sampleset: dimod.SampleSet) -> Dict[str, Union[float, int]]:
    """
    Calculate the best energy and its frequency from a sampleset.

    Args:
        sampleset (dimod.SampleSet): The sampleset to analyze.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the best energy, number of samples,
        and frequency of the best energy.
    """
    best_energy = sampleset.first.energy
    best_freq = 0
    num_samples = 0
    for sample in sampleset.record:
        num_samples += sample.num_occurrences
        if sample.energy <= best_energy + 1e-6:
            best_freq += sample.num_occurrences
    return {
        'best_energy': best_energy,
        'num_samples': num_samples,
        'best_energy_freq': best_freq
    }



class QUBOSolver:
    """Base class for QUBO solvers."""

    def __init__(self):
        self.answer = {'solution': None, 'best_obj': None, 'time': None}

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the solver's solution to a serializable dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the solver information.
        """
        return self.answer

    def answer_from_serializable(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the solution from a serializable dictionary.
        """
        return answer
    
    def solve_bqm(self, bqm: BinaryQuadraticModel, params: Optional[Dict] = None) -> Any:
        """
        Solve a QUBO problem.

        Args:
            bqm (BinaryQuadraticModel): The QUBO problem to solve.
            params (Optional[Dict]): Additional parameters for solving the problem.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")



class DWaveSolver(QUBOSolver):
    """A wrapper for the D-Wave solver."""

    def __init__(self, token_list: Union[List[str], str], solver_id: str, logging: Optional[Any] = None):
        """
        Initialize the DWaveSolver class.

        Args:
            token_list (Union[List[str], str]): List of tokens for accessing D-Wave solvers or a file path to a JSON file.
            solver_id (str): ID of the D-Wave solver to use.
            logging (Optional[Any]): Logging object for error reporting.
        """
        super().__init__()
        self.token_list: List[str] = []
        self.expired_tokens: List[Dict[str, str]] = []
        self.current_token_index: int = 0
        self.solver_id: str = solver_id
        self.sampler: Optional[Union[DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite]] = None
        self.logging = logging

        self._initialize_token_list(token_list)
        self._setup_client()

    def _initialize_token_list(self, token_list: Union[List[str], str]) -> None:
        """
        Initialize the token list from either a list of strings or a JSON file.

        Args:
            token_list (Union[List[str], str]): List of tokens or path to JSON file containing tokens.

        Raises:
            ValueError: If the token_list format is invalid.
        """
        if isinstance(token_list, str):
            self.token_file = token_list
            if token_list.endswith('.json'):
                with open(token_list, 'r') as file:
                    token_data = json.load(file)
                self.token_list = token_data.get('active', [])
                self.expired_tokens = token_data.get('expired', [])
            else:
                raise ValueError("Invalid file format. Expected JSON file.")
        elif isinstance(token_list, list):
            self.token_list = token_list
        else:
            raise ValueError("Invalid token_list format. Expected list or file path to JSON file.")

    def _setup_client(self) -> None:
        """
        Set up the D-Wave client and sampler.

        This method tries to set up the client and sampler using the token list and solver parameters.
        If a token fails, it moves to the next token in the list until a successful setup is achieved.

        Raises:
            Exception: If all tokens have expired.
        """
        while self.current_token_index < len(self.token_list):
            try:
                self.sampler = DWaveSampler(token=self.token_list[self.current_token_index], solver=self.solver_id)
                break
            except (SolverFailureError, ConfigFileError, SolverNotFoundError, SolverAuthenticationError) as e:    
                self._handle_token_failure(e)

        if self.current_token_index >= len(self.token_list):
            self.save_tokens()
            raise Exception("All tokens expired")

    def _handle_token_failure(self, error: Exception) -> None:
        """
        Handle a token failure by logging the error and moving to the next token.

        Args:
            error (Exception): The error that occurred during token usage.
        """
        self.expired_tokens.append({
            "token": self.token_list[self.current_token_index], 
            "error": str(error), 
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "traceback": traceback.format_exc()
        })
        self.current_token_index += 1
        self.save_tokens()

    def _get_sampler(self, params: Dict) -> Union[FixedEmbeddingComposite, EmbeddingComposite]:
        """
        Get the appropriate sampler based on the provided parameters.

        Args:
            params (Dict): Solver parameters.

        Returns:
            Union[FixedEmbeddingComposite, EmbeddingComposite]: The selected sampler.
        """
        if 'embedding' in params:
            return FixedEmbeddingComposite(self.sampler, embedding=params['embedding'])
        return EmbeddingComposite(self.sampler)
    
    def _validate_fast_anneal_constraints(self, bqm: BinaryQuadraticModel) -> None:
        """
        Validates that for a given BinaryQuadraticModel in the SPIN form, all linear biases (h)
        and diagonal quadratic values are zero.

        Args:
            bqm (BinaryQuadraticModel): The BinaryQuadraticModel to validate, in either BINARY or SPIN form.

        Raises:
            ValueError: If any linear bias or diagonal value is not zero in the SPIN form.
        """
        # Convert BQM to SPIN form if it is in BINARY form
        if bqm.vartype == dimod.BINARY:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

        # Check for non-zero linear biases
        if any(bqm.linear[v] != 0 for v in bqm.linear):
            raise ValueError("When fast_anneal is True, all linear biases (h) must be zero in SPIN form.")
        
        # Check for non-zero diagonal quadratic values
        if any(value != 0 for (i, j), value in bqm.quadratic.items() if i == j):
            raise ValueError("When fast_anneal is True, all diagonal values in the quadratic matrix must be zero in SPIN form.")

    def _prepare_solve_parameters(self, params: Dict, bqm: BinaryQuadraticModel) -> Dict:
        """
        Prepare the parameters for solving the BQM.

        Args:
            params (Dict): User-provided parameters.
            bqm (BinaryQuadraticModel): The BQM to be solved.

        Returns:
            Dict: Prepared solve parameters.
        """
        params = params or {}
        solve_parameters = {'num_reads':  params.get('num_reads', 200)}
        # List of parameter names to check
        parameter_names = ['anneal_schedule', 'annealing_time']

        # Iterate over the list and update solve_parameters if the parameter exists in params
        for param_name in parameter_names:
            if param_name in params:
                solve_parameters[param_name] = params[param_name]
        
        if params.get('fast_anneal'):
            solve_parameters['fast_anneal'] = True
            self._validate_fast_anneal_constraints(bqm)

        return solve_parameters

    def _handle_solver_failure(self, error: SolverFailureError) -> None:
        """
        Handle a solver failure by logging the error and moving to the next token.

        Args:
            error (SolverFailureError): The error that occurred during solving.
        """
        if self.logging:
            self.logging.error(f"Solver failure: {error}")
        else:
            print(f"Warning: Solver failure: {error} for token {self.token_list[self.current_token_index]}")
        self._handle_token_failure(error)
        self.sampler = None
        self.best_obj = None

    def _handle_unexpected_error(self, error: Exception) -> None:
        """
        Handle an unexpected error by logging it.

        Args:
            error (Exception): The unexpected error that occurred.
        """
        if self.logging:
            self.logging.error(f"An unexpected error occurred: {error}")
        self.best_obj = None

    # Static method to get a list of available solvers
    @staticmethod
    def get_available_solvers(token):
        return Client.from_config(token=token).get_solvers()
    
    def get_active_tokens(self):
        return self.token_list[self.current_token_index:]
    
    def get_tokens(self):
        """
        Get the active and expired tokens.
        """
        return {"active": self.token_list[self.current_token_index:], "expired": self.expired_tokens}
    
    def save_tokens(self, file_path=None):
        """
        Save the active and expired tokens to a JSON file.
        """
        if file_path is None:
            file_path = self.token_file
        if file_path:    
            with open(file_path, 'w') as file:
                json.dump(self.get_tokens(), file, indent=4)
    
    @staticmethod
    def _dwave_qpu_time(qpu_timing, params, num_reads):
        """Compute the QPU time for the DWave solver."""
        qpu_time = 0
        if params.get("qpu_access_time", False):
            qpu_time = qpu_timing.get('qpu_access_time', 0)
        else:
            if params.get('qpu_sampling_time', False):
                qpu_time = qpu_timing.get('qpu_sampling_time', 0)
            else:
                # Time per sample
                time_per_sample = ["qpu_anneal_time_per_sample", "qpu_readout_time_per_sample", "qpu_delay_time_per_sample"] 
                for t in time_per_sample:
                    if params.get(t, False):
                        qpu_time += qpu_timing.get(t, 0) * num_reads
                # Overhead time
                overhead_time = ["qpu_programming_time", "qpu_access_overhead_time", "post_processing_overhead_time"]
                for t in overhead_time:
                    if params.get(t, False):
                        qpu_time += qpu_timing.get(t, 0)
        qpu_time = qpu_time * 1e-6  # Return time in seconds
        if params.get('chain_break_time', False):
            qpu_time += qpu_timing.get('chain_break_time', 0)
        return  qpu_time
    
    def get_adjacency(self) -> nx.Graph:
        """
        Get the adjacency graph of the D-Wave solver.

        Returns:
            nx.Graph: The adjacency graph of the D-Wave solver.
        """
        adjacency = nx.Graph()
        adjacency.add_edges_from(self.sampler.edgelist)
        return adjacency

    def solve_bqm(self, bqm: BinaryQuadraticModel, params: Optional[Dict] = None) -> dimod.SampleSet:
        """
        Solve the QUBO problem using the D-Wave solver.

        Args:
            bqm (BinaryQuadraticModel): The QUBO problem to solve.
            params (Optional[Dict]): Additional parameters for solving the problem.

        Returns:
            dimod.SampleSet: The samples and information about the solver.

        Raises:
            Exception: If an unexpected error occurs during solving.
        """
        params = params or {}
        solve_parameters = self._prepare_solve_parameters(params, bqm)

        while True:
            if not self.sampler:
                self._setup_client()
            try:
                sampler = self._get_sampler(params)
                response = {}
                if not params.get('chain_break_method', False):                
                    response = sampler.sample(bqm, **solve_parameters)
                    self.answer = { 'best_obj'   :  response.first.energy,
                                    'time': self._dwave_qpu_time(response.info['timing'], params.get('timing', {}), solve_parameters['num_reads']),
                                    'solution': response}
                    self.default_answer = self.answer
                else:
                    # Return a collection of samples using different chain_break_methods
                    result = ising_util.solve_bqm_with_chain_break_methods(bqm, params['embedding'], self.sampler, **solve_parameters)
                    
                    # Return the best energy from majority_vote as the best energy
                    qpu_time = self._dwave_qpu_time(result['embedded'].info['timing'], params.get('timing', {}), solve_parameters['num_reads'])
                    # result.pop('embedded') 
                    self.answer = []
                    for name, res in result.items():
                        self.answer.append({
                            'variation': name,
                            'solution': res,
                            'best_obj': res.first.energy,
                            'time': qpu_time + (res.info['timing'].get('chain_break_time', 0) if params.get('chain_break_time', False) else 0)
                        })
                        if name == 'majority_vote':
                            self.default_answer = self.answer[-1]
                
                # Steepest descent post-processing 
                if params.get('steepest_descent', False):
                    start_time = time.time()
                    improved_result = SteepestDescentSolver().sample(bqm, initial_states=self.default_answer['solution'])
                    improvement_time = time.time() - start_time
                    sd_answer = {
                        'variation': 'steepest',
                        'best_obj': improved_result.first.energy, 
                        'time': self.default_answer['time'] + improvement_time,
                        'solution': improved_result
                    }
                    if isinstance(self.answer, list):
                        self.answer.append(sd_answer)
                    else:
                        self.answer = [self.answer, sd_answer]
                return self.answer
            except SolverFailureError as e:
                self._handle_solver_failure(e)
            except Exception as e:
                self._handle_unexpected_error(e)
                raise



class SimulatedAnnealingSolver(QUBOSolver):
    """A wrapper for the SimulatedAnnealingSampler."""

    def __init__(self):
        super().__init__()
        self.sampler = SimulatedAnnealingSampler()
        
    def solve_bqm(self, bqm: BinaryQuadraticModel, params: Optional[Dict] = None) -> dimod.SampleSet:
        """
        Solve the QUBO problem using Simulated Annealing.

        Args:
            bqm (BinaryQuadraticModel): The QUBO problem to solve.
            params (Optional[Dict]): Additional parameters for solving the problem.

        Returns:
            dimod.SampleSet: The samples and information about the solver.
        """
        params = params or {}
        if 'num_reads' not in params: 
            params['num_reads'] = 100

        # response = self.sampler.sample(bqm, **params)
        # self.answer = { 'best_obj': response.first.energy,
        #                 'time':  sum([int(t) for t in response.info['timing'].values()]) * 1e-9,  # Convert time to seconds
        #                 'solution': response}
        
        response = {}
        if not params.get('chain_break_method', False):                
            response = self.sampler.sample(bqm, **params)
            self.answer = { 'best_obj'   :  response.first.energy,
                            'time': sum([int(t) for t in response.info['timing'].values()]) * 1e-9,  # Convert time to seconds
                            'solution': response}
            self.default_answer = self.answer
        else:
            # Return a collection of samples using different chain_break_methods
            embedding = params.get('embedding', None)
            if not embedding:
                raise ValueError("Embedding must be provided for chain break methods.")
            params.pop('embedding')  
            result = ising_util.solve_bqm_with_chain_break_methods(bqm, embedding, self.sampler, **params)
            
            # Return the best energy from majority_vote as the best energy
            qpu_time = sum([int(t) for t in result['embedded'].info['timing'].values()]) * 1e-9  # Convert time to seconds
            # result.pop('embedded') 
            self.answer = []
            for name, res in result.items():
                self.answer.append({
                    'variation': name,
                    'solution': res,
                    'best_obj': res.first.energy,
                    'time': qpu_time + (res.info['timing'].get('chain_break_time', 0) if params.get('chain_break_time', False) else 0)
                })
                if name == 'majority_vote':
                    self.default_answer = self.answer[-1]

        # Steepest descent post-processing 
        if params.get('steepest_descent', False):
            start_time = time.time()
            improved_result = SteepestDescentSolver().sample(bqm, initial_states=self.default_answer['solution'])
            improvement_time = time.time() - start_time
            sd_answer = {
                'variation': 'steepest',
                'best_obj': improved_result.first.energy, 
                'time': self.default_answer['time'] + improvement_time,
                'solution': improved_result
            }
            if isinstance(self.answer, list):
                self.answer.append(sd_answer)
            else:
                self.answer = [self.answer, sd_answer]
        return self.answer



class GurobiSolver(QUBOSolver):
    """A wrapper for the Gurobi solver."""

    def __init__(self):
        super().__init__()
        self.best_solution_time: Optional[float] = None
        self.start_time: float = 0

    def mycallback(self, model: gp.Model, where: int) -> None:
        """
        Callback function for Gurobi solver to track the best solution.

        Args:
            model (gp.Model): The Gurobi model being solved.
            where (int): Indicates where in the solution process the callback is called.
        """
        if where == gp.GRB.Callback.MIP:
            current_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            if current_obj < self.best_obj:
                self.best_obj = current_obj
                self.best_solution_time = time.time() - self.start_time

    def solve_bqm(self, bqm: BinaryQuadraticModel, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solves a Binary Quadratic Model (BQM) using the Gurobi optimizer.

        Args:
            bqm (BinaryQuadraticModel): The BQM to be solved.
            params (Optional[Dict]): Parameters for the solver, including 'time_limit'.

        Returns:
            Dict[str, Any]: A dictionary containing the solution, energy, running time, gap,
            the lower bound, and the CPU time when the best solution is found.
        """
        params = params or {}
        was_spin = bqm.vartype == dimod.SPIN
        if was_spin:
            bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

        model = self._create_gurobi_model(bqm, params)
        self._setup_and_solve_model(model)

        return self._extract_results(model, bqm, was_spin)

    def _create_gurobi_model(self, bqm: BinaryQuadraticModel, params: Dict) -> gp.Model:
        """
        Create and set up the Gurobi model based on the BQM.

        Args:
            bqm (BinaryQuadraticModel): The BQM to be solved.
            params (Dict): Additional parameters for the Gurobi solver.

        Returns:
            gp.Model: The created Gurobi model.
        """
        model = gp.Model("BQM")
        self._set_gurobi_parameters(model, params)
        x = self._create_variables(model, bqm)
        objective = self._create_objective(bqm, x)
        model.setObjective(objective, gp.GRB.MINIMIZE)
        return model

    def _set_gurobi_parameters(self, model: gp.Model, params: Dict) -> None:
        """
        Set Gurobi solver parameters based on the provided dictionary.

        Args:
            model (gp.Model): The Gurobi model to configure.
            params (Dict): Dictionary of parameter settings.
        """
        gurobi_params = ["TimeLimit", "Threads", "OutputFlag", "Method"]
        for param in gurobi_params:
            if param in params:
                model.setParam(param, params[param])
        model.setParam("MIPFocus", 1)

    def _create_variables(self, model: gp.Model, bqm: BinaryQuadraticModel) -> Dict[Any, gp.Var]:
        """
        Create binary variables in the Gurobi model for each variable in the BQM.

        Args:
            model (gp.Model): The Gurobi model.
            bqm (BinaryQuadraticModel): The Binary Quadratic Model.

        Returns:
            Dict[Any, gp.Var]: A dictionary mapping BQM variables to Gurobi variables.
        """
        return {var: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{var}") for var in bqm.variables}

    def _create_objective(self, bqm: BinaryQuadraticModel, x: Dict[Any, gp.Var]) -> gp.QuadExpr:
        """
        Create the objective function for the Gurobi model based on the BQM.

        Args:
            bqm (BinaryQuadraticModel): The Binary Quadratic Model.
            x (Dict[Any, gp.Var]): Dictionary mapping BQM variables to Gurobi variables.

        Returns:
            gp.QuadExpr: The quadratic expression representing the objective function.
        """
        linear_expr = gp.LinExpr(sum(bqm.linear[var] * x[var] for var in bqm.variables))
        quadratic_expr = gp.QuadExpr(sum(coeff * x[var_i] * x[var_j] 
                                         for (var_i, var_j), coeff in bqm.quadratic.items()))
        return linear_expr + quadratic_expr

    def _setup_and_solve_model(self, model: gp.Model) -> None:
        """
        Set up the callback and solve the Gurobi model.

        Args:
            model (gp.Model): The Gurobi model to solve.
        """
        self.best_solution_time = None
        self.best_obj = float('inf')  # Assuming minimization
        self.start_time = time.time()
        model.optimize(self.mycallback)

    def _extract_results(self, model: gp.Model, bqm: BinaryQuadraticModel, was_spin: bool) -> Dict[str, Any]:
        """
        Extract and format the results from the solved Gurobi model.

        Args:
            model (gp.Model): The solved Gurobi model.
            bqm (BinaryQuadraticModel): The original Binary Quadratic Model.
            was_spin (bool): Whether the original BQM was in SPIN format.

        Returns:
            Dict[str, Any]: A dictionary containing the solution details.
        """
        solution, obj, gap, lbound = None, None, None, None
        runtime = time.time() - self.start_time

        if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
            solution = self._format_solution(model, was_spin)
            obj = model.objVal + bqm.offset
            self.best_obj = obj
            self.time = runtime
            lbound = model.objBound + bqm.offset
            gap = abs(obj - lbound)/abs(obj) if model.status == gp.GRB.TIME_LIMIT else 0

        return {
            "solution": solution,
            "best_obj": obj,
            "time": runtime,
            "gap": gap,
            "lbound": lbound,
            "best_solution_time": self.best_solution_time
        }

    def _format_solution(self, model: gp.Model, was_spin: bool) -> Dict[Any, Union[int, float]]:
        """
        Format the solution based on the original BQM variable type.

        Args:
            model (gp.Model): The solved Gurobi model.
            was_spin (bool): Whether the original BQM was in SPIN format.

        Returns:
            Dict[Any, Union[int, float]]: The formatted solution.
        """
        solution = {v.VarName: (2 * int(round(v.X)) - 1) if was_spin else v.X 
                    for v in model.getVars()}
        return solution



def solve_qubo_problem(bqm: BinaryQuadraticModel, 
                        solver_type: str = 'dwave', 
                        solver_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Solve a QUBO problem using the specified solver.

    Args:
        bqm (BinaryQuadraticModel): The QUBO problem to solve.
        solver_type (str): Type of solver to use ('dwave', 'simulated_annealing', or 'gurobi').
        solver_params (Optional[Dict]): Additional parameters for the solver.

    Returns:
        Dict[str, Any]: The solution and related information.

    Raises:
        ValueError: If an invalid solver type is specified.
    """
    solver_params = solver_params or {}

    if solver_type == 'dwave':
        solver = DWaveSolver(solver_params.get('token_list', []), 
                            solver_params.get('solver_id', 'Advantage_system6.4'))
    elif solver_type == 'simulated_annealing':
        solver = SimulatedAnnealingSolver()
    elif solver_type == 'gurobi':
        solver = GurobiSolver()
    else:
        raise ValueError(f"Invalid solver type: {solver_type}")

    return solver.solve_bqm(bqm, solver_params)

def run_improved_spectral(adj, args):
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    
    solver = ImprovedSpectralSolver(adj, device=device)
    
    method = getattr(args, 'variant', 'grad')
    
    print(f"[*] Running Improved Spectral Solver with method: {method}")
    
    if method == 'sdp':
        spins, cut = solver.solve_sdp_proxy(n_rounding=100)
        
    elif method == 'iter':
        spins, cut = solver.solve_iterative(max_iter=getattr(args, 'max_iter', 50))
        
    else: 
        solver.solve_sdp_proxy(n_rounding=10) 
        
        lr = getattr(args, 'lr', 0.05)
        steps = getattr(args, 'steps', 100)
        
        spins, cut, _ = solver.solve_gradient_descent(lr=lr, steps=steps)

    return cut, spins
