"""
QQA (Quasi-Quantum Annealing) Solver for Ising problems
Uses QQA4CO library for optimization
"""

import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp

import torch

# Add QQA4CO to path for importing
_qqa4co_path = Path(__file__).parent / "QQA4CO" / "src"
if str(_qqa4co_path) not in sys.path:
    sys.path.insert(0, str(_qqa4co_path))

try:
    import qqa  # QQA4CO must be importable
except ImportError as e:
    raise ImportError(
        f"Could not import 'qqa' from {_qqa4co_path}. "
        "Make sure QQA4CO is in models/QQA4CO/src/"
    ) from e


class IsingProblem:
    """
    QQA problem wrapper for the Ising model.

    Variables: x in [0,1]^n (continuous, handled by QQA)
    Mapping:  s = 2x - 1  (spins in [-1,1])
    Energy:   E(s) = -0.5 * s^T J s  (J symmetric, diag 0)
    """

    def __init__(self, J, device="cpu"):
        """
        Parameters
        ----------
        J : np.ndarray or scipy.sparse
            Coupling matrix (n x n), symmetric, zero diagonal.
        device : str or torch.device
            'cpu' or 'cuda'.
        """
        # Normalize device to include index for CUDA devices (e.g., 'cuda' -> 'cuda:0')
        device_obj = torch.device(device)
        if device_obj.type == 'cuda' and device_obj.index is None:
            # If CUDA device without index, use the default CUDA device (usually cuda:0)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device_obj
        self.dtype = torch.float32
        self.J_dense = None
        self.J_sparse = None
        self.J_sparse_csr = None
        self.J_sparse_T_mat = None
        self._dense_sparse_matmul = False

        if sp.issparse(J):
            J_coo = J.tocoo()
            indices = np.vstack((J_coo.row, J_coo.col)).astype(np.int64)
            values = J_coo.data.astype(np.float32)
            self.J_sparse = torch.sparse_coo_tensor(
                torch.from_numpy(indices),
                torch.from_numpy(values),
                torch.Size(J_coo.shape),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()
            self.J_sparse_csr = self.J_sparse.to_sparse_csr()
            try:
                self.J_sparse_T_mat = (
                    self.J_sparse.transpose(0, 1).coalesce().to_sparse_csr()
                )
                _probe = torch.zeros(
                    (1, J_coo.shape[0]), device=self.device, dtype=self.dtype
                )
                torch.matmul(_probe, self.J_sparse_T_mat)
                self._dense_sparse_matmul = True
            except RuntimeError:
                self.J_sparse_T_mat = None
                self._dense_sparse_matmul = False
            self.J_dense = None
            self.n = J_coo.shape[0]
        else:
            J = np.asarray(J, dtype=np.float32)
            self.J_dense = torch.from_numpy(J).to(self.device)
            self.n = J.shape[0]

        self.num_nodes = self.n

    def loss_fn(self, x):
        """
        QQA minimizes this loss over x in [0,1]^n (batch of solutions).

        x: Tensor of shape (batch_size, n), values in [0,1].
        Returns: Tensor of shape (batch_size,) with energies.
        """
        # Normalize device comparison: compare type and index separately
        # Handle case where one device is 'cuda' (no index) and other is 'cuda:0'
        x_device = x.device
        x_idx = x_device.index if x_device.index is not None else 0
        self_idx = self.device.index if self.device.index is not None else 0
        
        if x_device.type != self.device.type or (x_device.type == 'cuda' and x_idx != self_idx):
            raise ValueError(
                f"Input tensor is on {x.device}, but IsingProblem expects {self.device}. "
                "Please ensure the QQA run and IsingProblem share the same device."
            )
        s = (2.0 * x - 1.0).to(dtype=self.dtype)  # Map to spins in [-1, 1]
        Js = self._apply_J(s)
        return -0.5 * torch.sum(s * Js, dim=1)

    def _apply_J(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Multiply a batch of vectors by J.

        vectors: Tensor of shape (batch, n)
        """
        if self.J_sparse is not None:
            if self._dense_sparse_matmul and self.J_sparse_T_mat is not None:
                try:
                    return torch.matmul(vectors, self.J_sparse_T_mat)
                except RuntimeError:
                    # Backend might not support dense@sparse (e.g., CPU without CSR support).
                    self._dense_sparse_matmul = False
                    self.J_sparse_T_mat = None
            vectors_T = vectors.transpose(0, 1).contiguous()
            sparse_mat = self.J_sparse_csr if self.J_sparse_csr is not None else self.J_sparse
            product = torch.sparse.mm(sparse_mat, vectors_T)
            return product.transpose(0, 1).contiguous()
        return torch.matmul(vectors, self.J_dense)

    def energy_from_spins(self, spins):
        """
        Compute energies for spin configurations s in {-1, +1}^n.

        spins: array-like or tensor of shape (n,) or (batch, n)
        Returns: tensor of energies (batch,) or scalar if single spin vector.
        """
        spins_tensor = torch.as_tensor(spins, dtype=self.dtype, device=self.device)
        if spins_tensor.dim() == 1:
            spins_tensor = spins_tensor.unsqueeze(0)
        Js = self._apply_J(spins_tensor)
        energies = -0.5 * torch.sum(spins_tensor * Js, dim=1)
        return energies.squeeze(0) if energies.numel() == 1 else energies


class QQASolver:
    """
    Quasi-Quantum Annealing (QQA) solver wrapper for Ising problems.

    Uses QQA4CO (ICLR 2025) on the Ising model energy:
        E(s) = -0.5 * s^T J s,  s in {-1, +1}^n
    """

    def __init__(
        self,
        instance_name: str,
        dataset: str,
        random_seed: int = None,
        device: str = None,
        # QQA hyperparameters (defaults can be adjusted)
        sol_size: int = 100,
        learning_rate: float = 1.0,
        temp: float = 1e-3,
        min_bg: float = -3.0,
        max_bg: float = 0.1,
        curve_rate: float = 4.0,
        div_param: float = 0.2,
        num_epochs: int = 3000,
        check_interval: int = 500,
        plot_dynamics: bool = False,
        run_name: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        instance_name : str
            Name of the instance file (for logging).
        dataset : str
            Name of the dataset (results path).
        random_seed : int, optional
            Global seed (numpy + torch).
        device : str or None
            'cuda' or 'cpu'. If None → 'cuda' if GPU available.
        sol_size, learning_rate, temp, min_bg, max_bg, curve_rate, div_param :
            QQA hyperparameters (see QQA4CO README).
        num_epochs : int
            Total number of annealing iterations.
        check_interval : int
            Frequency of internal QQA logging.
        plot_dynamics : bool
            If True, QQA plots energy curves.
        run_name : str, optional
            Solver name for results folder.
        """
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sol_size = sol_size
        self.learning_rate = learning_rate
        self.temp = temp
        self.min_bg = min_bg
        self.max_bg = max_bg
        self.curve_rate = curve_rate
        self.div_param = div_param
        self.num_epochs = num_epochs
        self.check_interval = check_interval
        self.plot_dynamics = plot_dynamics

        self.solver_name = run_name if run_name else "qqa"

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        print(
            f"Initialized {self.solver_name} solver (QQA). "
            f"Results will be saved under '{self.solver_name}'."
        )
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  device: {self.device}")
        print(f"  sol_size: {self.sol_size}")
        print(f"  learning_rate: {self.learning_rate}")
        print(f"  temp: {self.temp}")
        print(f"  min_bg: {self.min_bg}")
        print(f"  max_bg: {self.max_bg}")
        print(f"  curve_rate: {self.curve_rate}")
        print(f"  div_param: {self.div_param}")
        print(f"  num_epochs: {self.num_epochs}")
        print(f"  check_interval: {self.check_interval}")
        print(f"  plot_dynamics: {self.plot_dynamics}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J):
        """
        Execute QQA on the Ising model defined by J.

        Parameters
        ----------
        J : np.ndarray or scipy.sparse
            Coupling matrix (n x n), symmetric, diag ~ 0.

        Returns
        -------
        energy : float
            Ising energy of the final solution.
        spins : np.ndarray (n,)
            Spins in {-1, +1}.
        time_taken : float
            Total execution time (s).
        """
        start_time = time.time()

        problem = IsingProblem(J, device=self.device)
        n = problem.num_nodes

        print(f"[DEBUG] Starting QQA with n={n}, sol_size={self.sol_size}, device={self.device}")

        # Normalize device string to match IsingProblem's device (ensure explicit index for CUDA)
        device_str = str(problem.device)  # Use problem's normalized device

        best_sol, best_obj, runtime, x_final = qqa.batch_annealing(
            problem,
            sol_size=self.sol_size,
            learning_rate=self.learning_rate,
            temp=self.temp,
            min_bg=self.min_bg,
            max_bg=self.max_bg,
            curve_rate=self.curve_rate,
            div_param=self.div_param,
            num_epochs=self.num_epochs,
            check_interval=self.check_interval,
            device=device_str,
            plot_dynamics=self.plot_dynamics,
        )

        candidate_batches = []
        batch_tags = []
        source_lengths = {}

        def _append_batch(tag: str, tensor: torch.Tensor):
            if tensor is None or not isinstance(tensor, torch.Tensor):
                return
            data = tensor.detach()
            if data.dim() == 1:
                data = data.unsqueeze(0)
            data = data.to(device=self.device, dtype=problem.dtype)
            candidate_batches.append(data)
            batch_tags.extend([tag] * data.shape[0])
            source_lengths[tag] = source_lengths.get(tag, 0) + data.shape[0]

        _append_batch("best_sol_snapshot", best_sol)
        _append_batch("final_batch", x_final)

        if not candidate_batches:
            raise RuntimeError("QQA did not return candidate solutions to evaluate.")

        x_batch = torch.cat(candidate_batches, dim=0)
        print(
            f"[DEBUG] Aggregated {x_batch.shape[0]} candidates from sources {source_lengths}"
        )
        print(
            f"[DEBUG] x_batch dtype/device: {x_batch.dtype}/{x_batch.device}, "
            f"range: [{x_batch.min().item():.4f}, {x_batch.max().item():.4f}]"
        )

        # Evaluate all solutions in batch and find the best
        with torch.no_grad():
            # Round according to paper: x_round = x.round()
            x_round = x_batch.round()
            print(f"[DEBUG] x_round range: [{x_round.min().item():.4f}, {x_round.max().item():.4f}]")
            
            # Compute energies for all rounded solutions
            losses_round = problem.loss_fn(x_round)
            best_idx = torch.argmin(losses_round)
            x_best_round = x_round[best_idx]
            
            best_origin = batch_tags[best_idx.item()]
            print(f"[DEBUG] Best solution index: {best_idx.item()} (source={best_origin})")
            print(f"[DEBUG] Best loss (from rounded): {losses_round[best_idx].item():.6f}")
            print(f"[DEBUG] QQA reported best_obj: {best_obj:.6f}")

        # Convert to spins using paper method: spins = 2*x_round - 1
        with torch.no_grad():
            spins_tensor = 2.0 * x_best_round - 1.0
            print(f"[DEBUG] spins_tensor range: [{spins_tensor.min().item():.4f}, {spins_tensor.max().item():.4f}]")
            print(f"[DEBUG] Unique spin values: {torch.unique(spins_tensor).tolist()}")
            
            energy = float(problem.energy_from_spins(spins_tensor).item())
            print(f"[DEBUG] Final energy: {energy:.6f}")

        spins = spins_tensor.cpu().numpy().astype(np.int8)

        time_taken = time.time() - start_time

        print(f"QQA finished. Best energy: {energy:.6f}")
        print(f"Best objective (QQA loss): {float(best_obj):.6f}")
        print(f"Runtime reported by QQA: {float(runtime):.4f}s")
        print(f"Total wall-clock time: {time_taken:.4f}s")

        self._store_results(
            energy=energy,
            spins=spins,
            time_taken=time_taken,
            best_obj=float(best_obj),
            qqa_runtime=float(runtime),
        )

        return energy, spins, time_taken

    def _store_results(
        self,
        energy: float,
        spins: np.ndarray,
        time_taken: float,
        best_obj: float,
        qqa_runtime: float,
    ):
        """
        Save results to CSV, same as other solvers.
        """
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        instance_base = (
            self.instance_name.replace(".txt", "")
            .replace(".bin", "")
            .replace(".gz", "")
        )
        csv_file = results_dir / f"{instance_base}.csv"

        result_data = {
            "instance_name": self.instance_name,
            "dataset": self.dataset,
            "seed": self.random_seed,
            "solver_name": self.solver_name,
            "energy": energy,
            "time": time_taken,
            "qqa_best_obj": best_obj,
            "qqa_runtime": qqa_runtime,
            "sol_size": self.sol_size,
            "learning_rate": self.learning_rate,
            "temp": self.temp,
            "min_bg": self.min_bg,
            "max_bg": self.max_bg,
            "curve_rate": self.curve_rate,
            "div_param": self.div_param,
            "num_epochs": self.num_epochs,
            "check_interval": self.check_interval,
            "device": str(self.device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()

        if not df.empty:
            duplicate_mask = (
                (df["instance_name"] == result_data["instance_name"])
                & (df["dataset"] == result_data["dataset"])
                & (df["seed"] == result_data["seed"])
                & (df["solver_name"] == result_data["solver_name"])
            )

            if duplicate_mask.any():
                df.loc[duplicate_mask, list(result_data.keys())] = list(
                    result_data.values()
                )
            else:
                df = pd.concat(
                    [df, pd.DataFrame([result_data])], ignore_index=True
                )
        else:
            df = pd.DataFrame([result_data])

        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"Results saved to: {csv_file}")
        print(f"Energy: {energy:.6f}")
        print(f"Time: {time_taken:.4f}s")
        print("-" * 40)

