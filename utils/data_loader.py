"""
Data loading for multiple dataset formats
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.sparse import coo_matrix, csr_matrix
import warnings
import sys
import os

# Add parent directory to path to import read_binary_ising
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_binary_ising import read_ising_auto


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


class DataLoader:
    """Data loader supporting multiple dataset formats"""
    
    def __init__(self, dataset_name: str, datasets_path: str = "datasets"):
        self.dataset_name = dataset_name
        self.datasets_path = Path(datasets_path)
        self.dataset_path = self.datasets_path / dataset_name
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Dynamic format detection - no longer hardcoded
        self.dataset_format = self._detect_dataset_format()
    
    def _detect_dataset_format(self) -> str:
        """Detect dataset format based on dataset name and available files"""
        # if dataset_name contains substring "Gset" or "gset" return 'gset'
        if "Gset" in self.dataset_name or "gset" in self.dataset_name:
            return 'gset'
        else:
            # Check if dataset contains binary files
            has_binary = any(f.suffix in ['.bin', '.gz'] for f in self.dataset_path.glob('*.*'))
            if has_binary:
                return 'binary'
            else:
                # For all other datasets, use linear bias format
                return 'graph_linear_bias'
    
    def list_instances(self) -> List[str]:
        """List all available instances in the dataset"""
        if self.dataset_name == "Gset":
            instances = [f.name for f in self.dataset_path.glob("*.txt")]
        elif self.dataset_name == "hand_test2":
            instances = [f.name for f in self.dataset_path.glob("*.txt") if not f.name.endswith("_embedding.json")]
        else:
            # Support multiple file formats: .txt, .bin, .bin.gz
            txt_files = [f.name for f in self.dataset_path.glob("*.txt")]
            bin_files = [f.name for f in self.dataset_path.glob("*.bin")]
            bin_gz_files = [f.name for f in self.dataset_path.glob("*.bin.gz")]
            instances = txt_files + bin_files + bin_gz_files
        instances.sort()
        return instances
    
    def load_instance(self, instance_name: str) -> csr_matrix:
        """Load an Ising instance and return the coupling matrix J as CSR sparse"""
        instance_path = self.dataset_path / instance_name
        if not instance_path.exists():
            raise FileNotFoundError(f"Instance file not found: {instance_path}")
        
        # Use dynamically detected format
        if self.dataset_format == 'gset':
            return self._load_gset_instance(instance_path)
        elif self.dataset_format == 'binary':
            return self._load_binary_instance(instance_path)
        elif self.dataset_format == 'graph_linear_bias':
            return self._load_linear_bias_instance(instance_path)
        else:
            return self._load_dense_matrix(instance_path)
    
    def _robust_parse_line(self, line: str, expected_parts: int = 2) -> Optional[Tuple]:
        """Robustly parse a line into numeric values"""
        try:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                return None
                
            parts = line.split()
            if len(parts) < expected_parts:
                return None
            
            if expected_parts == 2:
                i = int(float(parts[0]))
                j = int(float(parts[1]))
                return (i, j, 1.0)
            else:
                i = int(float(parts[0]))
                j = int(float(parts[1]))
                w = float(parts[2])
                return (i, j, w)
        except (ValueError, IndexError, TypeError):
            return None
    
    def _load_gset_instance(self, file_path: Path) -> csr_matrix:
        """Load Gset format instance"""
        try:
            with open(file_path, 'r') as f:
                n, m = map(int, f.readline().strip().split())
                
                rows, cols, data = [], [], []
                for line in f:
                    # parsed = self._robust_parse_line(line, expected_parts=2)
                    parsed = self._robust_parse_line(line, expected_parts=3)
                    if parsed is not None:
                        i, j, w = parsed
                        i, j = i - 1, j - 1
                        if 0 <= i < n and 0 <= j < n and i != j:
                            rows.extend([i, j])
                            cols.extend([j, i])
                            data.extend([-w, -w])
                
                J = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
                return self._remove_isolated_nodes(J)
        except Exception as e:
            raise DataLoadError(f"Error reading Gset file {file_path}: {e}")
    
    def _load_linear_bias_instance(self, file_path: Path) -> csr_matrix:
        """Load graph with linear bias format"""
        try:
            with open(file_path, 'r') as f:
                n, m = map(int, f.readline().strip().split())
                
                node_weights = [float(f.readline().strip()) for _ in range(n)]
                
                edges, weights = [], []
                for line in f:
                    parsed = self._robust_parse_line(line, expected_parts=3)
                    if parsed is not None:
                        i, j, w = parsed
                        i, j = i - 1, j - 1
                        if 0 <= i < n and 0 <= j < n and i != j:
                            edges.append((i, j))
                            weights.append(w)
                
                add_bias_node = any(h != 0 for h in node_weights)
                new_node_id = n
                n_vertices = n + 1 if add_bias_node else n
                
                if add_bias_node:
                    for i, h in enumerate(node_weights):
                        if h != 0:
                            edges.append((new_node_id, i))
                            weights.append(h)
                
                rows, cols, data = [], [], []
                for (i, j), w in zip(edges, weights):
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([w, w])
                
                J = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
                return self._remove_isolated_nodes(J)
        except Exception as e:
            raise DataLoadError(f"Error reading linear bias file {file_path}: {e}")
    
    def _load_dense_matrix(self, file_path: Path) -> csr_matrix:
        """Load dense matrix format (fallback)"""
        try:
            J = np.loadtxt(file_path)
            if not np.allclose(J, J.T):
                J = (J + J.T) / 2
            J_sparse = csr_matrix(J)
            return self._remove_isolated_nodes(J_sparse)
        except Exception as e:
            raise DataLoadError(f"Error reading dense matrix file {file_path}: {e}")
    
    def _remove_isolated_nodes(self, J: csr_matrix) -> csr_matrix:
        """
        Remove isolated nodes (rows/columns with all zeros) from the coupling matrix.
        Returns the reduced matrix with isolated nodes removed.
        """
        if J.nnz == 0:
            # If matrix is completely empty, return as is
            return J
        
        # Find nodes that have at least one connection
        # For symmetric matrix, we only need to check row sums
        row_sums = np.array(np.abs(J).sum(axis=1)).flatten()
        connected_nodes = np.where(row_sums != 0)[0]
        
        # Check if any isolated nodes were found
        n_isolated = J.shape[0] - len(connected_nodes)
        if n_isolated > 0:
            isolated_nodes = np.where(row_sums == 0)[0]
            warnings.warn(
                f"Found {n_isolated} isolated node(s) at indices {isolated_nodes.tolist()}. "
                f"These nodes will be removed as they don't affect the optimization problem. "
                f"Matrix size reduced from {J.shape[0]} to {len(connected_nodes)} nodes.",
                UserWarning
            )
            
            # Extract submatrix with only connected nodes
            J_reduced = J[np.ix_(connected_nodes, connected_nodes)]
            return J_reduced
        
        return J
    
    def _load_binary_instance(self, file_path: Path) -> csr_matrix:
        """Load binary format instance using read_binary_ising functions"""
        try:
            n, m, h_values, edges = read_ising_auto(str(file_path))
            
            # Convert to numpy arrays for easier handling
            h_array = np.array(h_values, dtype=float)
            
            # Check if we need to add a bias node
            add_bias_node = any(h != 0 for h in h_array)
            new_node_id = n
            n_vertices = n + 1 if add_bias_node else n
            
            # Prepare edge data
            rows, cols, data = [], [], []
            
            # Add original edges
            for i, j, weight in edges:
                if 0 <= i < n and 0 <= j < n and i != j:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([weight, weight])
            
            # Add bias node connections if needed
            if add_bias_node:
                for i, h in enumerate(h_array):
                    if h != 0:
                        rows.extend([new_node_id, i])
                        cols.extend([i, new_node_id])
                        data.extend([h, h])
            
            # Create sparse matrix
            J = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
            return self._remove_isolated_nodes(J)
            
        except Exception as e:
            raise DataLoadError(f"Error reading binary file {file_path}: {e}")
    
    def get_instance_info(self, instance_name: str) -> dict:
        """Get information about an instance"""
        J = self.load_instance(instance_name)
        return {
            'name': instance_name,
            'size': J.shape[0],
            'density': J.nnz / (J.shape[0] ** 2),
            'max_weight': J.data.max() if J.nnz > 0 else 0,
            'min_weight': J.data.min() if J.nnz > 0 else 0,
            'mean_weight': J.data.mean() if J.nnz > 0 else 0,
        }
