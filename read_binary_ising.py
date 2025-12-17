import struct
import numpy as np
import gzip

def read_binary_ising(filepath):
    """
    Read uncompressed binary Ising instance file.
    
    Format:
    - 4 bytes: n (number of nodes)
    - 4 bytes: m (number of edges)
    - n * 4 bytes: h values (linear biases)
    - m * 3 * 4 bytes: edge data (i, j, weight) for each edge
    
    Returns:
    --------
    n : int
        Number of nodes
    m : int
        Number of edges
    h_values : list
        Linear biases
    edges : list of tuples
        Edge data as (i, j, weight)
    """
    with open(filepath, 'rb') as f:
        # Read header
        n, m = struct.unpack('ii', f.read(8))
        
        # Read linear biases
        h_values = list(struct.unpack(f'{n}i', f.read(4 * n)))
        
        # Read quadratic biases
        edges = []
        for _ in range(m):
            i, j, weight = struct.unpack('iii', f.read(12))
            edges.append((i, j, weight))
    
    return n, m, h_values, edges

def read_binary_ising_csr_compressed(filepath):
    """
    Read CSR format compressed binary Ising instance file.
    
    Format (gzipped):
    - 4 bytes: magic number 'CSZ1'
    - 4 bytes: n (number of nodes)
    - 4 bytes: m (number of edges)
    - n * 4 bytes: h values (linear biases)
    - (n+1) * 4 bytes: row_ptr (CSR row pointers)
    - m * 4 bytes: col_idx (CSR column indices)
    - m * 4 bytes: weights (edge weights)
    
    Returns:
    --------
    n : int
        Number of nodes
    m : int
        Number of edges
    h_values : numpy array
        Linear biases
    edges : list of tuples
        Edge data as (i, j, weight) reconstructed from CSR
    """
    with gzip.open(filepath, 'rb') as f:
        # Read and verify magic number
        magic = f.read(4)
        if magic != b'CSZ1':
            raise ValueError(f"Invalid magic number: {magic}. Expected b'CSZ1'")
        
        # Read header
        n, m = struct.unpack('ii', f.read(8))
        
        # Read linear biases
        h_values = np.frombuffer(f.read(n * 4), dtype=np.int32)
        
        # Read CSR data
        row_ptr = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32)
        col_idx = np.frombuffer(f.read(m * 4), dtype=np.int32)
        weights = np.frombuffer(f.read(m * 4), dtype=np.int32)
        
        # Reconstruct edges from CSR format
        edges = []
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            for k in range(start, end):
                j = col_idx[k]
                weight = weights[k]
                edges.append((i, j, weight))
    
    return n, m, h_values, edges

def read_binary_ising_compressed(filepath):
    """
    Read compressed binary Ising instance file (simple gzip, not CSR).
    
    Format (gzipped):
    - 4 bytes: magic number 'ISGZ'
    - 4 bytes: n (number of nodes)
    - 4 bytes: m (number of edges)
    - n * 4 bytes: h values
    - m * 4 bytes: edge_i
    - m * 4 bytes: edge_j
    - m * 4 bytes: weights
    """
    with gzip.open(filepath, 'rb') as f:
        # Read and verify magic number
        magic = f.read(4)
        
        if magic == b'ISGZ':
            # Simple compressed format (maintains order)
            n, m = struct.unpack('ii', f.read(8))
            
            h_values = np.frombuffer(f.read(n * 4), dtype=np.int32)
            edge_i = np.frombuffer(f.read(m * 4), dtype=np.int32)
            edge_j = np.frombuffer(f.read(m * 4), dtype=np.int32)
            weights = np.frombuffer(f.read(m * 4), dtype=np.int32)
            
            edges = [(edge_i[k], edge_j[k], weights[k]) for k in range(m)]
            return n, m, h_values, edges
            
        elif magic == b'CSZ1':
            # CSR compressed format (original implementation)
            return read_binary_ising_csr_compressed_internal(filepath)
        else:
            raise ValueError(f"Invalid magic number: {magic}")

def read_binary_ising_csr_compressed_internal(filepath):
    """Read CSR format (kept for backward compatibility)."""
    with gzip.open(filepath, 'rb') as f:
        magic = f.read(4)
        n, m = struct.unpack('ii', f.read(8))
        
        h_values = np.frombuffer(f.read(n * 4), dtype=np.int32)
        row_ptr = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32)
        col_idx = np.frombuffer(f.read(m * 4), dtype=np.int32)
        weights = np.frombuffer(f.read(m * 4), dtype=np.int32)
        
        edges = []
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            for k in range(start, end):
                j = col_idx[k]
                weight = weights[k]
                edges.append((i, j, weight))
    
    return n, m, h_values, edges

def read_ising_auto(filepath):
    """
    Automatically detect format and read Ising instance.
    
    Supports:
    - Uncompressed binary (.bin)
    - Compressed CSR binary (.bin.gz or .gz)
    - Text format (.txt)
    
    Returns:
    --------
    n : int
        Number of nodes
    m : int
        Number of edges
    h_values : list or numpy array
        Linear biases
    edges : list of tuples
        Edge data as (i, j, weight)
    """
    if filepath.endswith('.gz'):
        return read_binary_ising_compressed(filepath)
    elif filepath.endswith('.bin'):
        return read_binary_ising(filepath)
    elif filepath.endswith('.txt'):
        return read_text_ising(filepath)
    else:
        # Try to auto-detect
        try:
            return read_binary_ising_compressed(filepath)
        except:
            try:
                return read_binary_ising(filepath)
            except:
                return read_text_ising(filepath)

def read_text_ising(filepath):
    """
    Read text format Ising instance file.
    
    Format:
    <n> <m>
    <h_1>
    ...
    <h_n>
    <i> <j> <weight>
    ...
    
    Returns:
    --------
    n : int
        Number of nodes
    m : int
        Number of edges
    h_values : list
        Linear biases
    edges : list of tuples
        Edge data as (i, j, weight) with 0-based indexing
    """
    with open(filepath, 'r') as f:
        # Read header
        n, m = map(int, f.readline().split())
        
        # Read linear biases
        h_values = []
        for _ in range(n):
            h_values.append(int(f.readline().strip()))
        
        # Read edges (convert from 1-based to 0-based indexing)
        edges = []
        for _ in range(m):
            i, j, weight = map(int, f.readline().split())
            edges.append((i - 1, j - 1, weight))
    
    return n, m, h_values, edges

def verify_ising_instance(filepath1, filepath2):
    """
    Verify two Ising instance files contain the same data.
    
    Useful for checking compressed vs uncompressed formats.
    """
    print(f"Reading {filepath1}...")
    n1, m1, h1, edges1 = read_ising_auto(filepath1)
    
    print(f"Reading {filepath2}...")
    n2, m2, h2, edges2 = read_ising_auto(filepath2)
    
    # Check header
    if n1 != n2 or m1 != m2:
        print(f"❌ MISMATCH: Header differs")
        print(f"  File 1: n={n1}, m={m1}")
        print(f"  File 2: n={n2}, m={m2}")
        return False
    
    print(f"✓ Header matches: n={n1}, m={m1}")
    
    # Check h values
    h1_array = np.array(h1)
    h2_array = np.array(h2)
    if not np.array_equal(h1_array, h2_array):
        print(f"❌ MISMATCH: h values differ")
        diff_count = np.sum(h1_array != h2_array)
        print(f"  {diff_count}/{n1} values differ")
        return False
    
    print(f"✓ h values match (all {n1} values)")
    
    # Sort edges for comparison (CSR may reorder them)
    edges1_sorted = sorted(edges1)
    edges2_sorted = sorted(edges2)
    
    if edges1_sorted != edges2_sorted:
        print(f"❌ MISMATCH: Edges differ")
        
        # Find differences
        set1 = set(edges1_sorted)
        set2 = set(edges2_sorted)
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        
        if only_in_1:
            print(f"  {len(only_in_1)} edges only in file 1")
            print(f"    First few: {list(only_in_1)[:5]}")
        if only_in_2:
            print(f"  {len(only_in_2)} edges only in file 2")
            print(f"    First few: {list(only_in_2)[:5]}")
        
        return False
    
    print(f"✓ Edges match (all {m1} edges)")
    
    print(f"\n✅ Files are identical!")
    return True

def print_instance_stats(filepath):
    """
    Print statistics about an Ising instance file.
    """
    import os
    
    print(f"File: {filepath}")
    print(f"Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    n, m, h_values, edges = read_ising_auto(filepath)
    
    print(f"\nInstance statistics:")
    print(f"  Nodes: {n:,}")
    print(f"  Edges: {m:,}")
    print(f"  Avg degree: {2 * m / n:.2f}")
    
    h_array = np.array(h_values)
    print(f"\nLinear biases (h):")
    print(f"  Min: {h_array.min()}")
    print(f"  Max: {h_array.max()}")
    print(f"  Mean: {h_array.mean():.2f}")
    print(f"  First 10: {h_array[:10].tolist()}")
    
    # Analyze edges
    edge_weights = [w for _, _, w in edges]
    weight_array = np.array(edge_weights)
    
    print(f"\nEdge weights (J):")
    print(f"  Min: {weight_array.min()}")
    print(f"  Max: {weight_array.max()}")
    print(f"  Mean: {weight_array.mean():.2f}")
    
    print(f"\nFirst 10 edges:")
    for i, j, w in edges[:10]:
        print(f"  ({i}, {j}): {w}")

# Usage examples
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python read_binary_ising.py <filepath>                    # Read and display stats")
        print("  python read_binary_ising.py <file1> <file2>              # Compare two files")
        print("\nSupported formats:")
        print("  - .bin      : Uncompressed binary")
        print("  - .bin.gz   : Compressed CSR binary")
        print("  - .txt      : Text format")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single file: print stats
        filepath = sys.argv[1]
        print_instance_stats(filepath)
    
    elif len(sys.argv) == 3:
        # Two files: compare
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        verify_ising_instance(file1, file2)