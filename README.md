# Spectral Annealing Pipeline - Experimental Setup Guide

This document provides a comprehensive guide for running experiments in the Spectral Annealing pipeline, including setup instructions, experiment descriptions, and future work.

## Table of Contents

1. [Iteration Count Measurement Experiment](#1-iteration-count-measurement-experiment)
2. [Numerical Precision Experiment](#2-numerical-precision-experiment)
3. [SDP and SPONGE Solvers Setup](#3-sdp-and-sponge-solvers-setup)
4. [Future Experiments](#4-future-experiments)

---

## 1. Iteration Count Measurement Experiment

### Overview

This experiment measures the average iteration count of Lanczos iterations with and without warm start in the Spectral Annealing solver. The goal is to provide empirical evidence that warm starts drastically reduce the number of iterations needed for convergence.

### What Was Changed

Detailed changes are documented in `instructions_iter_mods.md`. Key modifications include:

1. **CuPy Library Modification** (`_eigen.py`):
   - Modified `eigsh()` to return iteration count as a third return value
   - Changes: Lines 141 and 143 now return `iter` count

2. **Spectral Annealing Class** (`models/spectral_annealing.py`):
   - Added `disable_warm_start` parameter to control warm start behavior
   - Added iteration count tracking and storage in CSV results
   - Added `alpha_index` field to exclude first alpha step from averaging

### What Needs to Be Changed

**Important**: The modification to `_eigen.py` affects ALL files that use `cupyx.scipy.sparse.linalg.eigsh()`. If you use other solver files, they need to be updated:

- `spectral_annealing_trevisan.py`
- `spectral_annealing_no_fast_eig.py`
- `spectral_annealing_no_fast_LS_no_fast_eig.py`
- `spectral_annealing_no_fast_LS.py`
- `spec_signed_lap.py`
- `spec_adj.py`

**Update pattern**: Change from:
```python
evals, evecs_y = cpx_splinalg.eigsh(...)
```

To:
```python
evals, evecs_y, iter_count = cpx_splinalg.eigsh(...)
```

(You can ignore `iter_count` if not needed in those files)

### How to Run the Experiment

#### Step 1: Run with Warm Start (Default)

```bash
conda activate spec-gpu
python main.py --solver SpecAnn --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

#### Step 2: Run with Cold Start (Baseline)

You need to modify the config file or add the parameter. For now, you can create a custom config or modify the code temporarily to set `disable_warm_start=True`.

**Option A: Modify config file** (`configs/default.yml`):
```yaml
SpecAnn-Cold:
  solver_name: "spectral_annealing"
  params:
    mode: 0
    beta: null
    use_local_search: true
    eigsh_tol: 1e-6
    eigsh_maxiter: 1000
    disable_warm_start: true
```

Then run:
```bash
python main.py --solver SpecAnn-Cold --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

---

## 2. Numerical Precision Experiment

### Overview

This experiment compares the performance and solution quality of Spectral Annealing using 32-bit (float32) vs 64-bit (float64) numerical precision. This is important for understanding the trade-off between memory usage, computational speed, and numerical accuracy.

### Implementation Details

The `spectral_annealing.py` solver now supports a `precision` parameter:
- `precision=32`: Uses `cp.float32` (32-bit floating point, default)
- `precision=64`: Uses `cp.float64` (64-bit floating point)

The precision is applied to:
- All matrix operations (J_csr, D_vec, D_inv_sqrt)
- Eigenvalue solver (eigsh)
- Local search operations (flip_refinement, gpu_final_sweep)
- CUDA kernels (separate kernels for float32 and float64)

The precision value is also stored in the CSV results for tracking.

### How to Run the Experiment

#### Step 1: Run with 32-bit Precision

```bash
conda activate spec-gpu
python main.py --solver SpecAnn --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

The config file (`configs/default.yml`) has `SpecAnn` configured with `precision: 32`.

#### Step 2: Run with 64-bit Precision

```bash
python main.py --solver SpecAnn-64 --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

The config file has `SpecAnn-64` configured with `precision: 64`.

---

## 3. SDP and SPONGE Solvers Setup

### SDP Solver (MOSEK)

#### Prerequisites

1. **Install Python packages**:
```bash
conda activate environment-name
pip install mosek cvxpy
```

2. **Obtain MOSEK License**:
   - Visit: https://www.mosek.com/products/academic-licenses/
   - Register with your academic email
   - Download the license file `mosek.lic`

3. **Configure License**:
   - Place the license file at: `~/mosek/mosek.lic`
   - Full path: `/home/YOUR_USERNAME/mosek/mosek.lic`

   If you have the license file in the project:
   ```bash
   mkdir -p ~/mosek
   cp mosek.lic ~/mosek/mosek.lic
   ```

4. **Verify Installation**:
```bash
conda activate spec-gpu
python -c "import cvxpy as cp; x = cp.Variable(); prob = cp.Problem(cp.Minimize(x**2), [x >= 1]); prob.solve(solver=cp.MOSEK); print(f'Status: {prob.status}')"
```

#### Running SDP Solver

```bash
conda activate spec-gpu
python main.py --solver SDP --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

**Note**: SDP solver can be slow for large instances as it converts the sparse matrix to dense for CVXPY.

### SPONGE Solver

#### Prerequisites

1. **Install SigNet**:
```bash
conda activate enviroment-name
pip install git+https://github.com/alan-turing-institute/SigNet.git
```

2. **Verify Installation**:
```bash
python -c "from signet.cluster import Cluster; print('SigNet installed successfully')"
```

#### Running SPONGE Solver

```bash
conda activate spec-gpu
python main.py --solver SPONGE --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

**Configuration** (in `configs/default.yml`):
```yaml
SPONGE:
  solver_name: "sponge"
  params:
    tau_p: 1.0
    tau_n: 1.0
    use_sym: true
```

---

## 4. Future Experiments

### 4.1 QQA and ISCo Solvers

#### Objective

Implement and integrate QQA (Quantum-inspired Quadratic Approximation) and ISCo (Iterative Spectral Clustering) solvers into the pipeline for comparison with existing methods.

---

### 4.2 Local Search Analysis in Spectral Annealing

#### Objective

Conduct a comprehensive analysis of the impact of local search on Spectral Annealing solution quality and performance. This includes understanding when local search helps, its computational cost, and optimal parameter settings.

