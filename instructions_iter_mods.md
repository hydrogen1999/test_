# Instructions: Iteration Count Measurement Implementation

## Overview

This document describes the modifications made to implement rigorous measurement of average iteration counts with and without warm start in the Spectral Annealing implementation. These changes enable empirical evidence that warm starts drastically reduce Lanczos iterations.

## Part 1: What Was Measured

We measure:
1. **Average inner iteration count per continuation step**: The number of Lanczos iterations taken by `eigsh()` at each continuation step α
2. **Comparison metrics**: Average iterations with warm start vs. without warm start, showing speedup factors

## Part 2: Key Technical Problem Solved

**Problem**: CuPy's `eigsh()` does not expose iteration count by default.

**Solution**: Modified the `eigsh()` function in CuPy's library to return the iteration count as a third return value. The iteration count is tracked in the `iter` variable inside the thick-restart Lanczos loop.

## Part 3: Modifications Made

### 3.1 Modification to CuPy Library (`_eigen.py`)

**File**: `/home/guerrerocajv/anaconda3/envs/spec-gpu/lib/python3.11/site-packages/cupyx/scipy/sparse/linalg/_eigen.py`

**Changes**:
- Modified the `eigsh()` function to return iteration count as an additional return value
- Line 141: Changed `return w[idx], x[:, idx]` to `return w[idx], x[:, idx], iter`
- Line 143: Changed `return cupy.sort(w)` to `return cupy.sort(w), iter`

**Why**: The `iter` variable inside `eigsh()` tracks the exact number of Lanczos update steps, which is what we need to measure. This variable is incremented in the thick-restart loop (line 131: `iter += ncv - k`) and initialized after the first Lanczos iteration (line 100: `iter = ncv`).

**Impact**: This change affects ALL files that use `cupyx.scipy.sparse.linalg.eigsh()`, including:
- `spectral_annealing.py` (main file, updated)
- `spectral_annealing_trevisan.py` (needs update if used)
- `spectral_annealing_no_fast_eig.py` (needs update if used)
- Other solver files (may need updates if used)

### 3.2 Modification to Spectral Annealing Class

**File**: `models/spectral_annealing.py`

#### 3.2.1 Added `disable_warm_start` Parameter

**Location**: `__init__` method (line 49)

**Change**: Added `disable_warm_start: bool = False` parameter

**Why**: This flag allows running the same continuation schedule without warm start for baseline comparison. When `True`, `v0=None` is passed to `eigsh()` at every step, simulating cold start behavior.

#### 3.2.2 Store Random Seed

**Location**: `__init__` method (line 66)

**Change**: Store `self.random_seed = random_seed` explicitly

**Why**: Ensures the same random seed is used for both warm and cold start runs, making the comparison fair.

#### 3.2.3 Reset Random Seed in `solve()` Method

**Location**: `solve()` method (lines 109-111)

**Change**: Added explicit random seed reset at the start of the alpha loop:
```python
if self.random_seed is not None:
    cp.random.seed(self.random_seed)
```

**Why**: Ensures reproducibility and fair comparison between warm and cold start runs.

#### 3.2.4 Track Alpha Index

**Location**: `solve()` method (line 121)

**Change**: Changed `for alpha in alpha_list:` to `for alpha_idx, alpha in enumerate(alpha_list):`

**Why**: Allows identifying the first alpha step (index 0) which should be excluded from averaging since both warm and cold start behave identically at α=0.

#### 3.2.5 Implement Warm Start Control

**Location**: `solve()` method (lines 130-137)

**Changes**:
- Added logic to disable warm start when flag is set: `v0_input = None if self.disable_warm_start else prev_v0_y`
- Only update `prev_v0_y` if warm start is enabled

**Why**: Enables controlled comparison between warm and cold start behavior.

#### 3.2.6 Capture and Store Iteration Count

**Location**: `solve()` method (line 133)

**Change**: Modified to capture iteration count: `evals, evecs, evecs_y, iter_count = self._solve_eigenvalue_problem(...)`

**Location**: `solve()` method (lines 157-173)

**Change**: Added to `alpha_results` dictionary:
- `'eigsh_iterations': int(iter_count)` - The iteration count for this alpha step
- `'warm_start': not self.disable_warm_start` - Boolean flag indicating if warm start was used
- `'alpha_index': alpha_idx` - Index of alpha step (0-based)

**Why**: Stores iteration data for each continuation step, enabling post-processing to compute averages and speedups.

#### 3.2.7 Modified `_solve_eigenvalue_problem()` to Return Iteration Count

**Location**: `_solve_eigenvalue_problem()` method (lines 269, 276)

**Changes**:
- Line 269: Changed `evals, evecs_y = cpx_splinalg.eigsh(...)` to `evals, evecs_y, iter_count = cpx_splinalg.eigsh(...)`
- Line 276: Changed return statement to include `iter_count`: `return evals[:k], X[:, :k], evecs_y[:, :k], iter_count`

**Why**: Propagates the iteration count from the library function to the calling code.

