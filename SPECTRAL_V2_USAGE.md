# Spectral Annealing V2 - Usage Guide

## New Solver Available

The new modular Spectral Annealing V2 solver has been integrated and is available through the selector.

## Available Solvers

- `spectral_annealing` - Original solver (backward compatible, uses legacy implementation)
- `spectral_annealing_legacy` - Explicit legacy implementation
- `spectral_annealing_v2` - **NEW** Enhanced version with modular components

## New Features in V2

1. **Eigenvector Predictors**: Choose from 4 prediction strategies
   - `diagonal` - Diagonal scaling predictor
   - `secant` - Secant extrapolation
   - `hybrid` - Combination of diagonal and secant
   - `subspace` - Rayleigh-Ritz in 2D subspace

2. **Spectral-Radius-Based Shift**: Uses ρₖ = (1+ε) max{λ₁, Gershgorin} instead of beta

3. **Residual Check**: Validates predictor quality and falls back if needed

4. **Lanczos SPD**: Uses positive definite operator S_ρ = ρI - N^α

## Usage Examples

### Command Line

```bash
python main.py --solver spectral_annealing_v2 --dataset barabasi_albert_m20 --instance G1.txt --pred_mode hybrid --pred_tol 1e-3
```

### Configuration File (configs/default.yml)

Add to your config file:

```yaml
SpecAnn-V2-Diagonal:
  solver_name: "spectral_annealing_v2"
  params:
    mode: 0
    use_local_search: true
    eigsh_tol: 1e-6
    eigsh_maxiter: 1000
    precision: 32
    pred_mode: "diagonal"
    pred_tol: 1e-3

SpecAnn-V2-Hybrid:
  solver_name: "spectral_annealing_v2"
  params:
    mode: 0
    use_local_search: true
    eigsh_tol: 1e-6
    eigsh_maxiter: 1000
    precision: 32
    pred_mode: "hybrid"
    pred_tol: 1e-3

SpecAnn-V2-Subspace:
  solver_name: "spectral_annealing_v2"
  params:
    mode: 0
    use_local_search: true
    eigsh_tol: 1e-6
    eigsh_maxiter: 1000
    precision: 32
    pred_mode: "subspace"
    pred_tol: 1e-3
```

### Python API

```python
from models.spectral_annealing.v2 import SpectralAnnealingV2
import numpy as np

# Create solver instance
solver = SpectralAnnealingV2(
    instance_name="G1.txt",
    dataset="barabasi_albert_m20",
    random_seed=42,
    pred_mode="hybrid",  # New parameter
    pred_tol=1e-3,       # New parameter
    use_local_search=True,
    precision=32
)

# Load your matrix J
J = np.load("your_matrix.npy")  # or load from file

# Solve
solver.solve(J)
```

## Parameters

### New Parameters (V2 only)

- `pred_mode` (str): Predictor mode
  - `"diagonal"` - Fast, good for smooth alpha transitions
  - `"secant"` - Good for linear extrapolation
  - `"hybrid"` - Balanced approach (recommended)
  - `"subspace"` - Most accurate but slower (requires matrix-vector products)

- `pred_tol` (float): Residual tolerance for predictor validation
  - Default: `1e-3`
  - If predictor residual > pred_tol, falls back to diagonal predictor

### Existing Parameters (same as legacy)

- `mode`, `beta`, `alpha`, `use_local_search`, `max_local_iterations`
- `eigsh_tol`, `eigsh_maxiter`, `disable_warm_start`, `precision`
- All other parameters work the same as the original solver

## What's Preserved

✅ All GPU kernels (unchanged)  
✅ Local search functions (unchanged)  
✅ Energy computation (unchanged)  
✅ I/O and CSV storage (unchanged)  
✅ All existing functionality

## What's New

🆕 Modular predictor system  
🆕 Spectral-radius-based shift strategy  
🆕 Residual validation  
🆕 SPD operator construction  
🆕 Better numerical stability

## Backward Compatibility

The original `spectral_annealing` solver name still works and uses the legacy implementation. No breaking changes!

