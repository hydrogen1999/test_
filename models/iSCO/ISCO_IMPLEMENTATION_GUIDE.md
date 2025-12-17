# iSCO Implementation Guide using DISCS

## Overview

**iSCO** (improved Sampling for Combinatorial Optimization) is a modern sampling-based approach for solving combinatorial optimization problems that leverages:
1. **Energy-Based Models (EBM)** for flexible distribution representation
2. **Simulated Annealing (SA)** with a sequence of decaying temperatures
3. **Path Auxiliary Sampler (PAS)** for efficient discrete space exploration
4. **Gradient-based approximations** for fast neighborhood ratio estimation
5. **JAX-based parallelization** for efficient computation on accelerators

This guide explains how to use the DISCS framework to implement and run iSCO experiments as described in the paper "Revisiting Sampling for Combinatorial Optimization" (Sun et al., 2023).

---

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [iSCO Components in DISCS](#isco-components-in-discs)
3. [Installation and Setup](#installation-and-setup)
4. [Running iSCO Experiments](#running-isco-experiments)
5. [Configuration Details](#configuration-details)
6. [Problem-Specific Examples](#problem-specific-examples)
7. [Advanced Usage](#advanced-usage)
8. [Understanding the Results](#understanding-the-results)

---

## Key Concepts

### 1. Problem Formulation

iSCO converts a combinatorial optimization problem into a sampling problem:

**Original problem:**
```
min_{x ∈ S} a(x)
s.t. b(x) = 0
```

**Energy function (penalty form):**
```
f(x) = a(x) + λ·b(x)
```

**Target distribution:**
```
p_τ(x) ∝ exp(-f(x)/τ)
```

### 2. Temperature Schedule

iSCO uses a sequence of decaying temperatures:
```
P = [p_τ0(x), p_τ1(x), ..., p_τT(x)]
```
where `τ0 > τ1 > ... > τT → 0`

This creates a progression from exploration (high τ) to exploitation (low τ).

### 3. Efficient Neighborhood Exploration

The key innovation is computing energy differences efficiently:
```
Δ(x) = [f(y₁) - f(x), f(y₂) - f(x), ..., f(yₙ) - f(x)]
```

For many CO problems, this can be approximated using gradients:
```
p(y)/p(x) ≈ exp(⟨-∇f(x), y-x⟩)
```

### 4. Path Auxiliary Sampler (PAS)

PAS is the core sampler that:
- Proposes multiple dimensional changes in parallel
- Uses locally balanced functions for better mixing
- Applies Metropolis-Hastings correction

---

## iSCO Components in DISCS

The DISCS framework already contains all the necessary components to run iSCO:

### Core Components

| iSCO Component | DISCS Implementation | Location |
|----------------|---------------------|----------|
| Path Auxiliary Sampler | `path_auxiliary.py` | `discs/samplers/` |
| Temperature Schedule | `t_schedule` in experiment config | `discs/experiment/` |
| CO Models | `maxcut.py`, `mis.py`, etc. | `discs/models/` |
| Gradient Approximation | `approx_with_grad=True` | Sampler config |
| Parallel Execution | JAX pmap/vmap | `discs/experiment/sampling.py` |

### Key Files

1. **Sampler**: `discs/discs/samplers/path_auxiliary.py`
   - Implements the PAS algorithm
   - Supports both binary and categorical variables
   - Includes gradient approximation

2. **Models**: `discs/discs/models/`
   - `maxcut.py`: Maximum Cut problem
   - `mis.py`: Maximum Independent Set
   - `normcut.py`: Normalized Cut (graph partitioning)
   - `maxclique.py`: Maximum Clique

3. **Experiment Runner**: `discs/discs/experiment/sampling.py`
   - Manages temperature schedules
   - Handles chain generation
   - Coordinates sampler and model

4. **Configurations**: `discs/discs/experiment/configs/`
   - Problem-specific temperature schedules
   - Batch sizes and chain lengths

---

## Installation and Setup

### 1. Environment Setup

According to the project configuration, activate the conda environment:

```bash
conda activate apachejit
```

### 2. Install DISCS

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/
pip install -e .
```

### 3. Download Data

Download the combinatorial optimization datasets from:
[DISCS-DATA](https://drive.google.com/drive/u/1/folders/1nEppxuUJj8bsV9Prc946LN_buo30AnDx)

Place the data in an appropriate location and update the `data_root` path in the model configs.

---

## Running iSCO Experiments

### Basic Usage

The iSCO algorithm in DISCS is executed by combining:
- **Sampler**: `path_auxiliary` (the PAS implementation)
- **Temperature schedule**: Exponential or linear decay
- **CO model**: Your target problem (maxcut, mis, etc.)

### Example 1: Maximum Cut (Basic)

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/
model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

### Example 2: Maximum Independent Set

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/
model=mis graph_type=er_density sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

### Example 3: Normalized Cut (Graph Partitioning)

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/
model=normcut graph_type=nets sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

### Example 4: Maximum Clique

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/
model=maxclique graph_type=rb sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

---

## Configuration Details

### iSCO-Specific Configuration

To properly configure iSCO experiments, you need to set parameters in three places:

#### 1. Sampler Configuration

File: `discs/discs/samplers/configs/path_auxiliary_config.py`

```python
def get_config():
  sampler_config = dict(
      name='path_auxiliary',
      use_fast_path=True,           # Use fast PAS implementation
      num_flips=1,                  # Initial number of flips
      adaptive=True,                # Adaptively adjust number of flips
      target_acceptance_rate=0.574, # Target acceptance rate
      balancing_fn_type='SQRT',     # Locally balanced function: 'SQRT', 'RATIO', 'MAX', 'MIN'
      approx_with_grad=True,        # Use gradient approximation (key for iSCO!)
  )
  return config_dict.ConfigDict(sampler_config)
```

**Key Parameter**: `approx_with_grad=True` enables the gradient-based neighborhood ratio estimation that makes iSCO efficient.

#### 2. Experiment Configuration

File: `discs/discs/experiment/configs/<problem>/<graph_type>.py`

```python
def get_config():
  exp_config = dict(
      experiment=dict(
          batch_size=16,                  # Number of parallel chains
          t_schedule='exp_decay',         # Temperature schedule: 'exp_decay' or 'linear'
          chain_length=10000,             # Total MCMC steps
          log_every_steps=100,            # Logging frequency
          init_temperature=1.0,           # Initial temperature τ₀
          decay_rate=0.1,                 # Temperature decay rate
          final_temperature=0.001,        # Final temperature τ_T
          save_root='./discs/results',    # Output directory
      )
  )
  return config_dict.ConfigDict(exp_config)
```

**Temperature Schedules:**

- **Exponential decay**: `τ_t = τ_0 * exp(-decay_rate * t / chain_length)`
- **Linear decay**: `τ_t = τ_0 - (τ_0 - τ_final) * t / chain_length`

#### 3. Model Configuration

File: `discs/discs/models/configs/<model>_config.py`

For CO problems, set the `data_root` path:

```python
def get_config():
  model_config = dict(
      name='maxcut',
      data_root='/path/to/DISCS-DATA/sco/',  # Update this path!
      penalty_lambda=1.0,                     # Penalty coefficient λ
  )
  return config_dict.ConfigDict(model_config)
```

---

## Problem-Specific Examples

### 1. Maximum Cut Problem

**Problem**: Partition graph nodes to maximize edges between partitions.

**Energy function**: 
```
f(x) = -∑_{(i,j)∈E} w_{ij} · (x_i ⊕ x_j)
```

**Run command**:
```bash
# Barabási-Albert graphs
model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# Erdős-Rényi graphs  
model=maxcut graph_type=er sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# OPTSICOM benchmark
model=maxcut graph_type=optsicom sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

**Configuration** (`discs/discs/experiment/configs/maxcut/ba.py`):
```python
exp_config = dict(
    experiment=dict(
        batch_size=16,
        t_schedule='exp_decay',
        chain_length=10000,
        init_temperature=1.0,
        decay_rate=0.1,
        final_temperature=0.5,
    )
)
```

### 2. Maximum Independent Set (MIS)

**Problem**: Select maximum number of nodes with no adjacent nodes selected.

**Energy function**:
```
f(x) = -∑_i c_i·x_i + λ·∑_{(i,j)∈E} x_i·x_j
```

**Run command**:
```bash
# ER graphs with varying density
model=mis graph_type=er_density sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# SATLIB benchmark
model=mis graph_type=satlib sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

**Configuration** (`discs/discs/experiment/configs/mis/er_density.py`):
```python
exp_config = dict(
    experiment=dict(
        batch_size=16,
        t_schedule='exp_decay',
        chain_length=5000,
        init_temperature=1.0,
        decay_rate=0.1,
        final_temperature=0.001,
    )
)
```

### 3. Normalized Cut (Graph Partitioning)

**Problem**: Partition graph into K clusters minimizing normalized cut.

**Energy function**:
```
f(x) = ∑_k cut(S_k, S̄_k) / vol(S_k, V)
```

**Run command**:
```bash
# Neural network compression
model=normcut graph_type=nets sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

**Note**: This problem uses categorical variables (not binary), so PAS automatically adapts.

**Configuration** (`discs/discs/experiment/configs/normcut/nets.py`):
```python
exp_config = dict(
    experiment=dict(
        batch_size=8,
        t_schedule='exp_decay',
        chain_length=20000,
        init_temperature=1.0,
        decay_rate=0.05,
        final_temperature=0.001,
    )
)
```

### 4. Maximum Clique

**Problem**: Find largest complete subgraph.

**Run command**:
```bash
# Random graphs (RB model)
model=maxclique graph_type=rb sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# Twitter social network
model=maxclique graph_type=twitter sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

---

## Advanced Usage

### Custom Configuration Sweep

Create a custom experiment configuration file:

**File**: `discs/run_configs/co/custom_isco_sweep.py`

```python
from ml_collections import config_dict

def get_config():
  """iSCO configuration sweep."""
  config = config_dict.ConfigDict(
      dict(
          model='maxcut',
          graph_type='ba',
          sampler='path_auxiliary',
          sweep=[
              {
                  # Temperature schedule sweep
                  'config.experiment.init_temperature': [0.5, 1.0, 2.0],
                  'config.experiment.decay_rate': [0.05, 0.1, 0.2],
                  'config.experiment.final_temperature': [0.001, 0.01],
                  
                  # Batch size sweep
                  'config.experiment.batch_size': [8, 16, 32],
                  
                  # Chain length sweep
                  'config.experiment.chain_length': [5000, 10000, 20000],
                  
                  # Sampler parameters
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO', 'MAX'],
                  'sampler_config.target_acceptance_rate': [0.4, 0.574, 0.7],
              },
          ],
      )
  )
  return config
```

**Run on Xmanager**:
```bash
config=discs/run_configs/co/custom_isco_sweep.py ./discs/run_xmanager.sh
```

### Modifying Temperature Schedules

Edit `discs/discs/experiment/sampling.py` to add custom schedules:

```python
def get_temperature_schedule(self, step):
  """Get temperature at given step."""
  if self.config.t_schedule == 'exp_decay':
    # Exponential decay: τ_t = τ_0 * exp(-r * t)
    temp = self.config.init_temperature * jnp.exp(
        -self.config.decay_rate * step / self.config.chain_length
    )
  elif self.config.t_schedule == 'linear':
    # Linear decay: τ_t = τ_0 - (τ_0 - τ_f) * t / T
    temp = self.config.init_temperature - (
        self.config.init_temperature - self.config.final_temperature
    ) * step / self.config.chain_length
  elif self.config.t_schedule == 'cosine':
    # Cosine annealing (add this!)
    temp = self.config.final_temperature + 0.5 * (
        self.config.init_temperature - self.config.final_temperature
    ) * (1 + jnp.cos(jnp.pi * step / self.config.chain_length))
  
  return jnp.maximum(temp, self.config.final_temperature)
```

### Comparing with Other Samplers

Run the same experiment with different samplers to compare:

```bash
# iSCO (Path Auxiliary Sampler)
model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# Gibbs with Gradients (GWG)
model=maxcut graph_type=ba sampler=gwg ./discs/experiment/run_sampling_local.sh

# Discrete MALA (DMALA)
model=maxcut graph_type=ba sampler=dmala ./discs/experiment/run_sampling_local.sh

# Discrete Langevin Monte Carlo (DLMC)
model=maxcut graph_type=ba sampler=dlmc ./discs/experiment/run_sampling_local.sh

# Random Walk (baseline)
model=maxcut graph_type=ba sampler=randomwalk ./discs/experiment/run_sampling_local.sh
```

---

## Understanding the Results

### Output Structure

Results are saved in `discs/results/` with the following structure:

```
discs/results/
├── <model_name>/
│   ├── <graph_type>/
│   │   ├── <sampler_name>/
│   │   │   ├── results.pkl      # Main results file
│   │   │   ├── config.json      # Experiment configuration
│   │   │   └── logs.txt         # Execution logs
```

### Results Content

For CO problems, `results.pkl` contains:

```python
{
    'objective_values': np.array,  # Shape: (num_chains, chain_length)
    'best_solutions': np.array,     # Best samples found
    'acceptance_rates': np.array,   # Acceptance rates over time
    'temperatures': np.array,       # Temperature schedule used
    'runtime': float,               # Total execution time
    'config': dict,                 # Full configuration
}
```

### Visualizing Results

Use the plotting tools provided:

```bash
# Plot objective values over time
./discs/plot_results/run_plot_co_through_time.sh

# Plot performance comparison across samplers
./discs/plot_results/run_plot_results.sh
```

### Key Metrics

1. **Best Objective Found**: `min(objective_values)`
2. **Speed-Quality Trade-off**: Objective value vs. runtime
3. **Convergence Speed**: Steps to reach near-optimal solution
4. **Sample Quality**: Distribution of final samples

### Python Analysis Example

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('discs/results/maxcut/ba/path_auxiliary/results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract data
objectives = results['objective_values']  # (num_chains, chain_length)
temperatures = results['temperatures']
runtime = results['runtime']

# Plot convergence
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Best objective over time
best_over_time = np.maximum.accumulate(objectives.max(axis=0))
ax1.plot(best_over_time)
ax1.set_xlabel('MCMC Step')
ax1.set_ylabel('Best Objective')
ax1.set_title('iSCO Convergence')
ax1.grid(True)

# Temperature schedule
ax2.plot(temperatures)
ax2.set_xlabel('MCMC Step')
ax2.set_ylabel('Temperature')
ax2.set_title('Simulated Annealing Schedule')
ax2.grid(True)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('isco_results.png', dpi=300)
plt.show()

# Print summary statistics
print(f"Best objective found: {objectives.max():.4f}")
print(f"Mean final objective: {objectives[:, -1].mean():.4f}")
print(f"Std final objective: {objectives[:, -1].std():.4f}")
print(f"Total runtime: {runtime:.2f} seconds")
print(f"Steps per second: {objectives.shape[1] / runtime:.2f}")
```

---

## Key Differences from Standard DISCS

While DISCS provides the infrastructure, **iSCO specifically refers to**:

1. **Using Path Auxiliary Sampler** (`path_auxiliary`) - not just any sampler
2. **Gradient approximation enabled** (`approx_with_grad=True`)
3. **Simulated annealing** with carefully tuned temperature schedules
4. **Parallel execution** on multiple chains
5. **Problem-specific gradient calculations** (already implemented in CO models)

### Standard DISCS vs. iSCO

| Aspect | Standard DISCS | iSCO Configuration |
|--------|----------------|-------------------|
| Sampler | Any sampler | `path_auxiliary` |
| Gradient use | Optional | `approx_with_grad=True` |
| Temperature | Can be fixed | Annealing schedule required |
| Focus | General sampling | CO optimization |
| Goal | ESS, mixing time | Solution quality, speed |

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `batch_size` in experiment config
- Reduce `chain_length`
- Use smaller graphs

**2. Slow Execution**
- Ensure `run_parallel=True` in experiment config
- Check JAX GPU/TPU availability
- Verify data is not being reloaded each step

**3. Poor Solution Quality**
- Increase `chain_length`
- Tune temperature schedule (slower decay)
- Adjust `init_temperature` and `final_temperature`
- Try different `balancing_fn_type`

**4. Data Not Found**
- Update `data_root` in model config
- Verify data downloaded from Google Drive
- Check file permissions

### Debugging Tips

**Enable verbose logging**:
```python
# In experiment config
exp_config = dict(
    experiment=dict(
        use_tqdm=True,              # Show progress bar
        log_every_steps=10,         # Log more frequently
        save_samples=True,          # Save intermediate samples
    )
)
```

**Test on small instance**:
```bash
# Modify chain_length for quick test
model=maxcut graph_type=ba sampler=path_auxiliary \
  config.experiment.chain_length=100 \
  ./discs/experiment/run_sampling_local.sh
```

---

## References

1. **iSCO Paper**: Sun, H., Goshvadi, K., Nova, A., Schuurmans, D., & Dai, H. (2023). "Revisiting Sampling for Combinatorial Optimization." *ICML 2023*.

2. **DISCS Framework**: Sun, H., et al. (2022). "DISCS: A Benchmark for Discrete Sampling." *NeurIPS 2022*.

3. **Path Auxiliary Sampler**: Sun, H., Dai, H., & Schuurmans, D. (2021). "Path Auxiliary Sampler for Discrete Variables."

4. **Discrete Langevin Dynamics**: Sun, H., et al. (2022). "Score-Based Discrete Sampling."

---

## Summary

To implement **iSCO** using DISCS:

1. **Use Path Auxiliary Sampler**: `sampler=path_auxiliary`
2. **Enable gradient approximation**: `approx_with_grad=True`
3. **Configure temperature annealing**: Set `t_schedule`, `init_temperature`, `decay_rate`
4. **Choose your CO problem**: `maxcut`, `mis`, `normcut`, or `maxclique`
5. **Run experiments**: Use provided shell scripts or Xmanager
6. **Analyze results**: Use plotting tools and pickle files

The power of iSCO comes from combining all these components together - modern discrete MCMC (PAS), efficient gradient-based proposals, and simulated annealing.

---

## Quick Start Checklist

- [ ] Activate conda environment: `conda activate apachejit`
- [ ] Install DISCS: `pip install -e .`
- [ ] Download CO data from Google Drive
- [ ] Update `data_root` in model configs
- [ ] Run basic test: `model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh`
- [ ] Check results in `discs/results/`
- [ ] Tune temperature schedule for your problem
- [ ] Compare with other samplers
- [ ] Create custom configuration sweeps

---

## Using iSCO for Custom Ising Instances

### Problem Description

You may want to solve a custom **Ising model** instance defined by an arbitrary interaction matrix **J** without bias terms. The energy function is:

```
E(s) = -∑_{i<j} J_{ij} * s_i * s_j
```

where:
- `s_i ∈ {-1, +1}` are spin variables
- `J` is an `N×N` symmetric interaction matrix with `J_{ii} = 0` (no self-interaction)
- The goal is to find the **ground state** (configuration minimizing E)

This is a fundamental problem in physics, optimization, and machine learning (e.g., spin glasses, QUBO problems).

### Why Current DISCS Models Won't Work

The existing `ising.py` model in DISCS is designed for:
- **2D lattice structures** (grid topology)
- **Uniform nearest-neighbor interactions** (λ parameter)
- **External field** (bias terms)

It **cannot** handle an arbitrary interaction matrix J.

### Implementation Strategy

To use iSCO for your custom Ising instance, you need to:

1. **Create a new model class** that extends `AbstractModel`
2. **Load your J matrix** into the model parameters
3. **Implement efficient energy and gradient computations**
4. **Configure iSCO sampler** (Path Auxiliary with gradients)
5. **Run optimization** with temperature annealing

### Step-by-Step Implementation

#### Step 1: Create General Ising Model

Create: `discs/discs/models/general_ising.py`

<details>
<summary>Key components to implement:</summary>

```python
"""General Ising Model for arbitrary J matrix without bias."""

from discs.models import abstractmodel
import jax.numpy as jnp
import jax

class GeneralIsing(abstractmodel.AbstractModel):
  """Ising model with arbitrary interaction matrix J."""
  
  def __init__(self, config):
    self.shape = config.shape  # (N,) where N is number of spins
    self.J_matrix = self._load_J_matrix(config.J_matrix_path)
    # Make J symmetric and zero diagonal
    self.J_matrix = (self.J_matrix + self.J_matrix.T) / 2.0
    self.J_matrix = self.J_matrix.at[jnp.diag_indices(N)].set(0.0)
  
  def forward(self, params, x):
    """Compute energy: E = -0.5 * s^T @ J @ s
    where s = 2*x - 1 maps {0,1} to {-1,+1}
    """
    J = params['J']
    temp = params.get('temperature', 1.0)
    spins = 2 * x.astype(jnp.float32) - 1  # {0,1} → {-1,+1}
    energy = -0.5 * jnp.sum(spins @ J * spins, axis=1)
    return -energy / temp  # Minimize -E = maximize E
  
  def get_value_and_grad(self, params, x):
    """Gradient for iSCO's efficient approximation."""
    x = x.astype(jnp.float32)
    def fun(z):
      energy = self.forward(params, z)
      return jnp.sum(energy), energy
    (_, energy), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return energy, grad
  
  def logratio_in_neighborhood(self, params, x):
    """Exact energy differences for 1-bit flips.
    
    For Ising: flipping spin i changes energy by:
    ΔE_i = -2 * s_i * h_i where h_i = ∑_j J_{ij} * s_j
    """
    J = params['J']
    temp = params.get('temperature', 1.0)
    spins = 2 * x.astype(jnp.float32) - 1
    
    # Compute local fields
    local_fields = spins @ J  # (batch, N)
    
    # Energy change for flipping each spin
    delta_E = -2.0 * spins * local_fields
    delta_objective = -delta_E / temp  # For minimizing -E
    
    # Sign for binary {0,1} representation
    sign = 1 - 2 * x
    logratio = sign * delta_objective
    
    current_energy = -0.5 * jnp.sum(spins @ J * spins, axis=1) / temp
    return -current_energy, logratio, 1, self.get_neighbor_fn

def build_model(config):
  return GeneralIsing(config.model)
```

</details>

**Key implementation details:**

- **Binary encoding**: Use `{0, 1}` internally, convert to spins `{-1, +1}` via `s = 2*x - 1`
- **Energy computation**: Efficient via matrix multiplication `s^T @ J @ s`
- **Gradient computation**: JAX automatic differentiation for iSCO
- **Local field method**: Exact energy differences for 1-bit flips: `h_i = ∑_j J_{ij} * s_j`

#### Step 2: Create Model Configuration

Create: `discs/discs/models/configs/general_ising_config.py`

```python
"""Config for general Ising model."""
from ml_collections import config_dict

def get_config():
  model_config = dict(
      name='general_ising',
      shape=(100,),  # N spins - UPDATE THIS!
      num_categories=2,
      J_matrix_path='/path/to/your/J_matrix.npy',  # UPDATE THIS!
      save_dir_name='general_ising',
  )
  return config_dict.ConfigDict(model_config)
```

**Supported J matrix formats:**
- `.npy`: NumPy binary format
- `.npz`: Compressed NumPy (expects key 'J')
- `.txt`: Space/comma-separated text file

#### Step 3: Create Experiment Configuration

Create: `discs/discs/experiment/configs/general_ising_experiment.py`

```python
"""Experiment config for Ising optimization."""
from ml_collections import config_dict

def get_config():
  exp_config = dict(
      experiment=dict(
          evaluator='co_eval',
          co_opt_prob=True,
          
          # Parallel chains
          batch_size=32,
          
          # Temperature schedule (CRITICAL for iSCO!)
          t_schedule='exp_decay',
          init_temperature=2.0,      # High T for exploration
          final_temperature=0.001,   # Low T for exploitation
          decay_rate=0.1,
          
          # Chain length
          chain_length=10000,
          
          log_every_steps=100,
          save_root='./discs/results',
      )
  )
  return config_dict.ConfigDict(exp_config)
```

**Temperature tuning tips:**
- **Larger systems** (N > 100): Start with higher `init_temperature` (3.0-5.0)
- **Strong frustration**: Slower decay (`decay_rate=0.05`)
- **Quick test**: Shorter chain (`chain_length=1000`)

#### Step 4: Prepare Your J Matrix

Your interaction matrix must satisfy:

1. **Shape**: `J.shape = (N, N)` where N is the number of spins
2. **Symmetric**: `J[i,j] = J[j,i]` (enforced automatically)
3. **Zero diagonal**: `J[i,i] = 0` (enforced automatically)
4. **Format**: NumPy array saved as `.npy` file

**Example: Create and save J matrix**

```python
import numpy as np

# Your custom J matrix
N = 100
J = np.random.randn(N, N)  # Example: random interactions
J = (J + J.T) / 2.0        # Make symmetric
np.fill_diagonal(J, 0.0)   # Zero diagonal

# Save to file
np.save('my_ising_instance.npy', J)
```

**Common Ising problems:**

- **Sherrington-Kirkpatrick (SK) model**: All-to-all random interactions
  ```python
  J = np.random.randn(N, N) / np.sqrt(N)
  ```

- **Sparse graph**: Only some pairs interact
  ```python
  from scipy.sparse import random
  J = random(N, N, density=0.1, format='coo', random_state=42)
  J = (J + J.T).toarray() / 2.0
  ```

- **Frustrated lattice**: Antiferromagnetic triangles
  ```python
  # 2D lattice with antiferromagnetic interactions
  J = -1.0 * adjacency_matrix  # Negative for antiferromagnetic
  ```

#### Step 5: Run iSCO on Your Instance

**Method A: Command line (after registering model)**

```bash
cd /home/guerrerocajv/Desktop/iSCO/discs/

# Set paths in configs first, then:
model=general_ising sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

**Method B: Python script**

Create `run_my_ising.py`:

```python
import numpy as np
import jax
from ml_collections import config_dict
import importlib
from discs.experiment import sampling
from discs.common import configs

# 1. Load your J matrix
J = np.load('my_ising_instance.npy')
N = J.shape[0]

# 2. Build configuration
config = configs.get_config()

# Model configuration
config.model.name = 'general_ising'
config.model.shape = (N,)
config.model.num_categories = 2
config.model.J_matrix_path = 'my_ising_instance.npy'

# Sampler configuration (iSCO settings!)
config.sampler.name = 'path_auxiliary'
config.sampler.use_fast_path = True
config.sampler.num_flips = 1
config.sampler.adaptive = True
config.sampler.target_acceptance_rate = 0.574
config.sampler.balancing_fn_type = 'SQRT'
config.sampler.approx_with_grad = True  # KEY!

# Experiment configuration
config.experiment.evaluator = 'co_eval'
config.experiment.co_opt_prob = True
config.experiment.batch_size = 32
config.experiment.chain_length = 10000
config.experiment.t_schedule = 'exp_decay'
config.experiment.init_temperature = 2.0
config.experiment.final_temperature = 0.001
config.experiment.decay_rate = 0.1

# 3. Build components
model = importlib.import_module('discs.models.general_ising').build_model(config)
sampler = importlib.import_module('discs.samplers.path_auxiliary').build_sampler(config)
evaluator = importlib.import_module('discs.evaluators.co_eval').build_evaluator(config)

# 4. Run optimization
experiment = sampling.Experiment(config)
rng = jax.random.PRNGKey(42)
results = experiment.run(rng, model, sampler, evaluator)

# 5. Extract best solution
best_obj = results['objective_values'].max()
best_idx = results['objective_values'].argmax()
best_config = results['samples'][best_idx]

print(f"Best Ising energy found: {-best_obj:.6f}")
print(f"Ground state configuration: {best_config}")
```

#### Step 6: Analyze Results

After running, results are saved in `discs/results/general_ising/`:

**Load and analyze:**

```python
import pickle
import numpy as np

# Load results
with open('discs/results/general_ising/results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract data
objectives = results['objective_values']  # (batch_size, chain_length)
samples = results.get('samples', None)

# Best solution
best_energy = -objectives.max()  # Convert back to energy
best_idx = np.unravel_index(objectives.argmax(), objectives.shape)
best_solution = samples[best_idx] if samples is not None else None

print(f"Ground state energy (estimated): {best_energy:.6f}")

# Verify with J matrix
if best_solution is not None:
    J = np.load('my_ising_instance.npy')
    spins = 2 * best_solution.astype(float) - 1
    verified_energy = -0.5 * np.sum(spins @ J * spins)
    print(f"Verified energy: {verified_energy:.6f}")

# Convergence plot
import matplotlib.pyplot as plt
plt.plot(objectives.max(axis=0))
plt.xlabel('MCMC Step')
plt.ylabel('Best Objective')
plt.title('iSCO Convergence on Ising Instance')
plt.savefig('ising_convergence.png')
```

### Why This Works: iSCO Efficiency for Ising

iSCO is particularly efficient for Ising models because:

1. **Exact gradient computation**: The energy is differentiable w.r.t. continuous relaxation
   ```
   ∇f(x) = -J @ (2x - 1) / temp
   ```

2. **Exact local fields**: Computing all 1-bit flip energies is O(N²) but can be done in one matrix-vector product:
   ```
   h = J @ s  where s = 2x - 1
   ΔE_i = -2 * s_i * h_i
   ```

3. **Parallel proposals**: Path Auxiliary Sampler proposes multiple spin flips simultaneously

4. **Automatic differentiation**: JAX computes gradients without manual derivation

### Comparison: Standard Ising vs General Ising

| Aspect | Standard `ising.py` | Your `general_ising.py` |
|--------|-------------------|----------------------|
| Topology | 2D lattice only | Arbitrary graph |
| Interactions | Nearest-neighbor, uniform | Any J_{ij} matrix |
| Bias terms | Supported | Not needed (J only) |
| Use case | Sampling from distribution | Optimization (ground state) |
| Input | Parameters (λ, μ, σ) | J matrix file/array |

### Performance Tuning

**For small instances (N < 100):**
```python
batch_size = 16
chain_length = 5000
init_temperature = 1.0
decay_rate = 0.1
```

**For medium instances (100 ≤ N < 500):**
```python
batch_size = 32
chain_length = 20000
init_temperature = 2.0
decay_rate = 0.05
```

**For large instances (N ≥ 500):**
```python
batch_size = 64
chain_length = 50000
init_temperature = 5.0
decay_rate = 0.03
# Consider using run_parallel = True for multiple GPUs
```

**For highly frustrated systems (spin glasses):**
```python
init_temperature = 3.0  # Higher exploration
decay_rate = 0.03       # Slower annealing
chain_length = 50000    # Longer runs
# Try multiple independent runs with different seeds
```

### Troubleshooting

**Problem: Poor solutions (high energy)**
- ✓ Increase `init_temperature` (start hotter)
- ✓ Decrease `decay_rate` (anneal slower)
- ✓ Increase `chain_length` (run longer)
- ✓ Increase `batch_size` (more parallel exploration)

**Problem: Slow execution**
- ✓ Check `approx_with_grad=True` is set
- ✓ Enable `run_parallel=True` for GPU parallelization
- ✓ Reduce `batch_size` if memory limited

**Problem: Not converging**
- ✓ Your instance may be in a frustrated regime (no clear ground state)
- ✓ Try different temperature schedules ('linear' vs 'exp_decay')
- ✓ Run multiple times with different random seeds
- ✓ Compare with exact solver on small instances to validate

### Validation on Small Instances

For small N (≤ 20), validate against exact enumeration:

```python
import numpy as np
from itertools import product

def exact_ground_state(J):
  """Brute force search (only for small N!)."""
  N = J.shape[0]
  best_energy = float('inf')
  best_config = None
  
  # Try all 2^N configurations
  for config in product([0, 1], repeat=N):
    spins = 2 * np.array(config) - 1
    energy = -0.5 * np.sum(spins[None, :] @ J * spins[None, :])
    if energy < best_energy:
      best_energy = energy
      best_config = config
  
  return best_energy, best_config

# Compare
exact_E, exact_config = exact_ground_state(J)
isco_E = -results['objective_values'].max()

print(f"Exact ground state: {exact_E:.6f}")
print(f"iSCO found: {isco_E:.6f}")
print(f"Gap: {isco_E - exact_E:.6f}")
```

### Summary Checklist

To use iSCO for your Ising instance with matrix J:

- [ ] **Create model**: Implement `general_ising.py` extending `AbstractModel`
- [ ] **Implement forward()**: Compute energy E = -0.5 * s^T @ J @ s
- [ ] **Implement get_value_and_grad()**: JAX auto-differentiation
- [ ] **Implement logratio_in_neighborhood()**: Exact local field computation
- [ ] **Prepare J matrix**: Save as `.npy` file, ensure symmetric with zero diagonal
- [ ] **Configure sampler**: Use `path_auxiliary` with `approx_with_grad=True`
- [ ] **Set temperature schedule**: Start high, decay to low
- [ ] **Run optimization**: Execute via script or command line
- [ ] **Analyze results**: Extract best configuration and verify energy

The key insight is that iSCO's gradient-based proposal mechanism is **perfect for Ising models** because:
- Energy is a quadratic form in spins
- Gradients are computed exactly and efficiently
- Local fields give exact flip energies
- JAX handles parallelization automatically

---

*For questions or issues, refer to the DISCS README and the original iSCO paper.*

