# QQA Solver Setup Guide

This guide provides simple instructions to install and run the QQA (Quasi-Quantum Annealing) solver in the Spectral Annealing pipeline.

## Overview

The QQA solver uses the QQA4CO library (ICLR 2025) to solve Ising problems. The QQA4CO repository is already included in `models/QQA4CO/`, so no additional installation is required.

## Prerequisites

### Environment Setup

The solver has been tested with the `qqa-gpu` conda environment. The following packages are installed in this environment:

- `torch` - version 2.6.0+cu124 (PyTorch with CUDA 12.4 support)
- `numpy` - version 2.1.2
- `pandas` - version 2.3.3
- `scipy` - version 1.15.3
- `matplotlib` - version 3.10.7
- `networkx` - version 3.3
- `tqdm` - version 4.67.1

### Install Dependencies

To set up the same environment:

```bash
# Create and activate environment
conda create -n qqa-gpu python=3.10
conda activate qqa-gpu

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install numpy==2.1.2 pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.7 networkx==3.3 tqdm==4.67.1
```

**Note**: If you encounter CUDA library errors when importing PyTorch, you can force CPU-only mode by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

## Running the QQA Solver

### Basic Usage

```bash
conda activate qqa-gpu
python main.py --solver QQA --dataset barabasi_albert_m20 --instance ba_128_1_0.txt
```

### Configuration

The QQA solver is configured in `configs/default.yml`:

```yaml
QQA:
  solver_name: "qqa"
  params:
    sol_size: 100
    learning_rate: 1.0
    temp: 0.001
    min_bg: -3.0
    max_bg: 0.1
    curve_rate: 4.0
    div_param: 0.2
    num_epochs: 3000
    check_interval: 500
    plot_dynamics: false
    device: null  # null means auto-detect (cuda if available, else cpu)
```

### Parameters

- `sol_size`: Number of parallel solutions in the batch (default: 100)
- `learning_rate`: Learning rate for gradient updates (default: 1.0)
- `temp`: Temperature for noise injection (default: 0.001)
- `min_bg`, `max_bg`: Background field range for annealing (default: -3.0 to 0.1)
- `curve_rate`: Penalty curve rate (default: 4.0)
- `div_param`: Diversity parameter (default: 0.2)
- `num_epochs`: Number of annealing iterations (default: 3000)
- `check_interval`: Logging frequency (default: 500)
- `plot_dynamics`: Whether to plot energy curves (default: false)
- `device`: Device to use ('cpu', 'cuda', or null for auto-detect)

