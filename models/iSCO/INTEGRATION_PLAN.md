# iSCO Integration Plan

This document describes how to stand up an **iSCO** solver inside the Spectral Annealing pipeline by reusing the vendored [DISCS](./discs) toolkit as much as possible. It summarizes the assets already available under `models/iSCO/`, clarifies the minimal dependency set we actually need, and lays out the implementation blueprint for the new solver (`ISCOSolver`) so it can plug into the existing `main.py → utils.core → solver_selector` execution path.

---

## 1. Assets You Already Have

- **Paper + reference**: `models/iSCO/sun23c.pdf` (Sun et al., 2023) and `models/iSCO/ISCO_IMPLEMENTATION_GUIDE.md` capture the algorithmic pieces (energy function reformulation, temperature schedule, gradient-augmented PAS sampler, JAX parallelism). The guide maps each conceptual block to a ready-made DISCS component you can call directly.
- **DISCS source tree** (`models/iSCO/discs/`) already checked in. For iSCO we mainly need:
  - `discs/samplers/path_auxiliary.py` (PAS implementation with gradient shortcut).
  - `discs/samplers/configs/path_auxiliary_config.py` (default knobs such as `num_flips`, `adaptive`, `target_acceptance_rate`).
  - `discs/experiment/sampling.py` (`CO_Experiment` class wires the sampler, temperature schedule, evaluator, and saver).
  - `discs/models/maxcut.py` + `discs/models/comb_ebm.py` (objective + gradient helpers for binary combinatorial problems).
  - `discs/common/experiment_saver.py` (writes tracking data under `config.experiment.save_root`).

---

## 2. Minimal Dependency Set

Activate the project’s Conda env (`conda activate apachejit`) and install DISCS in editable mode once:

```bash
cd /home/guerrerocajv/Desktop/SpecAnn/SpectralAnnealing_clean/models/iSCO/discs
pip install -e .
```

For iSCO on our CO workloads we only need the packages below (they are all listed in `discs/discs/requirements.txt`, but everything else—TensorFlow, HuggingFace, etc.—is for EBMs/text tasks and can be skipped to keep the environment light):

| Package | Version | Why it is needed |
|---------|---------|------------------|
| `jax[cuda]` | `0.3.25` (with the matching CUDA wheels) | Core array engine + PRNGs used everywhere in DISCS |
| `flax` | `0.6.3` | PyTree utilities, dataclasses, checkpoints |
| `optax` | `0.1.4` | Provides the temperature schedules in `CO_Experiment` |
| `ml_collections` | `0.1.1` | Config containers used throughout DISCS |
| `absl-py` | `1.3.0` | Logging utilities and (if ever needed) CLI flags |
| `clu` | `0.0.8` | Metric writers required by `discs/common/utils.setup_logging` |
| `numpy` | `1.21.6` | Host-side math + data conversion |
| `tqdm` | `4.64.1` | Progress bars inside experiments |
| `networkx` | `2.6.3` | Graph helpers (only needed if we build temp graphs) |
| `pickle5` | bundled | Loader expects it even if we don’t use the serialized datasets |

Optional but harmless to keep around: `python-sat` (only used by MIS configs). Everything else in the original `requirements.txt` can be omitted unless we later reuse DISCS’ EBMs or language models.

---

## 3. Integration Blueprint

### Step 0 – Understand the pipeline contract

The Spectral Annealing runner loads a solver class via `utils/selector.SolverSelector`. A solver must expose:

1. `__init__(self, instance_name, dataset, run_name=None, random_seed=None, **kwargs)`
2. `solve(self, J)` where `J` is a `scipy.sparse` CSR matrix of couplings.

The new `ISCOSolver` will live in `models/isco_solver.py` and be imported through the selector entry that already exists (`'isco': ('models.isco_solver', 'ISCOSolver')`). The YAML block in `configs/default.yml` (`isco:`) already sketches the runtime parameters we want to surface to the CLI.

### Step 1 – Build a graph-to-DISCS adapter

DISCS’ CO stack expects each batch element to carry the dictionaries produced by `discs.common.utils.graph2edges` (edge lists plus bidirectional copies, masks, and an optional `temperature` scalar). To keep everything in-memory, we can generate that structure straight from the sparse `J` that `DataLoader` hands us:

```python
def build_graph_params(J):
    coo = (J + J.T).tocoo()  # ensure symmetry
    mask = coo.row < coo.col  # upper triangle only
    edge_from = coo.row[mask].astype(np.int32)
    edge_to = coo.col[mask].astype(np.int32)
    # Convert Ising couplings into max-cut weights (negate once to maximize cuts)
    edge_weight = (-coo.data[mask]).astype(np.float32)

    params = {
        'edge_from': edge_from,
        'edge_to': edge_to,
        'edge_weight': edge_weight,
        'bidir_edge_from': np.concatenate([edge_from, edge_to]),
        'bidir_edge_to': np.concatenate([edge_to, edge_from]),
        'bidir_edge_weight': np.concatenate([edge_weight, edge_weight]),
        'num_edges': np.array([edge_from.size], dtype=np.int32),
        'edge_mask': np.ones(edge_from.size * 2, dtype=np.int32),
        'mask': None,  # leave unmasked unless we add fixed spins
    }
    return params
```

Keep `reference_obj` alongside (either a known optimum or a heuristic upper bound such as the best objective returned by our in-tree solvers) because `CO_Experiment` logs progress relative to it.

### Step 2 – Provide an inline graph loader

`CombEBM` calls `discs.common.utils.get_datagen`, which in turn uses `discs.graph_loader.graph_gen.get_graphs(config)`. Add a tiny loader that yields the single instance we are solving so we don’t have to convert datasets to DISCS’ pickle format:

1. **Add** `discs/graph_loader/inline_loader.py` implementing a `InlineGraphGenerator(GraphGenerator)` that receives an in-memory list like `[(instance_id, params_dict, reference_obj)]` and yields it once.
2. **Patch** `discs/graph_loader/graph_gen.py` to route `'inline'` graph types to this generator:

```python
elif config.model.graph_type == 'inline':
    from discs.graph_loader import inline_loader
    return inline_loader.InlineGraphGenerator(config.model.inline_instances)
```

The new config field `inline_instances` is a list so we can batch multiple problems later if needed.

### Step 3 – Build a config factory for iSCO

Inside `models/isco_solver.py` create a helper that merges the three DISCS config layers:

```python
from discs.common import configs as discs_configs
from discs.experiment.configs import co_experiment
from discs.samplers.configs import path_auxiliary_config

def build_isco_config(run_root, instance_name, graph_params, isco_args):
    config = discs_configs.get_config()

    # Experiment block (temperature schedule + bookkeeping)
    config.experiment.update(co_experiment.get_co_default_config())
    config.experiment.chain_length = isco_args.get('chain_length', 10000)
    config.experiment.batch_size = isco_args.get('batch_size', 8)
    config.experiment.t_schedule = isco_args.get('t_schedule', 'exp_decay')
    config.experiment.init_temperature = isco_args.get('init_temperature', 1.5)
    config.experiment.final_temperature = isco_args.get('final_temperature', 0.01)
    config.experiment.decay_rate = isco_args.get('decay_rate', 0.1)
    config.experiment.log_every_steps = isco_args.get('log_every_steps', 200)
    config.experiment.save_samples = isco_args.get('save_samples', False)
    config.experiment.save_root = run_root  # e.g. results/<dataset>/<run_name>/<instance>/

    # Model block
    num_nodes = graph_params['edge_from'].max() + 1 if graph_params['edge_from'].size else J.shape[0]
    config.model.name = 'maxcut'
    config.model.graph_type = 'inline'
    config.model.num_models = 1
    config.model.shape = (num_nodes,)
    config.model.max_num_nodes = num_nodes
    config.model.max_num_edges = graph_params['edge_from'].size
    config.model.inline_instances = [{
        'sample_idx': 0,
        'params': graph_params,
        'reference_obj': float(isco_args.get('reference_obj', 1.0)),
    }]

    # Sampler block
    config.sampler.update(path_auxiliary_config.get_config())
    config.sampler.num_flips = isco_args.get('num_flips', 1)
    config.sampler.adaptive = isco_args.get('adaptive', True)
    config.sampler.target_acceptance_rate = isco_args.get('target_acceptance_rate', 0.574)
    config.sampler.balancing_fn_type = isco_args.get('balancing_fn_type', 'SQRT')
    config.sampler.approx_with_grad = isco_args.get('approx_with_grad', True)

    return config
```

This mirrors the fields already exposed in `configs/default.yml` → `isco: params: …`, so command-line overrides will flow naturally when `merge_args_with_config` runs.

### Step 4 – Create the inline datagen iterator

`CO_Experiment._initialize_model_and_sampler` expects `model.make_init_params` to return a list where each entry is `(sample_idx, params_dict, reference_obj)` (see `sampling.py`). With the inline loader in place, you just need a thin iterator:

```python
class InlineGraphGenerator(GraphGenerator):
    def __init__(self, inline_instances):
        super().__init__()
        self.instances = inline_instances
        self._max_num_nodes = inline_instances[0]['params']['edge_from'].max() + 1
        self._max_num_edges = inline_instances[0]['params']['edge_from'].size
        self._num_instances = len(inline_instances)

    def graph2edges(self, g):
        raise NotImplementedError  # unused

    def sample_gen(self, phase, repeat=False):
        for item in self.instances:
            yield (
                item['sample_idx'],
                item['params'],
                item['reference_obj'],
            )
        if repeat:
            while True:
                yield from self.sample_gen(phase, repeat=False)
```

Store it under `discs/graph_loader/inline_loader.py` so `graph_gen.get_graphs` can import it.

### Step 5 – Implement `ISCOSolver`

Skeleton:

```python
class ISCOSolver:
    def __init__(self, instance_name, dataset, run_name=None, random_seed=None, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.run_name = run_name or "isco"
        self.random_seed = random_seed or 42
        self.isco_args = kwargs

    def solve(self, J):
        graph_params = build_graph_params(J)
        run_root = Path(f"results/{self.dataset}/{self.run_name}/{self.instance_name}")
        run_root.mkdir(parents=True, exist_ok=True)

        config = build_isco_config(str(run_root), self.instance_name, graph_params, self.isco_args)

        model = maxcut.build_model(config)
        sampler = path_auxiliary.build_sampler(config)
        experiment = sampling.CO_Experiment(config)
        evaluator = co_eval.build_evaluator(config)
        saver = experiment_saver.build_saver(config)

        experiment.get_results(model, sampler, evaluator, saver)

        # Parse saver output (results.pkl) and push to our CSV format
        best_spins, best_energy = self._extract_best(run_root, J)
        self._store_results(best_spins, best_energy, run_root)
```

Implementation notes:

- `CO_Experiment` already updates the temperature at every step (`sampling.py`, `_compute_chain`). The schedule parameters we injected earlier govern the annealing profile.
- `experiment_saver.save_co_resuts` writes `results.pkl` containing `best_ratio`, `best_samples`, and `running_time`. Reuse that to populate the pipeline’s CSV (mirroring what `SASolver` does in `models/simulated_annealing.py`).
- If you need deterministic behaviour, fold `random_seed` into DISCS’ PRNG by calling `config.experiment.seed = self.random_seed` (and pass it to `jax.random.PRNGKey` before `experiment.get_results`).

### Step 6 – Wire up configs and selector

- Ensure `utils/selector.SolverSelector` keeps the `'isco': ('models.isco_solver', 'ISCOSolver')` entry (already present).
- The sample config block in `configs/default.yml` is ready; after the solver class exists you can run:

```bash
python main.py --solver isco --dataset barabasi_albert_m20 --instance ba_64_1_0.txt
```

to exercise the full path.

### Step 7 – Results + logging

- DISCS writes logs, plots, and pickles under `config.experiment.save_root`. Point this to the pipeline’s standard folder (`results/<dataset>/<run_name>/<instance>/isco/`) so all solvers remain co-located.
- After `experiment.get_results` finishes, parse `results.pkl` to extract:
  - `best_samples` → best binary assignment. Convert to spin values (`{0,1} → {-1,+1}`) before writing our CSV.
  - `best_ratio_mean` × reference value → actual energy (max-cut weight). Convert back to Ising energy with the same penalty we used in `build_graph_params`.
  - `running_time` → populate the `time` column.

Reuse `_store_results` logic from `models/simulated_annealing.py` or `models/tabu_search.py` for consistent CSV schema.

---

## 4. Manual debugging via DISCS CLI (optional)

Until the inline loader is in place you can still sanity check PAS by using the shipped shell wrapper:

```bash
cd models/iSCO/discs
model=maxcut graph_type=er sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh
```

The CLI pulls configs from `discs/run_configs`, which is useful to compare against our custom setup (e.g., confirm default `chain_length`, `t_schedule`, etc.).

---

## 5. Validation Checklist

1. **Unit conversion test** – Feed a tiny (n=8) ER graph where we know the optimum and make sure `build_graph_params` plus the result parsing reproduces the correct Ising energy.
2. **Sampler sanity** – Run with `batch_size=1`, `chain_length=200`, `t_schedule='constant'` to confirm the PAS step does flip bits and `best_ratio` improves over time (watch `results.pkl`).
3. **Integration smoke test** – Execute `python main.py --solver isco ...` on one dataset instance, check that:
   - `results/<dataset>/<run_name>/<instance>.csv` gains a new row with solver=`isco`.
   - `results/<dataset>/<run_name>/<instance>/isco/results.pkl` exists and contains chains.
4. **Performance baselining** – Compare energies/runtime vs. `spectral_annealing` and `tabu_search` on the same seed to ensure parity before tuning.

Once this checklist is green we can proceed with the actual code integration and, later on, optional enhancements (multi-instance batching, saving PAS trajectories, exposing extra PAS knobs via CLI).

---

Feel free to annotate/correct this plan; after approval we can implement the inline loader, solver class, and config factory exactly as outlined above.

