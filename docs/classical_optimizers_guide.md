# Classical Optimizers Guide

Non-RL baselines for ACC beamline optimisation. Both scripts use the same `config.json` and `ElegantWrapper` as the RL pipeline.

## Scripts

| Script | Algorithm | Best for |
|---|---|---|
| `scipy_optimization.py` | SciPy Differential Evolution | Large iteration budgets, global search |
| `bayesian_optimization.py` | Bayesian Optimisation (GP) | Sample-efficient search, fewer evaluations |

---

## Quick Start

Run from the `acc_elegant_rl_training/` directory:

```bash
# SciPy Differential Evolution
python classical_optimizers/scipy_optimization.py --maxiter 100 --seed 42

# Bayesian Optimisation (bayes_opt backend)
python classical_optimizers/bayesian_optimization.py --n-calls 200 --seed 42

# Bayesian Optimisation (scikit-optimize backend)
python classical_optimizers/bayesian_optimization.py --backend skopt --n-calls 200 --seed 42
```

## CLI Options

### scipy_optimization.py

| Flag | Default | Description |
|---|---|---|
| `--config` | `../config.json` | Path to config file |
| `--maxiter` | `2000` | Max DE generations |
| `--seed` | from config | Random seed |
| `--workers` | `1` | Parallel workers |
| `--checkpoint` | `scipy_checkpoint.pkl` | Checkpoint file path |
| `--checkpoint-interval` | `100` | Save checkpoint every N iterations |
| `--resume` | off | Resume from checkpoint |
| `--results-dir` | `results_scipy_{seed}/` | Elegant simulation output directory |

### bayesian_optimization.py

| Flag | Default | Description |
|---|---|---|
| `--config` | `../config.json` | Path to config file |
| `--backend` | `bayes_opt` | `bayes_opt` or `skopt` |
| `--n-calls` | `500` | Total objective evaluations |
| `--n-init` | `10` | Random initial evaluations |
| `--seed` | from config | Random seed |
| `--results-dir` | `results_bayesopt_{seed}/` | Elegant simulation output directory |

## Output Files

Saved in the working directory (`acc_elegant_rl_training/`):

| File | Contents |
|---|---|
| `scipy_log_{seed}.csv` | Per-evaluation log: eval number, reward, particles, best reward, variables |
| `scipy_result_{seed}.pkl` | Final pickle with best parameters, history, and verification reward |
| `scipy_checkpoint.pkl` | DE checkpoint for resuming (SciPy only) |
| `bayesopt_log_{seed}.csv` | Per-evaluation log (Bayesian optimisation) |
| `bayesopt_result_{seed}.pkl` | Final pickle with best parameters and history |

The CSV logs are comparable to the RL training CSV (`big_beamline_original_{seed}.csv`).

## Configuration

Both scripts read from `config.json`. The relevant sections are:

- **`platform`** — `elegant_path`, `sdds_path`
- **`simulation`** — `input_beamline_file`, `input_beam_file`, `beamline_name`, `output_beamline_file`, `elegant_input_filename`, `override_dynamic_command`, `overridden_command`
- **`environment`** — `init_num_particles`, `reset_specific_keys_bool`
- **`training`** — `seed` (used as default when `--seed` is not passed)

## Resuming (SciPy only)

```bash
# First run (creates checkpoint)
python classical_optimizers/scipy_optimization.py --maxiter 500 --seed 42

# Resume from where it left off
python classical_optimizers/scipy_optimization.py --maxiter 500 --seed 42 --resume
```
