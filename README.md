# RLABC: Reinforcement Learning for Accelerator Beamline Control

This repository contains the source code for the RLABC framework described in:

> A. Ibrahim, F. Ratnikov, M. Kaledin, A. Petrenko, D. Derkach,
> "RL-ABC: Reinforcement Learning for Accelerator Beamline Control,"
> *Computer Physics Communications* (2026).

RLABC is a Python framework that automatically transforms standard Elegant beamline lattice configurations into reinforcement learning environments, enabling RL-based optimization of particle accelerator beamlines.

## Prerequisites

This software requires the **Elegant** simulation program to be installed as an external dependency. Elegant is not a Python package; it is a standalone beam dynamics simulation code that RLABC invokes via subprocess calls during training and evaluation.

**Elegant** (ELEctron Generation ANd Tracking) is developed and maintained by the Advanced Photon Source at Argonne National Laboratory:

- Download and installation: https://www.aps.anl.gov/Accelerator-Operations-Physics/Software
- Reference: M. Borland, "elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation," Advanced Photon Source LS-287, Argonne National Laboratory (2000). https://doi.org/10.2172/761286

The Elegant distribution includes the **SDDS Toolkit** (Self Describing Data Sets), a set of command-line utilities for data I/O. RLABC uses the following executables from this toolkit:

| Executable      | Purpose within RLABC                                      |
|-----------------|------------------------------------------------------------|
| `elegant`       | Runs beam dynamics simulations                             |
| `sdds2stream`   | Extracts particle coordinate data from SDDS output files   |
| `sddsquery`     | Queries column names in SDDS files                         |
| `sddsprocess`   | Post-processes SDDS data (coordinate transformations)      |

After installing Elegant, note the paths to:
1. The directory containing the `elegant` binary and SDDS utilities (configured as `platform.elegant_path`)
2. The `defns.rpn` file from the SDDS Toolkit (configured as `platform.sdds_path`)

These paths are set in `config.json` or `config.yaml` (see the Configuration section below).

## Directory Structure

```
acc_elegant_rl_training/
├── train.py                  # Main training script (custom DDPG agent)
├── sb_train.py               # Training script using Stable-Baselines3 (SAC, DDPG, TD3)
├── config_manager.py         # Configuration loading and CLI argument parsing
├── config.json               # Configuration file (JSON format)
├── config.yaml               # Configuration file (YAML format)
├── machine.lte               # Elegant lattice file for the VEPP-5 beamline
├── track.ele                 # Elegant command file (beam and simulation settings)
├── requirements.txt          # Python dependencies (pip)
├── pyproject.toml            # Project metadata and dependencies
├── environment.yml           # Conda environment specification
│
├── rl_framework/             # Core framework modules
│   ├── __init__.py
│   ├── Environment.py        # Gymnasium environment (state, action, reward logic)
│   ├── Elegant.py            # Elegant simulation wrapper (SDDS I/O, lattice processing)
│   ├── Utils.py              # Lattice parsing, observation construction, helpers
│   ├── visulize.py           # Beam optics and phase-space visualization
│   └── Agents/
│       ├── __init__.py
│       └── DDPG.py           # DDPG agent (actor, critic, replay buffer, noise)
│
├── classical_optimizers/     # Baseline optimization methods
│   ├── bayesian_optimization.py   # Bayesian optimization via bayesian-optimization / skopt
│   ├── scipy_optimization.py      # Differential evolution via SciPy
│   └── README.md
│
│
├── docs/                     # Additional documentation
│   ├── SETUP.md              # Platform-specific installation details
│   ├── DEPLOYMENT_GUIDE.md   # Deployment on HPC clusters
│   └── ...
└── 
```

## Requirements

- Elegant and the SDDS Toolkit (see Prerequisites above)
- Python 3.10

The Python dependencies are listed in `requirements.txt`. The main libraries used are:

| Library             | Purpose                                    |
|---------------------|--------------------------------------------|
| PyTorch >= 2.0      | Neural network training                    |
| Gymnasium >= 0.29   | RL environment interface                   |
| Stable-Baselines3   | Alternative RL algorithms (SAC, TD3)       |
| NumPy, Pandas       | Numerical computation and data handling    |
| SciPy               | Differential evolution baseline            |
| Matplotlib          | Visualization                              |
| TensorBoard         | Training monitoring                        |
| PyYAML              | Configuration file parsing                 |

## Installation

1. **Install Elegant and the SDDS Toolkit** following the instructions in the Prerequisites section above.

2. **Install Python dependencies** using one of:

   ```bash
   # Option A: pip (in a virtual environment)
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Option B: Conda
   conda env create -f environment.yml
   conda activate acc-elegant-rl
   ```

3. **Configure paths** by editing `config.json` (or `config.yaml`). Set `platform.elegant_path` to the directory containing the Elegant binary, and `platform.sdds_path` to the path of the `defns.rpn` file from the SDDS Toolkit. For example:

   ```json
   {
     "platform": {
       "os_type": "linux",
       "elegant_path": "/usr/local/elegant/bin/",
       "sdds_path": "/usr/local/elegant/defns.rpn"
     }
   }
   ```

## Running the Program

### Training with the custom DDPG agent

```bash
python train.py
```

This reads `config.json` by default. Configuration parameters can be overridden via command-line arguments:

```bash
python train.py --training.n_episodes 1000 --training.seed 42 --agent.alpha 0.0002
```

### Training with Stable-Baselines3

```bash
python sb_train.py --algo SAC --total-timesteps 100000 --seed 42
```

Supported algorithms: `SAC`, `DDPG`, `TD3`.

### Evaluating a pretrained agent

The `models/` directory contains pretrained DDPG checkpoints. To run evaluation using these weights:

```bash
python train.py --training.load_model true --training.n_episodes 10
```

### Classical optimization baselines

```bash
# Differential evolution
python classical_optimizers/scipy_optimization.py

# Bayesian optimization
python classical_optimizers/bayesian_optimization.py
```

## Configuration

Training is configured through `config.json` or `config.yaml`. The main parameter groups are:

| Section       | Key parameters                                                       |
|---------------|----------------------------------------------------------------------|
| `platform`    | `elegant_path`, `sdds_path`, `os_type`                               |
| `simulation`  | `input_beamline_file`, `input_beam_file`, `beamline_name`            |
| `environment` | `init_num_particles` (default: 1000), `n_bins` (default: 5), `stage` |
| `training`    | `n_episodes`, `seed`, `greedy` (exploration noise scale)             |
| `agent`       | `agent_type`, `alpha`, `beta`, `batch_size`, `gamma`, `tau`          |
| `logging`     | `results_path`, `logger_file_name`                                   |

The `stage` parameter controls curriculum learning. When set to `null`, the full beamline (all 37 parameters) is optimized. Integer values (1--4) activate progressively larger subsets of the beamline, as described in the paper.

## Sample Input and Output

### Input files

- `machine.lte`: Elegant lattice file defining the VEPP-5 positron injection beamline. Contains 11 quadrupole magnets (Q1L0--Q1L10), 4 dipole magnets (BM1--BM4), drift spaces, apertures (MAXAMP elements), and an RF debuncher cavity.
- `track.ele`: Elegant command file specifying the beam properties (400 MeV central momentum, 1000 particles) and simulation outputs (sigma matrix, centroid, Twiss parameters, floor coordinates).

### Output from a test run

Running `python train.py` with the default configuration (`config.json`, `n_episodes=10`) produces:

- **Console output**: Per-episode reward and particle transmission count.
- **CSV log** (in `results_{seed}/`): Per-step records including episode number, reward, surviving particle count, current element, and the full set of magnet parameter values at each step.
- **TensorBoard logs** (in `results_{seed}/`): Scalar summaries of reward and transmission for monitoring via `tensorboard --logdir=results_{seed}/`.

A pretrained evaluation log is provided in `logs/ddpg_eval_3036000.csv`. Each row records one step within an evaluation episode: the episode number, cumulative reward, surviving particle count, the watch point name, and the full dictionary of magnet parameter values applied at that step. The pretrained agent achieves 72.2% particle transmission on the VEPP-5 beamline (722 out of 1000 particles reaching the beamline exit).

## How It Works

RLABC reformulates beamline tuning as a Markov decision process. The framework:

1. **Preprocesses** the Elegant lattice file to insert diagnostic watch points before each tunable magnet (`rl_framework/Utils.py`).
2. **Constructs** a 57-dimensional state vector at each watch point from beam statistics (percentiles), a spatial histogram, the covariance matrix, and aperture parameters (`rl_framework/Utils.py: process_particle_data`).
3. **Runs Elegant** simulations between consecutive watch points via SDDS file exchange (`rl_framework/Elegant.py`).
4. **Computes** a reward based on particle transmission, penalizing early beam loss (`rl_framework/Environment.py`).
5. **Trains** an RL agent (DDPG by default) to select magnet parameters that maximize transmission (`rl_framework/Agents/DDPG.py`).

The environment implements the Gymnasium interface, making it compatible with any RL library that supports this standard.

## License

MIT License. See the repository root for the full license text.

## Contact

For questions regarding the code, please contact the corresponding author: aibrahim@hse.ru
