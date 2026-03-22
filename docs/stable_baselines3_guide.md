# Stable-Baselines3 Training Guide

Training ACC beamline magnet parameters using industry-standard RL algorithms from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Supported Algorithms](#supported-algorithms)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Output Files](#output-files)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Resuming Training](#resuming-training)
- [Evaluation](#evaluation)
- [Custom Network Architectures](#custom-network-architectures)
- [Comparison with Custom DDPG Agent](#comparison-with-custom-ddpg-agent)
- [Tips for Choosing an Algorithm](#tips-for-choosing-an-algorithm)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Overview

`sb_train.py` is a drop-in training script that uses the same `ACCElegantEnvironment` and `config.json` as the rest of the project, but swaps in SB3's battle-tested implementations of DDPG, SAC, and TD3 instead of the custom from-scratch agent.

**Why use SB3?**

- Well-tested implementations with thousands of users
- Built-in logging (TensorBoard, CSV), checkpointing, and evaluation
- Easy algorithm switching (DDPG, SAC, TD3) without code changes
- Reproducible results with seeded training
- Active maintenance and community support

**When to use the custom agent instead?**

- You need full control over the training loop internals
- You want to experiment with non-standard update rules
- You need the sequential element-by-element action conversion (`convert=True`)

## Requirements

### Python Packages

| Package | Minimum Version | Purpose |
|---|---|---|
| `stable-baselines3` | >= 2.0.0 | RL algorithms (DDPG, SAC, TD3) |
| `gymnasium` | >= 0.29.1 | Environment interface |
| `torch` | >= 2.0.0 | Neural network backend |
| `numpy` | >= 1.21.0 | Numerical computing |
| `tensorboard` | >= 2.7.0 | Training visualisation |
| `pandas` | >= 1.3.0 | Data handling (used by environment) |
| `scikit-learn` | >= 1.0.0 | Preprocessing (used by environment) |
| `pyyaml` | >= 6.0 | Config file loading |

All of these are already listed in the project's `requirements.txt`, `pyproject.toml`, and `environment.yml`.

### Installation

If starting from a fresh environment, use **one** of:

```bash
# Option 1: conda (recommended)
conda env create -f environment.yml
conda activate acc-elegant-rl

# Option 2: pip
pip install -r requirements.txt

# Option 3: uv (fastest, uses lockfile for exact reproducibility)
uv sync
```

### Verify installation

```bash
python -c "
import stable_baselines3
import gymnasium
import torch
print(f'SB3:        {stable_baselines3.__version__}')
print(f'Gymnasium:  {gymnasium.__version__}')
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.cuda.is_available()}')
"
```

### External Dependencies

The `elegant` particle accelerator simulator and SDDS toolkit must be installed and the paths set in `config.json`:

```json
{
  "platform": {
    "elegant_path": "/path/to/sdds/bin/",
    "sdds_path": "/path/to/sdds/defns.rpn"
  }
}
```

## Quick Start

All commands are run from the `acc_elegant_rl_training/` directory.

```bash
# Train with DDPG (default), 100k timesteps
python sb_train.py

# Train with SAC
python sb_train.py --algo SAC --total-timesteps 200000

# Train with TD3, custom seed
python sb_train.py --algo TD3 --seed 42

# Evaluate a saved model
python sb_train.py --mode evaluate --resume sb_models/ddpg_0/ddpg_final.zip
```

## Supported Algorithms

### DDPG (Deep Deterministic Policy Gradient)

- **Type:** Off-policy, deterministic actor-critic
- **Action noise:** Required (Gaussian or Ornstein-Uhlenbeck)
- **Best for:** Continuous control with low-dimensional action spaces
- **SB3 default `learning_starts`:** 100 timesteps

```bash
python sb_train.py --algo DDPG --noise-type gaussian --noise-sigma 0.1
```

### SAC (Soft Actor-Critic)

- **Type:** Off-policy, stochastic actor-critic with entropy regularisation
- **Action noise:** Not needed (entropy handles exploration automatically)
- **Best for:** General-purpose continuous control; usually the safest default choice
- **SB3 default `learning_starts`:** 100 timesteps

```bash
python sb_train.py --algo SAC
```

### TD3 (Twin Delayed DDPG)

- **Type:** Off-policy, deterministic actor-critic (improved DDPG)
- **Action noise:** Required (same options as DDPG)
- **Best for:** Environments where DDPG is unstable; uses twin critics and delayed updates
- **SB3 default `learning_starts`:** 100 timesteps

```bash
python sb_train.py --algo TD3 --noise-type ou --noise-sigma 0.2
```

## CLI Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config.json` | Path to JSON or YAML config file |
| `--algo` | str | `DDPG` | Algorithm: `DDPG`, `SAC`, or `TD3` |
| `--total-timesteps` | int | `100000` | Total training timesteps |
| `--learning-rate` | float | from config | Learning rate (overrides `agent.alpha`) |
| `--batch-size` | int | from config | Mini-batch size (overrides `agent.batch_size`) |
| `--gamma` | float | from config | Discount factor |
| `--tau` | float | from config | Soft update coefficient |
| `--buffer-size` | int | from config | Replay buffer capacity |
| `--noise-type` | str | from config | `gaussian`, `ou`, or `none` (DDPG/TD3 only) |
| `--noise-sigma` | float | `0.1` | Noise standard deviation |
| `--seed` | int | from config | Random seed |
| `--eval-freq` | int | `5000` | Evaluate every N timesteps |
| `--eval-episodes` | int | `1` | Episodes per evaluation |
| `--save-freq` | int | `10000` | Checkpoint every N timesteps |
| `--log-interval` | int | `10` | Log to console every N episodes |
| `--log-dir` | str | auto | TensorBoard/CSV log directory |
| `--models-dir` | str | auto | Model checkpoint directory |
| `--resume` | str | — | Path to `.zip` model to resume from |
| `--cpu` | flag | off | Force CPU even if CUDA is available |
| `--verbose` | int | `1` | SB3 verbosity: 0=silent, 1=info, 2=debug |
| `--net-arch` | int[] | SB3 default | MLP hidden layers, e.g. `--net-arch 400 300` |
| `--mode` | str | `train` | `train` or `evaluate` |

### Parameter Priority

CLI flags override config file values, which override built-in defaults:

```
CLI flag  >  config.json  >  SB3 default
```

For example, `--learning-rate 3e-4` overrides `agent.alpha` in `config.json`, which overrides SB3's built-in default of `1e-3`.

## Configuration

`sb_train.py` reads the same `config.json` used by `train.py` and the classical optimizers. Here is how config sections map to SB3 parameters:

### `config.json` → SB3 mapping

| Config Key | SB3 Parameter | Notes |
|---|---|---|
| `agent.alpha` | `learning_rate` | SB3 uses one LR for both actor and critic |
| `agent.batch_size` | `batch_size` | |
| `agent.gamma` | `gamma` | |
| `agent.tau` | `tau` | |
| `agent.max_size` | `buffer_size` | |
| `agent.noise_type` | `action_noise` | Converted to `NormalActionNoise` or `OrnsteinUhlenbeckActionNoise` |
| `training.seed` | `seed` | |
| `environment.*` | Environment constructor args | Passed directly to `ACCElegantEnvironment` |
| `platform.*` | `elegant_path`, `sdds_path` | Paths to simulator binaries |
| `simulation.*` | Beamline file paths | |

**Important difference:** The custom DDPG agent has separate learning rates for actor (`alpha`) and critic (`beta`). SB3 uses a single `learning_rate` for both networks. The script uses `agent.alpha` as the SB3 learning rate.

### Minimal config.json

```json
{
  "platform": {
    "elegant_path": "/path/to/sdds/bin/",
    "sdds_path": "/path/to/sdds/defns.rpn"
  },
  "simulation": {
    "input_beamline_file": "machine.lte",
    "input_beam_file": "track",
    "output_beamline_file": "updated_machine_{seed}.lte",
    "beamline_name": "machine"
  },
  "environment": {
    "n_bins": 5,
    "init_num_particles": 1000
  },
  "training": {
    "seed": 0
  },
  "agent": {
    "alpha": 0.0001,
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005,
    "noise_type": "gaussian"
  }
}
```

## How It Works

### Environment Compatibility

The `ACCElegantEnvironment` is a standard Gymnasium `Env`:

- **Observation space:** `Box(shape=(57,))` — particle statistics, histograms, covariance, and ellipse parameters at each watch point
- **Action space:** `Box(shape=(4,), low=[-20, -0.005, -0.005, -0.005], high=[20, 0.005, 0.005, 0.005])` — K1, HKICK, VKICK, FSE for the current magnet
- **`reset()`** returns `(observation, info_dict)`
- **`step(action)`** returns `(observation, reward, done, truncated, info)`

SB3 algorithms output actions scaled to the action space bounds. The environment's `step(action, convert=False)` receives these already-scaled values directly — no additional conversion needed.

### Episode Structure

Each episode walks through the beamline magnet-by-magnet:

1. `reset()` initialises particles and runs the simulation with zero magnet values
2. Each `step()` sets the next magnet's parameters (K1, HKICK, VKICK or FSE), re-runs the simulation, and reads particle counts at the next watch point
3. The episode ends when all magnets are set or too few particles survive (<=3)
4. Typical episode length: ~7 steps (one per controllable magnet element)

### Reward

- **Surviving particles / initial particles** — ratio of beam transmission
- **Bonus** when particles survive to the final watch point
- **Penalty** when all particles are lost before the end

### Timesteps vs Episodes

SB3 counts **timesteps** (individual `env.step()` calls), not episodes. With ~7 steps per episode:

| Timesteps | Approximate Episodes |
|---|---|
| 200 | ~29 |
| 1,000 | ~143 |
| 10,000 | ~1,430 |
| 100,000 | ~14,300 |
| 1,000,000 | ~143,000 |

## Output Files

### Directory Structure

```
sb_logs/{algo}_{seed}/
├── progress.csv                 # Per-episode metrics in CSV format
└── events.out.tfevents.*        # TensorBoard binary log

sb_models/{algo}_{seed}/
├── {algo}_ckpt_10000_steps.zip  # Periodic checkpoints
├── {algo}_ckpt_20000_steps.zip
└── {algo}_final.zip             # Final model after training
```

### progress.csv Columns

| Column | Description |
|---|---|
| `rollout/ep_len_mean` | Average episode length (steps) |
| `rollout/ep_rew_mean` | Average episode reward |
| `time/episodes` | Total episodes completed |
| `time/fps` | Training speed (env steps per second) |
| `time/total_timesteps` | Total timesteps completed |
| `time/time_elapsed` | Wall-clock seconds |
| `train/actor_loss` | Actor (policy) network loss |
| `train/critic_loss` | Critic (Q-value) network loss |
| `train/ent_coef` | Entropy coefficient (SAC only) |
| `train/ent_coef_loss` | Entropy coefficient loss (SAC only) |
| `train/learning_rate` | Current learning rate |
| `train/n_updates` | Total gradient updates |
| `particles/count` | Surviving particles at episode end |
| `particles/episode` | Cumulative episode counter |

**Note:** `progress.csv` will be empty if `total_timesteps` < `learning_starts` (default 100), because SB3 only logs once training updates begin.

### Model Files (.zip)

Each `.zip` contains the full model state: policy weights, optimizer state, and replay buffer metadata. They can be loaded for resuming training or evaluation.

## Monitoring with TensorBoard

```bash
# From the acc_elegant_rl_training/ directory
tensorboard --logdir sb_logs/ --port 6006
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

**Key metrics to watch:**

- `rollout/ep_rew_mean` — is reward trending up?
- `train/critic_loss` — should decrease and stabilise
- `train/actor_loss` — may fluctuate but shouldn't diverge
- `particles/count` — the actual optimisation objective

**Comparing runs:**

```bash
# Compare DDPG vs SAC vs TD3
tensorboard --logdir sb_logs/ --port 6006
# All sub-directories are shown as separate runs
```

## Resuming Training

```bash
# Resume from the final model
python sb_train.py --resume sb_models/ddpg_0/ddpg_final.zip --total-timesteps 50000

# Resume from a specific checkpoint
python sb_train.py --resume sb_models/ddpg_0/ddpg_ckpt_10000_steps.zip --total-timesteps 50000

# Resume and change algorithm parameters
python sb_train.py --resume sb_models/sac_0/sac_final.zip --learning-rate 5e-5 --total-timesteps 100000
```

**Note:** When resuming, the replay buffer from the checkpoint is restored automatically. The `--total-timesteps` value is the number of *new* timesteps to train, not a cumulative total.

## Evaluation

```bash
# Evaluate a trained model for 5 episodes
python sb_train.py --mode evaluate --resume sb_models/ddpg_0/ddpg_final.zip --eval-episodes 5

# Evaluate SAC model
python sb_train.py --mode evaluate --resume sb_models/sac_0/sac_final.zip --eval-episodes 3
```

Evaluation runs deterministic rollouts (no exploration noise) and reports:
- Total reward per episode
- Number of surviving particles
- Number of steps

## Custom Network Architectures

By default, SB3 uses `[400, 300]` hidden layers for both actor and critic. Override with `--net-arch`:

```bash
# Match the custom DDPG agent's architecture (800, 600, 512, 256)
python sb_train.py --net-arch 800 600 512 256

# Smaller network for faster training
python sb_train.py --net-arch 256 256

# Larger network
python sb_train.py --net-arch 512 512 256
```

## Comparison with Custom DDPG Agent

| Feature | Custom DDPG (`train.py`) | SB3 (`sb_train.py`) |
|---|---|---|
| Algorithms | DDPG only | DDPG, SAC, TD3 |
| Learning rates | Separate actor/critic (`alpha`, `beta`) | Single `learning_rate` |
| Network architecture | Fixed (800→600→512→256) | Configurable via `--net-arch` |
| Action conversion | `convert=True` (actor outputs [-1,1], env scales) | SB3 handles scaling internally |
| Training loop | Manual episode loop | SB3's `model.learn()` |
| Logging | Custom CSV + TensorBoard | SB3 built-in (CSV + TensorBoard) |
| Checkpointing | Manual `save_models()` / `load_models()` | Automatic via `CheckpointCallback` |
| Replay buffer save/load | Manual pickle | Included in `.zip` checkpoint |
| Greedy decay | Linear decay over episodes | Fixed noise sigma (configure externally) |
| Evaluation | Built into agent | Separate `--mode evaluate` |
| Episode counting | `n_episodes` | `total_timesteps` (see conversion table above) |

## Tips for Choosing an Algorithm

| Scenario | Recommended | Why |
|---|---|---|
| First experiment / unsure | **SAC** | Most stable; automatic entropy tuning handles exploration |
| Matching the existing baseline | **DDPG** | Closest to the custom agent for fair comparison |
| DDPG is unstable | **TD3** | Twin critics + delayed updates reduce overestimation |
| Long training budget | **SAC** | Entropy regularisation helps avoid local optima |
| Fast iteration / debugging | **DDPG** | Simplest; fewer hyperparameters |

### Recommended Hyperparameters

**Conservative start (any algorithm):**
```bash
python sb_train.py --algo SAC --total-timesteps 100000 --learning-rate 1e-4 --batch-size 128
```

**Aggressive exploration (DDPG/TD3):**
```bash
python sb_train.py --algo DDPG --noise-sigma 0.3 --noise-type ou --total-timesteps 200000
```

**Large-scale run:**
```bash
python sb_train.py --algo SAC --total-timesteps 1000000 --save-freq 50000 --batch-size 256
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Your shell is using the wrong Python. Check with `which python`:

```bash
# If using pyenv, unset the override
pyenv shell --unset

# Or explicitly use the conda Python
/opt/anaconda3/bin/python sb_train.py
```

### `progress.csv` is empty

Training hasn't started yet. SB3 collects random samples for `learning_starts` timesteps (default 100) before the first gradient update. Use `--total-timesteps 200` or more.

### CUDA out of memory

```bash
python sb_train.py --cpu --batch-size 64
```

### MPS (Apple Silicon) not used

SB3 does not officially support MPS. The script falls back to CPU on macOS. This is expected and does not affect training correctness — only speed.

### Training is slow

Each timestep runs a full elegant simulation, which is the bottleneck. To speed things up:
- Reduce `--init_num_particles` via config
- Use `--save-freq` with a higher value to reduce I/O
- On HPC: use `--simulation.override_dynamic_command true` with parallel elegant

### Model file not found when resuming

Make sure you include the full path and `.zip` extension is handled automatically:
```bash
# Both work:
python sb_train.py --resume sb_models/ddpg_0/ddpg_final.zip
python sb_train.py --resume sb_models/ddpg_0/ddpg_final
```

## Examples

### Full training pipeline

```bash
# 1. Train DDPG for 100k steps
python sb_train.py --algo DDPG --total-timesteps 100000 --seed 0

# 2. Check TensorBoard
tensorboard --logdir sb_logs/ --port 6006

# 3. Evaluate the best checkpoint
python sb_train.py --mode evaluate --resume sb_models/ddpg_0/ddpg_final.zip --eval-episodes 5

# 4. Compare with SAC
python sb_train.py --algo SAC --total-timesteps 100000 --seed 0

# 5. Compare in TensorBoard (both runs visible)
tensorboard --logdir sb_logs/ --port 6006
```

### HPC / SLURM submission

```bash
#!/bin/bash
#SBATCH --job-name=sb3_sac
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --output=sb3_sac_%j.log

module load cuda/11.8
conda activate acc-elegant-rl

python sb_train.py \
    --algo SAC \
    --total-timesteps 1000000 \
    --seed $SLURM_JOB_ID \
    --save-freq 50000 \
    --log-interval 10
```

### Multi-seed comparison

```bash
for seed in 0 1 2 3 4; do
    python sb_train.py --algo SAC --total-timesteps 100000 --seed $seed &
done
wait
tensorboard --logdir sb_logs/
```
