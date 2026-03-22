#!/usr/bin/env python
"""
Stable-Baselines3 Training Script for ACC Elegant Simulations

Trains SAC, DDPG, or TD3 agents using Stable-Baselines3 to optimize
ACC beamline parameters.  Uses the same config.json and environment
as the custom RL pipeline.

Usage:
    python sb_train.py
    python sb_train.py --config config.json --algo SAC --total-timesteps 200000
    python sb_train.py --algo DDPG --seed 42 --noise-type ou
    python sb_train.py --algo TD3 --learning-rate 3e-4
    python sb_train.py --resume sb_models/ddpg_0/ddpg_final.zip
    python sb_train.py --mode evaluate --resume sb_models/ddpg_0/ddpg_final.zip
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
tempfile.tempdir = '/tmp'

import numpy as np
import torch

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from config_manager import ConfigManager
from rl_framework.Utils import set_seed, setLogger
from rl_framework.Environment import ACCElegantEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

ALGO_MAP = {
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
}


# ---------------------------------------------------------------------------
# Custom callback – logs particle counts to TensorBoard
# ---------------------------------------------------------------------------
class ParticleLoggingCallback(BaseCallback):
    """Writes particle-survival metrics that SB3 doesn't know about."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "number_of_particles" in info:
                self.logger.record(
                    "particles/count", info["number_of_particles"]
                )
            if info.get("done", False):
                self.episode_count += 1
                self.logger.record("particles/episode", self.episode_count)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Stable-Baselines3 training for ACC beamline optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sb_train.py
  python sb_train.py --algo SAC --total-timesteps 200000
  python sb_train.py --config config.json --algo DDPG --seed 42
  python sb_train.py --algo TD3 --learning-rate 3e-4
  python sb_train.py --resume sb_models/ddpg_0/ddpg_final.zip
  python sb_train.py --mode evaluate --resume sb_models/sac_0/sac_final.zip
        """,
    )

    p.add_argument(
        "--config", type=str, default="config.json",
        help="Path to JSON/YAML config file (default: config.json)",
    )
    p.add_argument(
        "--algo", type=str, default="DDPG",
        choices=list(ALGO_MAP.keys()),
        help="RL algorithm (default: DDPG)",
    )
    p.add_argument(
        "--total-timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100 000)",
    )
    p.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (overrides config agent.alpha)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config agent.batch_size)",
    )
    p.add_argument(
        "--gamma", type=float, default=None,
        help="Discount factor (overrides config)",
    )
    p.add_argument(
        "--tau", type=float, default=None,
        help="Soft-update coefficient (overrides config)",
    )
    p.add_argument(
        "--buffer-size", type=int, default=None,
        help="Replay buffer size (overrides config)",
    )
    p.add_argument(
        "--noise-type", type=str, default=None,
        choices=["gaussian", "ou", "none"],
        help="Action noise type for DDPG/TD3 (default from config or gaussian)",
    )
    p.add_argument(
        "--noise-sigma", type=float, default=0.1,
        help="Action noise standard deviation (default: 0.1)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)",
    )
    p.add_argument(
        "--eval-freq", type=int, default=5000,
        help="Evaluate every N timesteps (default: 5000)",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=1,
        help="Episodes per evaluation (default: 1)",
    )
    p.add_argument(
        "--save-freq", type=int, default=10_000,
        help="Save checkpoint every N timesteps (default: 10 000)",
    )
    p.add_argument(
        "--log-interval", type=int, default=10,
        help="SB3 log_interval passed to model.learn (default: 10)",
    )
    p.add_argument(
        "--log-dir", type=str, default=None,
        help="TensorBoard/CSV log directory (auto-generated if omitted)",
    )
    p.add_argument(
        "--models-dir", type=str, default=None,
        help="Directory to save model checkpoints (auto-generated if omitted)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .zip model file to resume training or evaluate from",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if GPU is available",
    )
    p.add_argument(
        "--verbose", type=int, default=1,
        help="SB3 verbosity (0=silent, 1=info, 2=debug)",
    )
    p.add_argument(
        "--net-arch", type=int, nargs="+", default=None,
        help="MLP hidden layer sizes, e.g. --net-arch 400 300",
    )
    p.add_argument(
        "--mode", type=str, choices=["train", "evaluate"], default="train",
        help="train or evaluate (default: train)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(cfg: dict, seed: int) -> ACCElegantEnvironment:
    """Build ACCElegantEnvironment from the shared config dict."""
    platform_cfg = cfg.get("platform", {})
    sim_cfg = cfg.get("simulation", {})
    env_cfg = cfg.get("environment", {})
    log_cfg = cfg.get("logging", {})

    def sub(s):
        return s.replace("{seed}", str(seed)) if isinstance(s, str) else s

    results_path = sub(log_cfg.get("results_path", f"results_sb_{seed}/"))
    if not results_path.endswith("/"):
        results_path += "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    logger_file = sub(log_cfg.get("logger_file_name", f"sb_train_{seed}.csv"))
    headers = tuple(
        log_cfg.get(
            "headers",
            [
                "reward",
                "initial_number_of_particles",
                "number_of_particles",
                "done",
                "itteration",
                "current_element",
                "dict_vars",
            ],
        )
    )
    env_logger, file_handler = setLogger(False, logger_file, headers)

    env = ACCElegantEnvironment(
        stage=env_cfg.get("stage", None),
        n_bins=env_cfg.get("n_bins", 5),
        init_num_particles=env_cfg.get("init_num_particles", 1000),
        logger=env_logger,
        file_handler=file_handler,
        reset_specific_keys_bool=env_cfg.get("reset_specific_keys_bool", False),
        input_beamline_file=sim_cfg.get("input_beamline_file", "machine.lte"),
        beamline_name=sim_cfg.get("beamline_name", "machine"),
        output_beamline_file=sub(
            sim_cfg.get("output_beamline_file", "updated_machine.lte")
        ),
        input_beam_file=sim_cfg.get("input_beam_file", "track"),
        elegant_input_filename=sub(
            sim_cfg.get("elegant_input_filename", "elegant_input.lte")
        ),
        elegant_path=platform_cfg.get("elegant_path", ""),
        sddsPath=platform_cfg.get("sdds_path", ""),
        override_dynamic_command=sim_cfg.get("override_dynamic_command", False),
        overridden_command=sim_cfg.get("overridden_command", "srun -n 2 Pelegant"),
        results_path=results_path,
        seed=seed,
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Config --------------------------------------------------------------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error(f"Config file not found: {cfg_path}")
        sys.exit(1)

    manager = ConfigManager()
    cfg = manager.load(str(cfg_path))

    train_cfg = cfg.get("training", {})
    agent_cfg = cfg.get("agent", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    set_seed(seed)
    log.info(f"Seed: {seed}")

    # --- Device (SB3 does not support MPS; fall back to CPU) -----------------
    if args.cpu or not torch.cuda.is_available():
        sb3_device = "cpu"
    else:
        sb3_device = "cuda"
    log.info(f"SB3 device: {sb3_device}")

    # --- Directories ---------------------------------------------------------
    algo_tag = args.algo.lower()
    log_dir = args.log_dir or f"sb_logs/{algo_tag}_{seed}"
    models_dir = args.models_dir or f"sb_models/{algo_tag}_{seed}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # --- Environment ---------------------------------------------------------
    log.info("Setting up environment ...")
    env = make_env(cfg, seed)
    log.info(f"Observation space : {env.observation_space}")
    log.info(f"Action space      : {env.action_space}")

    # --- Hyperparameters (CLI > config > defaults) ---------------------------
    lr = args.learning_rate or agent_cfg.get("alpha", 1e-4)
    batch_size = args.batch_size or agent_cfg.get("batch_size", 128)
    gamma = args.gamma if args.gamma is not None else agent_cfg.get("gamma", 0.99)
    tau = args.tau if args.tau is not None else agent_cfg.get("tau", 0.005)
    buffer_size = args.buffer_size or agent_cfg.get("max_size", 1_000_000)

    algo_cls = ALGO_MAP[args.algo]

    # --- Action noise (DDPG / TD3 only) --------------------------------------
    action_noise = None
    if args.algo in ("DDPG", "TD3"):
        noise_type = args.noise_type or agent_cfg.get("noise_type", "gaussian")
        n_actions = env.action_space.shape[-1]
        if noise_type == "gaussian":
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=args.noise_sigma * np.ones(n_actions),
            )
        elif noise_type == "ou":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=args.noise_sigma * np.ones(n_actions),
            )
        # noise_type == "none" → action_noise stays None

    # --- Policy kwargs (custom network architecture) -------------------------
    policy_kwargs = {}
    if args.net_arch:
        policy_kwargs["net_arch"] = args.net_arch

    # --- Build or resume model -----------------------------------------------
    if args.resume:
        log.info(f"Loading model from {args.resume}")
        model = algo_cls.load(args.resume, env=env, device=sb3_device)
    else:
        model_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            verbose=args.verbose,
            seed=seed,
            device=sb3_device,
        )
        if policy_kwargs:
            model_kwargs["policy_kwargs"] = policy_kwargs
        if args.algo in ("DDPG", "TD3") and action_noise is not None:
            model_kwargs["action_noise"] = action_noise

        model = algo_cls(**model_kwargs)

    # --- SB3 logger -----------------------------------------------------------
    sb3_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(sb3_logger)

    # --- Summary --------------------------------------------------------------
    log.info("=" * 70)
    log.info(f"Stable-Baselines3 — {args.algo}")
    log.info(f"  Total timesteps : {args.total_timesteps}")
    log.info(f"  Learning rate   : {lr}")
    log.info(f"  Batch size      : {batch_size}")
    log.info(f"  Gamma           : {gamma}")
    log.info(f"  Tau             : {tau}")
    log.info(f"  Buffer size     : {buffer_size}")
    if args.algo in ("DDPG", "TD3"):
        log.info(
            f"  Noise           : {args.noise_type or agent_cfg.get('noise_type', 'gaussian')} "
            f"(sigma={args.noise_sigma})"
        )
    log.info(f"  Net arch        : {args.net_arch or 'SB3 default [400, 300]'}")
    log.info(f"  Log directory   : {log_dir}")
    log.info(f"  Models directory: {models_dir}")
    if args.resume:
        log.info(f"  Resumed from    : {args.resume}")
    log.info("=" * 70)

    # --- Callbacks ------------------------------------------------------------
    callbacks = [
        ParticleLoggingCallback(),
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=models_dir,
            name_prefix=f"{algo_tag}_ckpt",
        ),
    ]

    # --- Train or evaluate ----------------------------------------------------
    if args.mode == "train":
        log.info("Starting training ...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
        )

        final_path = os.path.join(models_dir, f"{algo_tag}_final")
        model.save(final_path)
        log.info(f"Final model saved to {final_path}.zip")

        log.info(f"Running {args.eval_episodes} evaluation episode(s) ...")
        _run_eval(env, model, args.eval_episodes)

    elif args.mode == "evaluate":
        if not args.resume:
            log.error("--resume is required in evaluate mode.")
            sys.exit(1)
        log.info(f"Evaluating for {args.eval_episodes} episode(s) ...")
        _run_eval(env, model, args.eval_episodes)

    log.info("Done.")


def _run_eval(env, model, n_episodes: int):
    """Deterministic rollout and report."""
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        particles = info.get("number_of_particles", 0)
        log.info(
            f"  Episode {ep + 1}: reward={total_reward:.4f}, "
            f"particles={particles}, steps={step}"
        )


if __name__ == "__main__":
    main()
