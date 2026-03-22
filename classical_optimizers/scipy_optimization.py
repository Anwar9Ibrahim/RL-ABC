#!/usr/bin/env python
"""
SciPy Differential Evolution Optimizer for ACC Beamline.

Optimizes beamline magnet parameters (K1, HKICK, VKICK, FSE) to maximize
particle transmission using SciPy's differential_evolution algorithm.

Usage:
    python scipy_optimization.py
    python scipy_optimization.py --config config.json --maxiter 2000 --seed 0
    python scipy_optimization.py --resume  # resume from checkpoint
"""

import os
import sys
import argparse
import csv
import json
import pickle
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from rl_framework.Elegant import ElegantWrapper
from rl_framework.Utils import create_seeded_tracking_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("scipy_opt")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Build action-space bounds dynamically from wrapper variable names
# ---------------------------------------------------------------------------
def build_bounds(variables):
    """Build (low, high) bounds for each variable based on its name.

    Same logic as ACCElegantEnvironment._set_action_space so the classical
    optimisers and RL agent operate in identical parameter spaces.
    """
    low, high = [], []
    for var in variables:
        if "K1" in var:
            low.append(-20.0)
            high.append(20.0)
        elif any(x in var for x in ("VKICK", "HKICK", "FSE")):
            low.append(-0.005)
            high.append(0.005)
        else:
            low.append(-1.0)
            high.append(1.0)
    return list(zip(low, high))


# ---------------------------------------------------------------------------
# Checkpoint manager (callback for differential_evolution)
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Saves optimisation state on every new-best or at regular intervals."""

    def __init__(self, checkpoint_path="scipy_checkpoint.pkl", interval=100):
        self.checkpoint_path = checkpoint_path
        self.interval = interval
        self.iteration = 0
        self.best_fun = np.inf
        self.history = []

    def __call__(self, intermediate_result: OptimizeResult):
        self.iteration += 1
        current_fun = intermediate_result.fun
        is_new_best = current_fun < self.best_fun
        should_save = is_new_best or (self.iteration % self.interval == 0)

        self.history.append({
            "iteration": self.iteration,
            "fun": current_fun,
            "best_fun": min(current_fun, self.best_fun),
        })

        if should_save:
            if is_new_best:
                self.best_fun = current_fun

            state = {
                "iteration": self.iteration,
                "best_x": intermediate_result.x,
                "best_fun": current_fun,
                "population": getattr(intermediate_result, "population", None),
                "population_energies": getattr(intermediate_result, "population_energies", None),
                "history": self.history,
            }
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(state, f)

            tag = "New Best" if is_new_best else f"Interval {self.iteration}"
            logger.info(f"[{tag}] iter {self.iteration}: fun = {current_fun:.4f}")


# ---------------------------------------------------------------------------
# Objective function with CSV logging
# ---------------------------------------------------------------------------
def make_objective(elegant: ElegantWrapper, init_num_particles: int,
                   csv_writer, csv_file):
    """Return a closure that differential_evolution will *minimise*.

    Every evaluation is recorded to *csv_writer* so results are never lost.
    """
    call_count = 0
    best_reward = 0
    best_values = None

    def objective(values):
        nonlocal call_count, best_reward, best_values
        call_count += 1

        _, success, dict_vars = elegant.run_elegant_simulation(values)
        if not success:
            csv_writer.writerow({
                "eval": call_count, "reward": 0,
                "particles": 0, "best_reward": best_reward,
                "best_particles": best_reward,
                "init_particles": init_num_particles,
                "success": False, "dict_vars": "",
            })
            csv_file.flush()
            return 1e6

        reward, _ = elegant.get_results_for_Scipy(init_num_particles)
        if reward is None or reward == 0:
            csv_writer.writerow({
                "eval": call_count, "reward": 0,
                "particles": 0, "best_reward": best_reward,
                "best_particles": best_reward,
                "init_particles": init_num_particles,
                "success": True, "dict_vars": dict_vars,
            })
            csv_file.flush()
            return 1e6

        if reward > best_reward:
            best_reward = reward
            best_values = dict_vars
            logger.info(f"  New best at eval #{call_count}: "
                        f"reward = {reward} particles")

        csv_writer.writerow({
            "eval": call_count, "reward": reward,
            "particles": reward, "best_reward": best_reward,
            "best_particles": best_reward,
            "init_particles": init_num_particles,
            "success": True, "dict_vars": dict_vars,
        })
        csv_file.flush()

        if call_count % 50 == 0:
            logger.info(f"  eval #{call_count}: reward = {reward}  "
                        f"(best so far: {best_reward})")

        return -reward

    objective.get_best = lambda: (best_reward, best_values)
    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SciPy differential-evolution beamline optimiser")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "config.json"),
                   help="Path to JSON config file (default: ../config.json)")
    p.add_argument("--maxiter", type=int, default=2000, help="Maximum DE generations")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for DE")
    p.add_argument("--checkpoint", type=str, default="scipy_checkpoint.pkl",
                   help="Checkpoint file path")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Save checkpoint every N iterations")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint population")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Results directory (overrides config)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    platform_cfg = cfg.get("platform", {})
    sim_cfg = cfg.get("simulation", {})
    env_cfg = cfg.get("environment", {})
    train_cfg = cfg.get("training", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    np.random.seed(seed)

    def sub(s):
        return s.replace("{seed}", str(seed)) if isinstance(s, str) else s

    init_num_particles = env_cfg.get("init_num_particles", 1000)
    results_path = args.results_dir or f"results_scipy_{seed}/"
    if not results_path.endswith("/"):
        results_path += "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # --- Resolve config values with seed substitution ----------------------
    output_beamline_file = sub(sim_cfg.get("output_beamline_file", "updated_machine.lte"))
    elegant_input_filename = sub(sim_cfg.get("elegant_input_filename", "elegant_input.lte"))
    input_beam_file = sim_cfg.get("input_beam_file", "track")

    # --- Create seeded tracking .ele file (same as RL training) ------------
    seeded_beam_file = create_seeded_tracking_file(
        input_beam_file,
        elegant_input_filename,
        results_path,
        seed,
    )

    # --- Setup ElegantWrapper ----------------------------------------------
    elegant = ElegantWrapper(
        input_beamline_file=sim_cfg.get("input_beamline_file", "machine.lte"),
        input_beam_file=seeded_beam_file,
        beamline_name=sim_cfg.get("beamline_name", "machine"),
        output_beamline_file=output_beamline_file,
        elegant_path=platform_cfg.get("elegant_path", ""),
        sddsPath=platform_cfg.get("sdds_path", ""),
        results_path=results_path,
        elegant_input_filename=elegant_input_filename,
        overrid_dynmaic_commnad=sim_cfg.get("override_dynamic_command", False),
        overrideen_command=sim_cfg.get("overridden_command", "srun -n 2 Pelegant"),
        seed=seed,
    )
    elegant.reset_specific_keys_bool = env_cfg.get("reset_specific_keys_bool", False)

    # --- Build bounds from actual wrapper variables ------------------------
    variables = elegant.chroneological_variables
    BOUNDS = build_bounds(variables)
    logger.info(f"Variables ({len(variables)}): {variables}")

    # --- CSV log (saved OUTSIDE results_path to survive sim cleanup) -------
    csv_path = f"scipy_log_{seed}.csv"
    csv_fields = ["eval", "reward", "particles", "best_reward",
                  "best_particles", "init_particles", "success", "dict_vars"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    objective = make_objective(elegant, init_num_particles, csv_writer, csv_file)

    # --- Checkpoint / resume -----------------------------------------------
    manager = CheckpointManager(
        checkpoint_path=args.checkpoint,
        interval=args.checkpoint_interval,
    )

    init_population = "latinhypercube"
    if args.resume and os.path.exists(args.checkpoint):
        with open(args.checkpoint, "rb") as f:
            state = pickle.load(f)
        pop = state.get("population")
        if pop is not None:
            init_population = pop
            logger.info(f"Resuming from checkpoint (iter {state['iteration']}, "
                        f"best fun {state['best_fun']:.4f})")
        else:
            logger.warning("Checkpoint has no population; starting fresh.")
            init_population = "latinhypercube"

    # --- Run optimisation --------------------------------------------------
    logger.info("=" * 70)
    logger.info("SciPy Differential Evolution - Beamline Optimisation")
    logger.info(f"  Parameters : {len(BOUNDS)}")
    logger.info(f"  Max iters  : {args.maxiter}")
    logger.info(f"  Seed       : {seed}")
    logger.info(f"  Init parts : {init_num_particles}")
    logger.info(f"  Results    : {results_path}")
    logger.info("=" * 70)

    t0 = time.time()
    result = differential_evolution(
        objective,
        BOUNDS,
        maxiter=args.maxiter,
        workers=args.workers,
        polish=False,
        callback=manager,
        seed=seed,
        init=init_population,
    )
    elapsed = time.time() - t0

    # --- Report results ----------------------------------------------------
    best_reward, best_vars = objective.get_best()

    logger.info("=" * 70)
    logger.info("OPTIMISATION COMPLETE")
    logger.info(f"  Elapsed    : {elapsed:.1f}s")
    logger.info(f"  Iterations : {result.nit}")
    logger.info(f"  Func evals : {result.nfev}")
    logger.info(f"  Best score : {result.fun:.4f}  (particles ~= {-result.fun:.0f})")
    logger.info(f"  Best reward: {best_reward}")
    logger.info(f"  Success    : {result.success}")
    logger.info(f"  Message    : {result.message}")
    logger.info("=" * 70)

    if best_vars:
        logger.info("Best parameter values found:")
        for k, v in best_vars.items():
            logger.info(f"  {k:20s} = {v}")
        logger.info("=" * 70)

    # Run once more with the best parameters to verify
    logger.info("Verifying best solution ...")
    _, success, dict_vars = elegant.run_elegant_simulation(result.x)
    reward, _ = elegant.get_results_for_Scipy(init_num_particles)
    logger.info(f"  Verification reward: {reward}  (success={success})")
    if dict_vars:
        logger.info(f"  Variables: {dict_vars}")

    # Save final result AFTER verification (results_path is cleaned each sim)
    final_path = f"scipy_result_{seed}.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({
            "result_x": result.x,
            "result_fun": result.fun,
            "nit": result.nit,
            "nfev": result.nfev,
            "population": getattr(result, "population", None),
            "history": manager.history,
            "elapsed_s": elapsed,
            "seed": seed,
            "best_reward": best_reward,
            "best_vars": best_vars,
            "verification_reward": reward,
        }, f)
    logger.info(f"Final result saved to {final_path}")
    logger.info(f"CSV log saved to {csv_path}")

    csv_file.close()


if __name__ == "__main__":
    main()
