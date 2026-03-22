#!/usr/bin/env python
"""
Bayesian Optimisation for ACC Beamline.

Optimizes beamline magnet parameters (K1, HKICK, VKICK, FSE) to maximize
particle transmission using Gaussian-Process-based Bayesian Optimisation.

Two back-ends are supported (pick one via --backend):
  * "bayes_opt"  - bayesian-optimization library  (default)
  * "skopt"      - scikit-optimize (gp_minimize with Expected Improvement)

Usage:
    python bayesian_optimization.py
    python bayesian_optimization.py --config config.json --n-calls 500 --seed 42
    python bayesian_optimization.py --backend skopt --n-calls 200
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

from rl_framework.Elegant import ElegantWrapper
from rl_framework.Utils import create_seeded_tracking_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("bayesopt")
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
# Objective helper with CSV logging
# ---------------------------------------------------------------------------
def make_objective(elegant: ElegantWrapper, init_num_particles: int,
                   csv_writer, csv_file):
    """Return a closure that returns **negative** reward (for minimisers).

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

        if call_count % 25 == 0:
            logger.info(f"  eval #{call_count}: reward = {reward}  "
                        f"(best so far: {best_reward})")

        return -reward

    objective.get_best = lambda: (best_reward, best_values)
    return objective


# ---------------------------------------------------------------------------
# Backend: bayesian-optimization (bayes_opt)
# ---------------------------------------------------------------------------
def run_bayes_opt(objective_fn, bounds, n_calls, n_init, seed):
    from bayes_opt import BayesianOptimization

    n_params = len(bounds)
    pbounds = {f"x{i}": (lo, hi) for i, (lo, hi) in enumerate(bounds)}

    def bo_target(**params):
        x = [params[f"x{i}"] for i in range(n_params)]
        return -objective_fn(x)  # bayes_opt *maximises*, negate our neg-reward

    optimizer = BayesianOptimization(
        f=bo_target,
        pbounds=pbounds,
        random_state=seed,
        verbose=2,
    )

    optimizer.maximize(init_points=n_init, n_iter=n_calls - n_init)

    best_params = [optimizer.max["params"][f"x{i}"] for i in range(n_params)]
    best_score = -optimizer.max["target"]

    history = []
    for i, res in enumerate(optimizer.res):
        history.append({
            "iteration": i + 1,
            "target": res["target"],
            "params": [res["params"][f"x{j}"] for j in range(n_params)],
        })

    return {
        "best_x": np.array(best_params),
        "best_fun": best_score,
        "best_reward": -best_score,
        "history": history,
        "n_calls": n_calls,
        "seed": seed,
        "backend": "bayes_opt",
    }


# ---------------------------------------------------------------------------
# Backend: scikit-optimize (skopt)
# ---------------------------------------------------------------------------
def run_skopt(objective_fn, bounds, n_calls, n_init, seed):
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    n_params = len(bounds)
    skopt_space = [Real(lo, hi, name=f"x{i}") for i, (lo, hi) in enumerate(bounds)]

    @use_named_args(skopt_space)
    def skopt_objective(**params):
        x = [params[f"x{i}"] for i in range(n_params)]
        return objective_fn(x)

    gp_result = gp_minimize(
        skopt_objective,
        skopt_space,
        n_calls=n_calls,
        n_initial_points=n_init,
        acq_func="EI",
        random_state=seed,
        verbose=True,
    )

    history = []
    for i, (yi, xi) in enumerate(zip(gp_result.func_vals, gp_result.x_iters)):
        history.append({"iteration": i + 1, "fun": float(yi), "params": list(xi)})

    return {
        "best_x": np.array(gp_result.x),
        "best_fun": gp_result.fun,
        "best_reward": -gp_result.fun,
        "history": history,
        "n_calls": n_calls,
        "seed": seed,
        "backend": "skopt",
        "gp_result": gp_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Bayesian Optimisation for ACC beamline")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "config.json"),
                   help="Path to JSON config file (default: ../config.json)")
    p.add_argument("--backend", type=str, default="bayes_opt",
                   choices=["bayes_opt", "skopt"],
                   help="BO library to use (default: bayes_opt)")
    p.add_argument("--n-calls", type=int, default=500,
                   help="Total number of objective evaluations (default: 500)")
    p.add_argument("--n-init", type=int, default=10,
                   help="Random initial evaluations (default: 10)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (overrides config)")
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
    results_path = args.results_dir or f"results_bayesopt_{seed}/"
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
    N_PARAMS = len(BOUNDS)
    logger.info(f"Variables ({N_PARAMS}): {variables}")

    # --- CSV log (saved OUTSIDE results_path to survive sim cleanup) -------
    csv_path = f"bayesopt_log_{seed}.csv"
    csv_fields = ["eval", "reward", "particles", "best_reward",
                  "best_particles", "init_particles", "success", "dict_vars"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    objective = make_objective(elegant, init_num_particles, csv_writer, csv_file)

    # --- Run optimisation --------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"Bayesian Optimisation - {args.backend}")
    logger.info(f"  Parameters : {N_PARAMS}")
    logger.info(f"  Total calls: {args.n_calls}")
    logger.info(f"  Init points: {args.n_init}")
    logger.info(f"  Seed       : {seed}")
    logger.info(f"  Init parts : {init_num_particles}")
    logger.info(f"  Results    : {results_path}")
    logger.info("=" * 70)

    t0 = time.time()

    if args.backend == "bayes_opt":
        result_data = run_bayes_opt(objective, BOUNDS, args.n_calls,
                                    args.n_init, seed)
    else:
        result_data = run_skopt(objective, BOUNDS, args.n_calls,
                                args.n_init, seed)

    elapsed = time.time() - t0
    result_data["elapsed_s"] = elapsed

    # --- Report results ----------------------------------------------------
    best_reward, best_vars = objective.get_best()

    logger.info("=" * 70)
    logger.info("OPTIMISATION COMPLETE")
    logger.info(f"  Backend    : {args.backend}")
    logger.info(f"  Elapsed    : {elapsed:.1f}s")
    logger.info(f"  Best score : {result_data['best_fun']:.4f}  "
                f"(particles ~= {result_data['best_reward']:.0f})")
    logger.info(f"  Best reward: {best_reward}")
    logger.info("=" * 70)

    if best_vars:
        logger.info("Best parameter values found:")
        for k, v in best_vars.items():
            logger.info(f"  {k:20s} = {v}")
        logger.info("=" * 70)

    # Verify best solution (this clears results_path, so save files after)
    logger.info("Verifying best solution ...")
    best_x = result_data["best_x"]
    _, success, dict_vars = elegant.run_elegant_simulation(best_x)
    reward, _ = elegant.get_results_for_Scipy(init_num_particles)
    logger.info(f"  Verification reward: {reward}  (success={success})")
    if dict_vars:
        logger.info(f"  Variables: {dict_vars}")

    # Save AFTER verification (results_path is cleaned each sim call)
    save_data = {k: v for k, v in result_data.items() if k != "gp_result"}
    save_data["best_tracked_reward"] = best_reward
    save_data["best_tracked_vars"] = best_vars
    save_data["verification_reward"] = reward
    final_path = f"bayesopt_result_{seed}.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(save_data, f)
    logger.info(f"Result saved to {final_path}")
    logger.info(f"CSV log saved to {csv_path}")

    if "gp_result" in result_data:
        try:
            from skopt import dump as skopt_dump
            skopt_path = f"skopt_result_{seed}.pkl"
            skopt_dump(result_data["gp_result"], skopt_path)
            logger.info(f"Full skopt result saved to {skopt_path}")
        except Exception as e:
            logger.warning(f"Could not save raw skopt result (non-critical): {e}")

    csv_file.close()


if __name__ == "__main__":
    main()
