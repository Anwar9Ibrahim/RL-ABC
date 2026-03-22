# Classical Optimizers

Classical (non-RL) optimization baselines for ACC Elegant simulations.

## Available scripts

- `scipy_optimization.py` - Differential Evolution (global search)
- `bayesian_optimization.py` - Bayesian Optimization (`bayes_opt` or `skopt` backends)

Both scripts read the shared `config.json` and use the same `ElegantWrapper` as RL training.
