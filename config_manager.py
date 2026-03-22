"""
Configuration Manager for RL Training
Handles loading and merging configurations from YAML and command-line arguments
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


class ConfigManager:
    """Manages configuration loading from YAML files and command-line overrides"""

    def __init__(self):
        self.config: Dict[str, Any] = {}

    def load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if yaml is None:
            raise ImportError(
                "PyYAML is required. Install with: pip install pyyaml"
            )

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config if config else {}

    def load_json(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def load(self, config_path: str) -> Dict[str, Any]:
        """Auto-detect file type and load configuration"""
        config_file = Path(config_path)

        if config_file.suffix.lower() in [".yaml", ".yml"]:
            self.config = self.load_yaml(config_path)
        elif config_file.suffix.lower() == ".json":
            self.config = self.load_json(config_path)
        else:
            raise ValueError(
                f"Unsupported config format: {config_file.suffix}. Use .yaml, .yml, or .json"
            )

        return self.config

    def merge_cli_args(self, cli_args: Dict[str, Any]) -> None:
        """Merge command-line arguments into config (CLI args override config file)"""
        for key, value in cli_args.items():
            if value is not None:
                self._set_nested_value(self.config, key, value)

    def _set_nested_value(self, config: Dict, key_path: str, value: Any) -> None:
        """Set a nested value in config using dot notation (e.g., 'agent.alpha')"""
        keys = key_path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value from config using dot notation"""
        keys = key_path.split(".")
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to top-level keys"""
        return self.config.get(key, {})

    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary"""
        return self.config

    def __repr__(self) -> str:
        return json.dumps(self.config, indent=2, default=str)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser with all RLConfig parameters"""
    parser = argparse.ArgumentParser(
        description="RL Training Script for ACC Elegant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python train.py

  # Run with custom config file
  python train.py --config my_config.yaml

  # Override specific parameters
  python train.py --config config.yaml --training.n_episodes 100 --training.cpu true

  # JSON config file
  python train.py --config config.json

  # Multiple parameter overrides
  python train.py \\
    --config config.yaml \\
    --training.n_episodes 10000 \\
    --training.seed 42 \\
    --agent.alpha 0.0002 \\
    --logging.results_path ./results_v2/
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (YAML or JSON). Default: config.yaml",
    )

    # Platform/Simulation Arguments
    parser.add_argument(
        "--platform.elegant_path",
        type=str,
        help="Path to elegant executable",
    )
    parser.add_argument(
        "--simulation.override_dynamic_command",
        type=lambda x: x.lower() == "true",
        help="Override dynamic command creation (true/false)",
    )
    parser.add_argument(
        "--simulation.overridden_command",
        type=str,
        help="Command to run simulations (e.g., 'srun -n 2 Pelegant')",
    )
    parser.add_argument(
        "--simulation.input_beamline_file",
        type=str,
        help="Input beamline .lte file",
    )
    parser.add_argument(
        "--simulation.output_beamline_file",
        type=str,
        help="Output beamline .lte file",
    )

    # Environment Arguments
    parser.add_argument(
        "--environment.n_bins",
        type=int,
        help="Number of bins for 2D histogram state representation",
    )
    parser.add_argument(
        "--environment.init_num_particles",
        type=int,
        help="Initial number of particles for simulation",
    )
    parser.add_argument(
        "--environment.reset_specific_keys_bool",
        type=lambda x: x.lower() == "true",
        help="Reset Hkick, Vkick, and FSE (true/false)",
    )

    # Training Arguments
    parser.add_argument(
        "--training.n_episodes",
        type=int,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--training.max_steps",
        type=int,
        help="Maximum steps per episode (None = dynamic)",
    )
    parser.add_argument(
        "--training.seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--training.cpu",
        type=lambda x: x.lower() == "true",
        help="Force CPU usage (true/false)",
    )
    parser.add_argument(
        "--training.load_model",
        type=lambda x: x.lower() == "true",
        help="Load pre-trained model weights (true/false)",
    )
    parser.add_argument(
        "--training.greedy",
        type=float,
        help="Epsilon-greedy exploration rate",
    )

    # Agent Arguments
    parser.add_argument(
        "--agent.alpha",
        type=float,
        help="Actor learning rate",
    )
    parser.add_argument(
        "--agent.beta",
        type=float,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--agent.batch_size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--agent.gamma",
        type=float,
        help="Discount factor",
    )
    parser.add_argument(
        "--agent.tau",
        type=float,
        help="Soft update parameter",
    )
    parser.add_argument(
        "--agent.max_size",
        type=int,
        help="Maximum replay buffer size",
    )
    parser.add_argument(
        "--agent.noise_type",
        type=str,
        choices=["gaussian", "ou"],
        help="Type of exploration noise",
    )

    # Buffer Arguments
    parser.add_argument(
        "--buffer.load_buffer_bool",
        type=lambda x: x.lower() == "true",
        help="Load previous replay buffer (true/false)",
    )
    parser.add_argument(
        "--buffer.load_buffer_filepath",
        type=str,
        help="Path to load replay buffer from",
    )
    parser.add_argument(
        "--buffer.save_buffer_filepath",
        type=str,
        help="Path to save replay buffer to",
    )

    # Logging Arguments
    parser.add_argument(
        "--logging.results_path",
        type=str,
        help="Path for results output",
    )
    parser.add_argument(
        "--logging.tb_file_name",
        type=str,
        help="TensorBoard log file name",
    )
    parser.add_argument(
        "--logging.logger_file_name",
        type=str,
        help="CSV logger file name",
    )

    # Action
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
        default="train",
        help="Mode: 'train' for training, 'evaluate' for evaluation only",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate (when mode=evaluate)",
    )

    return parser


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = create_arg_parser()
    return parser.parse_args(args)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from file"""
    manager = ConfigManager()
    return manager.load(config_path)


def merge_configs(base_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments into base configuration"""
    # Convert Namespace to dict and filter out None values and non-config keys
    args_dict = vars(cli_args)
    cli_config = {
        k: v for k, v in args_dict.items() if v is not None and k not in ["config", "mode", "eval_episodes"]
    }

    manager = ConfigManager()
    manager.config = base_config
    manager.merge_cli_args(cli_config)
    return manager.config


if __name__ == "__main__":
    # Test the config manager
    try:
        parser = create_arg_parser()
        parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
