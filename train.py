#!/usr/bin/env python
"""
RL Training Script for ACC Elegant Simulations

This script trains a DDPG agent to optimize ACC beamline parameters.
It supports configuration via YAML/JSON files and command-line overrides.

Usage:
    # Basic usage with default config.yaml
    python train.py

    # With custom config file
    python train.py --config my_config.yaml

    # Override specific parameters
    python train.py --training.n_episodes 100 --training.cpu true

    # Evaluate only
    python train.py --mode evaluate --eval-episodes 5

For more examples:
    python train.py --help
"""

import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

# Set temp directory before any torch imports
tempfile.tempdir = '/tmp'

from config_manager import ConfigManager, parse_args, load_config, merge_configs
from rl_framework.Utils import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_environment_info():
    """Print information about the current Python environment"""
    conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')
    conda_env_path = os.environ.get('CONDA_PREFIX')

    logger.info("=" * 80)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 80)
    
    if conda_env_name:
        logger.info(f"Conda Environment: {conda_env_name}")
    if conda_env_path:
        logger.info(f"Conda Path: {conda_env_path}")
    
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("=" * 80)


class RLConfig:
    """Configuration class for RL training"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary
        
        Args:
            config_dict: Dictionary with configuration values
        """
        # Store raw config
        self._config = config_dict

        # Platform and Path configuration
        platform_cfg = config_dict.get('platform', {})
        self.os_type = platform_cfg.get('os_type', 'darwin').lower()
        self.elegant_path = platform_cfg.get('elegant_path', '')
        self.sdds_path = platform_cfg.get('sdds_path', '')

        # Training Configuration (must be loaded first to get seed)
        train_cfg = config_dict.get('training', {})
        self.seed = train_cfg.get('seed', 0)
        self.cpu = train_cfg.get('cpu', False)
        self.load_model = train_cfg.get('load_model', True)
        self.n_episodes = train_cfg.get('n_episodes', 10)
        self.max_steps = train_cfg.get('max_steps', None)
        self.greedy = train_cfg.get('greedy', 0.05)

        # Substitute {seed} placeholders in all configuration values
        self._substitute_seed_placeholders()

        # Simulation Configuration
        sim_cfg = config_dict.get('simulation', {})
        self.override_dynamic_command = sim_cfg.get('override_dynamic_command', False)
        self.overridden_command = sim_cfg.get('overridden_command', 'srun -n 2 Pelegant')
        self.input_beamline_file = sim_cfg.get('input_beamline_file', 'machine.lte')
        self.input_beam_file = sim_cfg.get('input_beam_file', 'track')
        self.output_beamline_file = sim_cfg.get('output_beamline_file', 'updated_machine.lte')
        self.beamline_name = sim_cfg.get('beamline_name', 'machine')
        self.elegant_input_filename = sim_cfg.get('elegant_input_filename', 'elegant_input.lte')

        # Environment Configuration
        env_cfg = config_dict.get('environment', {})
        self.stage = env_cfg.get('stage', None)
        self.n_bins = env_cfg.get('n_bins', 5)
        self.reset_specific_keys_bool = env_cfg.get('reset_specific_keys_bool', False)
        self.init_num_particles = env_cfg.get('init_num_particles', 1000)

        # Replay Buffer Configuration
        buffer_cfg = config_dict.get('buffer', {})
        self.load_buffer_bool = buffer_cfg.get('load_buffer_bool', True)
        self.load_buffer_filepath = buffer_cfg.get('load_buffer_filepath', 'DDPG_bigbeamline_buffer.pkl')
        self.save_buffer_filepath = buffer_cfg.get('save_buffer_filepath', 'rbBig_beamline_original.pkl')

        # Agent Configuration (DDPG)
        agent_cfg = config_dict.get('agent', {})
        self.agent_type = agent_cfg.get('agent_type', 'DDPG')
        self.alpha = agent_cfg.get('alpha', 1e-4)
        self.beta = agent_cfg.get('beta', 1e-3)
        self.batch_size = agent_cfg.get('batch_size', 128)
        self.gamma = agent_cfg.get('gamma', 0.99)
        self.tau = agent_cfg.get('tau', 0.005)
        self.max_size = agent_cfg.get('max_size', 1000000)
        self.noise_type = agent_cfg.get('noise_type', 'gaussian')
        self.log_interval = agent_cfg.get('log_interval', 100)
        self.eval_interval = agent_cfg.get('eval_interval', 1000)
        self.convert = agent_cfg.get('convert', True)

        # Logging Configuration
        log_cfg = config_dict.get('logging', {})
        self.results_path = log_cfg.get('results_path', 'results/')
        self.results_path = self._ensure_seed_suffix(self.results_path)
        self.tb_file_name = log_cfg.get('tb_file_name', 'big_beamline_original')
        self.logger_file_name = log_cfg.get('logger_file_name', 'big_beamline_original.csv')
        self.save_model_file_name = log_cfg.get('save_model_file_name', '')
        self.headers = tuple(log_cfg.get('headers', [
            'reward', 'initial_number_of_particles', 'number_of_particles',
            'done', 'itteration', 'current_element', 'dict_vars'
        ]))

        # Device configuration
        self.device = self._setup_device()

    def _ensure_seed_suffix(self, path_value: str) -> str:
        """Ensure directory-like path values are seed-specific."""
        normalized = str(path_value)
        has_trailing_slash = normalized.endswith(("/", "\\"))
        stripped = normalized.rstrip("/\\")
        if not stripped:
            stripped = "results"
        if not stripped.endswith(f"_{self.seed}"):
            stripped = f"{stripped}_{self.seed}"
        return f"{stripped}/" if has_trailing_slash else stripped

    def _substitute_seed_placeholders(self):
        """
        Recursively substitute {seed} placeholders in all configuration values.
        """
        def substitute_in_value(value):
            if isinstance(value, str):
                return value.replace("{seed}", str(self.seed))
            return value
        
        def substitute_in_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    substitute_in_dict(value)
                elif isinstance(value, str):
                    d[key] = substitute_in_value(value)
        
        substitute_in_dict(self._config)

    def _setup_device(self) -> torch.device:
        """Setup computation device based on availability"""
        if not self.cpu and torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return torch.device("cuda:0")
        elif not self.cpu and torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders)")
            return torch.device("mps")
        else:
            logger.info("Using CPU")
            return torch.device("cpu")

    def setup_environment(self):
        """Setup and return the environment"""
        from rl_framework.Environment import ACCElegantEnvironment
        from rl_framework.Utils import setLogger

        # Set up loggers first
        logger_obj, file_handler = setLogger(
            self.load_model,
            self.logger_file_name,
            self.headers
        )

        env = ACCElegantEnvironment(
            stage=self.stage,
            n_bins=self.n_bins,
            init_num_particles=self.init_num_particles,
            logger=logger_obj,
            file_handler=file_handler,
            reset_specific_keys_bool=self.reset_specific_keys_bool,
            input_beamline_file=self.input_beamline_file,
            beamline_name=self.beamline_name,
            output_beamline_file=self.output_beamline_file,
            input_beam_file=self.input_beam_file,
            elegant_input_filename=self.elegant_input_filename,
            elegant_path=self.elegant_path,
            sddsPath=self.sdds_path,
            override_dynamic_command=self.override_dynamic_command,
            overridden_command=self.overridden_command,
            results_path=self.results_path,
            seed=self.seed
        )

        # Set max_steps dynamically from environment if not specified
        if self.max_steps is None:
            self.max_steps = env.max_num_of_vars

        return env, logger_obj, file_handler

    def setup_agent(self, env):
        """Setup and return the correct agent"""
        from rl_framework.Agents.DDPG import DDPGAgent

        if self.agent_type == "DDPG":
            agent = DDPGAgent(
                env=env,
                alpha=self.alpha,
                beta=self.beta,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                max_size=self.max_size,
                noise_type=self.noise_type,
                log_interval=self.log_interval,
                eval_interval=self.eval_interval,
                seed=self.seed,
                exp=self.tb_file_name,
                load=self.load_model,
                convert=self.convert
            )
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
        
        return agent

    def load_buffer(self, agent):
        """Load replay buffer if it exists"""
        if os.path.exists(self.load_buffer_filepath):
            agent.buffer.load(self.load_buffer_filepath)
            logger.info(f"Loaded replay buffer from {self.load_buffer_filepath}")
        else:
            logger.warning(f"Buffer file not found: {self.load_buffer_filepath}")

    def save_buffer(self, agent):
        """Save replay buffer"""
        agent.buffer.save(self.save_buffer_filepath)
        logger.info(f"Saved replay buffer to {self.save_buffer_filepath}")

    def load_agent_weights(self, agent, step: Optional[int] = None):
        """Load agent weights from checkpoints"""
        if self.load_model:
            if step:
                agent.load_models(step)
                logger.info(f"Loaded agent weights from step {step}")
            else:
                agent.load_models()
                logger.info("Loaded latest agent weights")
        else:
            logger.info("Training from scratch (not loading pre-trained weights)")

    def train(self, agent, env):
        """Run training with the current configuration"""
        logger.info(f"Starting training for {self.n_episodes} episodes...")
        
        scores = agent.train(
            n_episodes=self.n_episodes,
            max_steps=self.max_steps,
            greedy=self.greedy
        )
        
        logger.info("Training completed")
        return scores

    def evaluate(self, agent, episodes: int = 1):
        """Evaluate the agent"""
        logger.info(f"Evaluating agent for {episodes} episode(s)...")
        agent.evaluate(episodes=episodes)
        logger.info("Evaluation completed")

    def __str__(self) -> str:
        """String representation of configuration"""
        config_str = "=" * 80 + "\n"
        config_str += "RL CONFIGURATION\n"
        config_str += "=" * 80 + "\n"
        config_str += f"  Agent Type: {self.agent_type}\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Training Episodes: {self.n_episodes}\n"
        config_str += f"  Max Steps per Episode: {self.max_steps}\n"
        config_str += f"  Learning Rates: Actor={self.alpha}, Critic={self.beta}\n"
        config_str += f"  Batch Size: {self.batch_size}\n"
        config_str += f"  Replay Buffer Size: {self.max_size}\n"
        config_str += f"  Discount Factor (γ): {self.gamma}\n"
        config_str += f"  Soft Update (τ): {self.tau}\n"
        config_str += f"  Initial Particles: {self.init_num_particles}\n"
        config_str += f"  Results Path: {self.results_path}\n"
        config_str += f"  TensorBoard Log: {self.tb_file_name}\n"
        
        if self.load_model:
            config_str += f"  Load Model: True (warm start)\n"
        else:
            config_str += f"  Load Model: False (train from scratch)\n"
        
        if self.load_buffer_bool:
            config_str += f"  Load Buffer: True (from {self.load_buffer_filepath})\n"
        else:
            config_str += f"  Load Buffer: False (fresh buffer)\n"
        
        config_str += "=" * 80
        
        return config_str


def main():
    """Main training function"""
    # Parse command-line arguments
    args = parse_args()

    # Print environment info
    print_environment_info()

    try:
        # Load configuration file
        logger.info(f"Loading configuration from: {args.config}")
        config_dict = load_config(args.config)

        # Merge CLI arguments (they override config file)
        config_dict = merge_configs(config_dict, args)

        # Create RLConfig instance
        config = RLConfig(config_dict)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        logger.info(f"Reproducibility seed set to: {config.seed}")

        # Print configuration
        logger.info(str(config))

        # Create results directory if it doesn't exist
        Path(config.results_path).mkdir(parents=True, exist_ok=True)

        # Setup environment
        logger.info("Setting up environment...")
        env, env_logger, file_handler = config.setup_environment()
        logger.info("Environment setup completed")

        # Setup agent
        logger.info("Setting up agent...")
        agent = config.setup_agent(env)
        logger.info("Agent setup completed")

        # Execute based on mode
        if args.mode == "train":
            # Load buffer if warm starting
            if config.load_buffer_bool:
                config.load_buffer(agent)

            # Load agent weights if warm starting
            if config.load_model:
                config.load_agent_weights(agent)

            # Train the agent
            scores = config.train(agent, env)

            # Save buffer for future training
            config.save_buffer(agent)

            # Evaluate after training
            config.evaluate(agent, episodes=1)

            logger.info("Training workflow completed successfully!")

        elif args.mode == "evaluate":
            # Load pre-trained weights
            config.load_agent_weights(agent)

            # Evaluate the agent
            config.evaluate(agent, episodes=args.eval_episodes)

            logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
