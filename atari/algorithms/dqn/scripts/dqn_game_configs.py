"""
Game configurations for DQN training.
Contains all game-specific settings, agents, and hyperparameters.
"""

import torch
from typing import Dict, Any
from dqn.agent import Agent
from dqn.pacman.agent import AgentImproved
from dqn.config.environment_config import EnvironmentConfig
from dqn.config.exploration_config import ExplorationConfig
from dqn.config.image_config import ImageConfig
from dqn.config.learning_config import LearningConfig
from dqn.config.model_config import ModelConfig
from dqn.config.training_config import TrainingConfig
from dqn.config.hyperparameters import HyperParameters


class DQNGameConfig:
    """Configuration for different DQN games."""

    def __init__(
        self,
        name: str,
        game_id: str,
        agent_class,
        environment_config: EnvironmentConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        learning_config: LearningConfig,
        exploration_config: ExplorationConfig,
        image_config: ImageConfig,
    ):
        self.name = name
        self.game_id = game_id
        self.agent_class = agent_class
        self.environment_config = environment_config
        self.model_config = model_config
        self.training_config = training_config
        self.learning_config = learning_config
        self.exploration_config = exploration_config
        self.image_config = image_config


def find_latest_dqn_model_episode(
    game_name: str, model_dir: str = "atari/scripts/dqn/models"
) -> int:
    """Find the latest episode number for the given DQN game by scanning the model directory."""
    import os
    import re

    pattern_map = {
        "pong": r"pong-cnn-(\d+)\.pkl$",
        "pacman": r"pacman-cnn-improved-(\d+)\.pkl$",
    }

    pattern = pattern_map.get(game_name)
    if not pattern:
        return 0

    max_episode = 0
    if not os.path.exists(model_dir):
        return 0

    for fname in os.listdir(model_dir):
        match = re.search(pattern, fname)
        if match:
            try:
                ep = int(match.group(1))
                if ep > max_episode:
                    max_episode = ep
            except Exception:
                continue

    return max_episode


# Game configurations
DQN_GAME_CONFIGS = {
    "pong": DQNGameConfig(
        name="Pong",
        game_id="ALE/Pong-v5",
        agent_class=Agent,
        environment_config=EnvironmentConfig(
            environment="ALE/Pong-v5",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            render_game_window=True,
        ),
        model_config=ModelConfig(
            save_models=True,
            model_path="../../models/dqn/pong/pong-cnn-",
            save_model_interval=10,
            train_model=True,
            load_model_from_file=True,
            load_file_episode=1560,  # Will be overridden by latest model
        ),
        training_config=TrainingConfig(
            batch_size=64,
            max_episode=100000,
            max_step=100000,
            max_memory_len=50000,
            min_memory_len=40000,
        ),
        learning_config=LearningConfig(gamma=0.97, alpha=0.00025),
        exploration_config=ExplorationConfig(
            epsilon_start=1.0, epsilon_decay=0.99, epsilon_minimum=0.05
        ),
        image_config=ImageConfig(target_h=80, target_w=64, crop_top=20),
    ),
    "pacman": DQNGameConfig(
        name="Ms. Pacman",
        game_id="ALE/MsPacman-v5",
        agent_class=AgentImproved,
        environment_config=EnvironmentConfig(
            environment="ALE/MsPacman-v5",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            render_game_window=True,
        ),
        model_config=ModelConfig(
            save_models=True,
            model_path="../../models/dqn/pacman/pacman-cnn-improved-",
            save_model_interval=10,
            train_model=True,
            load_model_from_file=True,
            load_file_episode=2450,  # Will be overridden by latest model
        ),
        training_config=TrainingConfig(
            batch_size=32,  # Smaller batch size for better learning
            max_episode=100000,
            max_step=100000,
            max_memory_len=100000,  # Larger memory for better experience diversity
            min_memory_len=10000,  # Start training earlier
        ),
        learning_config=LearningConfig(
            gamma=0.99,  # Slightly higher gamma for Pacman (longer-term planning)
            alpha=0.0001,  # Slightly lower learning rate for stability
        ),
        exploration_config=ExplorationConfig(
            epsilon_start=1.0,
            epsilon_decay=0.9995,  # Much slower decay for better exploration
            epsilon_minimum=0.01,  # Lower minimum for better exploitation
        ),
        image_config=ImageConfig(target_h=80, target_w=64, crop_top=20),
    ),
}


def get_dqn_game_config(game_name: str) -> DQNGameConfig:
    """Get DQN game configuration by name."""
    if game_name not in DQN_GAME_CONFIGS:
        raise ValueError(
            f"Unknown DQN game: {game_name}. Available games: {list(DQN_GAME_CONFIGS.keys())}"
        )
    return DQN_GAME_CONFIGS[game_name]


def get_dqn_hyperparameters(game_name: str, config: Dict[str, Any]) -> HyperParameters:
    """Get hyperparameters for a DQN game with optional overrides."""
    game_config = get_dqn_game_config(game_name)

    # Create new config objects to avoid modifying the original ones
    from dqn.config.environment_config import EnvironmentConfig
    from dqn.config.model_config import ModelConfig
    from dqn.config.training_config import TrainingConfig
    from dqn.config.learning_config import LearningConfig
    from dqn.config.exploration_config import ExplorationConfig
    from dqn.config.image_config import ImageConfig

    # Environment config
    render = config.get("render")
    if render is None:
        render = game_config.environment_config.render_game_window

    env_config = EnvironmentConfig(
        environment=game_config.environment_config.environment,
        device=game_config.environment_config.device,
        render_game_window=render,
    )

    # Model config
    load_episode = config.get("load_episode")
    if load_episode is None:
        load_episode = find_latest_dqn_model_episode(game_name)

    save_interval = config.get("save_interval")
    if save_interval is None:
        save_interval = game_config.model_config.save_model_interval

    model_config = ModelConfig(
        save_models=game_config.model_config.save_models,
        model_path=game_config.model_config.model_path,
        save_model_interval=save_interval,
        train_model=game_config.model_config.train_model,
        load_model_from_file=not config.get("no_load_network", False)
        and game_config.model_config.load_model_from_file,
        load_file_episode=load_episode,
    )

    # Training config
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = game_config.training_config.batch_size

    max_memory_len = config.get("memory_size")
    min_memory_len = game_config.training_config.min_memory_len
    if max_memory_len is None:
        max_memory_len = game_config.training_config.max_memory_len
    else:
        # Keep the training threshold below the buffer capacity
        min_memory_len = min(min_memory_len, max_memory_len // 2)

    training_config = TrainingConfig(
        batch_size=batch_size,
        max_episode=game_config.training_config.max_episode,
        max_step=game_config.training_config.max_step,
        max_memory_len=max_memory_len,
        min_memory_len=min_memory_len,
    )

    # Learning config
    alpha_value = config.get("learning_rate")
    if alpha_value is None:
        alpha_value = game_config.learning_config.alpha

    learning_config = LearningConfig(
        gamma=game_config.learning_config.gamma, alpha=alpha_value
    )

    # Exploration config
    epsilon_decay = config.get("epsilon_decay")
    if epsilon_decay is None:
        epsilon_decay = game_config.exploration_config.epsilon_decay

    exploration_config = ExplorationConfig(
        epsilon_start=game_config.exploration_config.epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_minimum=game_config.exploration_config.epsilon_minimum,
    )

    # Image config
    image_config = ImageConfig(
        target_h=game_config.image_config.target_h,
        target_w=game_config.image_config.target_w,
        crop_top=game_config.image_config.crop_top,
    )

    return HyperParameters(
        environment_config=env_config,
        model_config=model_config,
        training_config=training_config,
        learning_config=learning_config,
        exploration_config=exploration_config,
        image_config=image_config,
    )
