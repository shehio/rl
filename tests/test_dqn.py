"""
Tests for DQN implementation.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add atari/algorithms to the path so the `dqn` package resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atari", "algorithms"))

try:
    import torch
    from dqn.agent import Agent
    from dqn.model import DuelCNN
    from dqn.pacman.model import DuelCNNImproved
    from dqn.config.hyperparameters import HyperParameters
    from dqn.config.environment_config import EnvironmentConfig
    from dqn.config.model_config import ModelConfig
    from dqn.config.training_config import TrainingConfig
    from dqn.config.learning_config import LearningConfig
    from dqn.config.exploration_config import ExplorationConfig
    from dqn.config.image_config import ImageConfig
except ImportError as e:
    pytest.skip(f"Could not import DQN modules: {e}", allow_module_level=True)


def make_hyperparams(**training_overrides):
    """Build a HyperParameters object with CPU device and small test values."""
    return HyperParameters(
        environment_config=EnvironmentConfig(device=torch.device("cpu")),
        model_config=ModelConfig(),
        training_config=TrainingConfig(**training_overrides),
        learning_config=LearningConfig(),
        exploration_config=ExplorationConfig(),
        image_config=ImageConfig(),
    )


def make_mock_env(num_actions=6):
    """Mock Atari environment exposing the attributes the Agent reads."""
    env = MagicMock()
    env.observation_space.shape = (210, 160, 3)
    env.action_space.n = num_actions
    return env


class TestDuelCNN:
    """Test the DuelCNN model."""

    def test_model_initialization(self):
        """Test that the model can be initialized."""
        model = DuelCNN(h=80, w=64, output_size=6)
        assert model is not None
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "Alinear1")  # Advantage stream
        assert hasattr(model, "Vlinear1")  # Value stream

    def test_model_forward_pass_shape(self):
        """Forward pass returns one Q-value per action for each sample."""
        model = DuelCNN(h=80, w=64, output_size=6)
        model.eval()
        x = torch.randn(2, 4, 80, 64)
        q = model(x)
        assert q.shape == (2, 6)

    def test_dueling_aggregation_is_per_sample(self):
        """Q-values of one sample must not depend on other samples in the batch.

        Regression test for the dueling aggregation: the advantage mean must
        be taken per sample over actions (Wang et al. 2016), not over the
        whole batch.
        """
        torch.manual_seed(0)
        model = DuelCNN(h=80, w=64, output_size=6)
        model.eval()
        x1 = torch.randn(1, 4, 80, 64)
        x2 = torch.randn(1, 4, 80, 64)
        q_single = model(x1)
        q_batch = model(torch.cat([x1, x2]))
        assert torch.allclose(q_single[0], q_batch[0], atol=1e-5)


class TestDuelCNNImproved:
    """Test the improved (Pacman) DuelCNN model."""

    def test_model_forward_pass_shape(self):
        model = DuelCNNImproved(h=80, w=64, output_size=9)
        model.eval()
        x = torch.randn(2, 4, 80, 64)
        q = model(x)
        assert q.shape == (2, 9)

    def test_dueling_aggregation_is_per_sample(self):
        torch.manual_seed(0)
        model = DuelCNNImproved(h=80, w=64, output_size=9)
        model.eval()
        x1 = torch.randn(1, 4, 80, 64)
        x2 = torch.randn(1, 4, 80, 64)
        q_single = model(x1)
        q_batch = model(torch.cat([x1, x2]))
        assert torch.allclose(q_single[0], q_batch[0], atol=1e-5)


class TestDQNAgent:
    """Test the DQN Agent class."""

    def test_agent_initialization(self):
        agent = Agent(make_mock_env(), make_hyperparams())
        assert agent.action_size == 6
        assert agent.online_model is not None
        assert agent.target_model is not None
        assert hasattr(agent, "act")
        assert hasattr(agent, "train")

    def test_agent_act_explore(self):
        """With epsilon=1 the agent always explores and returns a valid action."""
        agent = Agent(make_mock_env(), make_hyperparams())
        agent.epsilon = 1.0
        state = np.random.randn(4, 80, 64).astype(np.float32)
        action = agent.act(state)
        assert 0 <= action < 6

    def test_agent_act_exploit(self):
        """With epsilon=0 the agent uses the network and returns a valid action."""
        agent = Agent(make_mock_env(), make_hyperparams())
        agent.epsilon = 0.0
        agent.online_model.eval()
        state = np.random.randn(4, 80, 64).astype(np.float32)
        action = agent.act(state)
        assert 0 <= action < 6

    def test_agent_preprocess(self):
        """preProcess crops, resizes, grayscales, and normalizes the frame."""
        agent = Agent(make_mock_env(), make_hyperparams())
        image = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        frame = agent.preProcess(image)
        assert frame.ndim == 2
        assert frame.size == 80 * 64
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0

    def test_agent_store_results(self):
        agent = Agent(make_mock_env(), make_hyperparams())
        state = np.random.randn(4, 80, 64).astype(np.float32)
        next_state = np.random.randn(4, 80, 64).astype(np.float32)
        agent.storeResults(state, 2, 1.0, next_state, False)
        assert len(agent.memory) == 1

    def test_agent_train_requires_min_memory(self):
        """train() is a no-op until the replay buffer reaches min_memory_len."""
        agent = Agent(make_mock_env(), make_hyperparams())
        loss, max_q = agent.train()
        assert loss == 0
        assert max_q == 0

    def test_agent_train_updates_model(self):
        """train() runs a real optimization step once memory is sufficient."""
        agent = Agent(
            make_mock_env(),
            make_hyperparams(batch_size=4, min_memory_len=4, max_memory_len=16),
        )
        for _ in range(4):
            state = np.random.randn(4, 80, 64).astype(np.float32)
            next_state = np.random.randn(4, 80, 64).astype(np.float32)
            agent.storeResults(state, 1, 1.0, next_state, False)
        loss, max_q = agent.train()
        assert torch.is_tensor(loss)
        assert loss.item() >= 0.0
        assert isinstance(max_q, float)

    def test_adaptive_epsilon_decays(self):
        agent = Agent(make_mock_env(), make_hyperparams())
        agent.epsilon = 1.0
        agent.adaptiveEpsilon()
        assert agent.epsilon == pytest.approx(agent.epsilon_decay)

    def test_adaptive_epsilon_respects_minimum(self):
        agent = Agent(make_mock_env(), make_hyperparams())
        agent.epsilon = agent.epsilon_minimum
        agent.adaptiveEpsilon()
        assert agent.epsilon == agent.epsilon_minimum


class TestDQNConfig:
    """Test that DQN configuration classes work with defaults."""

    def test_config_defaults(self):
        hyperparams = make_hyperparams()
        assert hyperparams.environment is not None
        assert hyperparams.model is not None
        assert hyperparams.training is not None
        assert hyperparams.learning.alpha is not None
        assert hyperparams.exploration.epsilon_start == 1.0
        assert hyperparams.image.target_h == 80
