"""
Tests for Policy Gradient implementation.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add atari/algorithms to the path so the `pg` package resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atari", "algorithms"))

try:
    from pg.agent import Agent, DOWN, UP
    from pg.memory import Memory
    from pg.hyperparameters import HyperParameters
    from pg.mlp_torch import MLP
except ImportError as e:
    pytest.skip(
        f"Could not import Policy Gradient modules: {e}", allow_module_level=True
    )


def make_hyperparams(batch_size=10, save_interval=100):
    return HyperParameters(
        learning_rate=1e-4,
        decay_rate=0.99,
        gamma=0.99,
        batch_size=batch_size,
        save_interval=save_interval,
    )


def make_mock_policy_network(probability=0.5, hidden_size=8):
    network = MagicMock()
    network.forward_pass.return_value = (probability, np.zeros(hidden_size))
    return network


class TestMemory:
    """Test the Memory class."""

    def test_memory_initialization(self):
        memory = Memory()
        assert memory.states == []
        assert memory.hidden_layers == []
        assert memory.dlogps == []
        assert memory.rewards == []

    def test_memory_str_reports_counts(self):
        memory = Memory()
        memory.states.append(np.zeros(4))
        memory.rewards.append(1.0)
        text = str(memory)
        assert "states=1" in text
        assert "rewards=1" in text


class TestMLP:
    """Test the PyTorch MLP policy network."""

    def test_mlp_initialization(self):
        mlp = MLP(
            input_count=16,
            hidden_layers_count=8,
            output_count=1,
            network_file="torch_mlp_test",
            game_name="testgame",
        )
        assert mlp is not None
        assert hasattr(mlp, "forward")
        assert hasattr(mlp, "forward_pass")

    def test_mlp_forward_pass(self):
        """forward_pass returns a probability and the hidden activations."""
        mlp = MLP(
            input_count=16,
            hidden_layers_count=8,
            output_count=1,
            network_file="torch_mlp_test",
            game_name="testgame",
        )
        state = np.random.randn(16).astype(np.float32)
        probability, hidden = mlp.forward_pass(state)
        assert 0.0 < probability < 1.0  # Sigmoid output
        assert hidden.shape == (8,)

    def test_mlp_train_consumes_gradient_buffer(self):
        """train() runs an optimization step and clears the gradient buffer."""
        mlp = MLP(
            input_count=16,
            hidden_layers_count=8,
            output_count=1,
            network_file="torch_mlp_test",
            game_name="testgame",
        )
        epx = np.random.randn(3, 16).astype(np.float32)
        eph = np.random.randn(3, 8).astype(np.float32)
        epdlogp = np.random.randn(3, 1).astype(np.float32)
        mlp.backward_pass(eph, epdlogp, epx)
        assert len(mlp.gradient_buffer) == 1
        mlp.train(learning_rate=1e-4, decay_rate=0.99)
        assert mlp.gradient_buffer == []

    def test_mlp_save_and_load_network(self, tmp_path, monkeypatch):
        """save_network/load_network round-trip through the models directory."""
        workdir = tmp_path / "atari" / "algorithms" / "pg"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        mlp = MLP(
            input_count=16,
            hidden_layers_count=8,
            output_count=1,
            network_file="torch_mlp_test",
            game_name="testgame",
        )
        mlp.save_network(100)

        saved = (
            tmp_path
            / "atari"
            / "models"
            / "pg"
            / "testgame"
            / "torch_mlp_test_testgame_i16_h8_o1_100"
        )
        assert saved.exists()

        # Loading restores the same weights
        other = MLP(
            input_count=16,
            hidden_layers_count=8,
            output_count=1,
            network_file="torch_mlp_test",
            game_name="testgame",
        )
        other.load_network(100)
        state = np.random.randn(16).astype(np.float32)
        probability_a, _ = mlp.forward_pass(state)
        probability_b, _ = other.forward_pass(state)
        assert probability_a == pytest.approx(probability_b, abs=1e-6)


class TestAgent:
    """Test the REINFORCE Agent."""

    def test_sample_action_returns_valid_action(self):
        agent = Agent(make_mock_policy_network(), make_hyperparams())
        state = np.random.randn(16)
        action = agent.sample_action(state)
        assert action in (DOWN, UP)

    def test_sample_and_record_action_records_memory(self):
        agent = Agent(make_mock_policy_network(), make_hyperparams())
        state = np.random.randn(16)
        action = agent.sample_and_record_action(state)
        assert action in (DOWN, UP)
        assert len(agent.memory.states) == 1
        assert len(agent.memory.hidden_layers) == 1
        assert len(agent.memory.dlogps) == 1

    def test_reap_reward(self):
        agent = Agent(make_mock_policy_network(), make_hyperparams())
        agent.reap_reward(1.0)
        agent.reap_reward(-1.0)
        assert agent.memory.rewards == [1.0, -1.0]

    def test_discount_and_normalize_rewards(self):
        """Rewards are discounted backwards and standardized to unit normal."""
        agent = Agent(make_mock_policy_network(), make_hyperparams())
        rewards = np.array([[0.0], [0.0], [1.0]])
        discounted = agent._Agent__discount_and_normalize_rewards(rewards, gamma=0.5)
        # Discounting: [0.25, 0.5, 1.0] before normalization, so ordering holds
        assert discounted[0] < discounted[1] < discounted[2]
        assert np.mean(discounted) == pytest.approx(0.0, abs=1e-8)
        assert np.std(discounted) == pytest.approx(1.0, abs=1e-6)

    def test_episode_end_updates_train_and_reset(self):
        """Episode end backpropagates, trains on batch boundary, resets memory."""
        network = make_mock_policy_network()
        agent = Agent(network, make_hyperparams(batch_size=10, save_interval=10))

        state = np.random.randn(16)
        for reward in (0.0, 0.0, 1.0):
            agent.sample_and_record_action(state)
            agent.reap_reward(reward)

        agent.make_episode_end_updates(episode_number=10)

        network.backward_pass.assert_called_once()
        network.train.assert_called_once()
        network.save_network.assert_called_once_with(10)
        assert agent.memory.states == []  # Fresh memory for next episode

    def test_episode_end_skips_training_without_rewards(self):
        """Without any reward signal there is nothing to learn from."""
        network = make_mock_policy_network()
        agent = Agent(network, make_hyperparams())

        state = np.random.randn(16)
        agent.sample_and_record_action(state)
        agent.reap_reward(0.0)

        agent.make_episode_end_updates(episode_number=1)
        network.backward_pass.assert_not_called()


class TestPacmanImports:
    """The Pacman-specific policy gradient modules should be importable."""

    def test_pacman_modules_import(self):
        from pg.pacman.multi_action_agent import MultiActionAgent
        from pg.pacman.cnn_torch_multiaction import CNNMultiAction
        from pg.pacman.mlp_torch_multiaction import MLPMultiAction
        from pg.pacman.memory_multiaction import MemoryMultiAction

        assert MultiActionAgent is not None
        assert CNNMultiAction is not None
        assert MLPMultiAction is not None
        assert MemoryMultiAction is not None
