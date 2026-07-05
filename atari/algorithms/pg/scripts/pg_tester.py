#!/usr/bin/env python3
"""
Policy Gradient Tester for Atari Games
Loads trained models and runs them for testing/evaluation.
"""

import sys
import os
import numpy as np
import argparse
from typing import Dict, Any

# Add atari/algorithms to path so the `pg` package resolves
algorithms_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, algorithms_root)

from pg.hyperparameters import HyperParameters
from pg.game import Game
from game_configs import (
    get_game_config,
    get_input_size,
    get_output_size,
    get_network_file,
)


class PolicyGradientTester:
    """Tester for policy gradient algorithms."""

    def __init__(self, game_name: str, config: Dict[str, Any]):
        self.game_name = game_name
        self.config = config
        self.game_config = get_game_config(game_name)
        self.previous_frame = None

    def setup_network(self, game: Game) -> Any:
        """Setup the appropriate network based on game type."""
        input_size = get_input_size(self.game_name)
        output_size = get_output_size(self.game_name, game.env)
        hidden_layers_count = self.config.get("hidden_layers_count", 200)
        network_file = self.config.get("network_file", get_network_file(self.game_name))

        if self.game_name == "pacman":
            print(f"Creating CNN policy network with {output_size} actions...")
            print(f"Input: 7 channels (80x80x7 color features)")
            # For CNN, pass input_channels (7) instead of total input_size
            return self.game_config.network_class(
                7,
                hidden_layers_count,
                output_size,
                network_file,
                self.game_config.game_id,
            )
        else:
            print(f"Creating MLP policy network...")
            return self.game_config.network_class(
                input_size,
                hidden_layers_count,
                output_size,
                network_file,
                self.game_config.game_id,
            )

    def setup_agent(self, policy_network: Any, hyperparams: HyperParameters) -> Any:
        """Setup the appropriate agent."""
        return self.game_config.agent_class(policy_network, hyperparams)

    def preprocess_state(self, game: Game) -> np.ndarray:
        """Preprocess the game state."""
        if self.game_config.preprocess_func:
            return self.game_config.preprocess_func(game, self.previous_frame)
        return game.get_frame_difference()

    def test(self):
        """Main testing loop."""
        try:
            print(f"Initializing {self.game_config.name} game for testing...")

            input_size = get_input_size(self.game_name)
            game = Game(
                self.game_config.game_id,
                self.config.get("render", True),  # Default to rendering for testing
                input_size,
                0,  # Start from episode 0 for testing
            )

            policy_network = self.setup_network(game)

            # Load the specified model or find the latest
            load_episode = self.config.get("load_episode")
            if load_episode is None:
                # Find the latest episode automatically
                import re

                model_dir = "atari/scripts/policy-gradient/models"

                pattern_map = {
                    "pong": r"torch_mlp_ALE_Pong_v5.*_(\d+)$",
                    "breakout": r"torch_mlp_ALE_Breakout_v5.*_(\d+)$",
                    "pacman": r"torch_mlp_pacman_ALE_MsPacman_v5_cnn.*_(\d+)$",
                }

                pattern = pattern_map.get(self.game_name)
                if not pattern:
                    print("Error: Cannot auto-detect episode for this game")
                    return

                max_episode = 0
                for fname in os.listdir(model_dir):
                    match = re.search(pattern, fname)
                    if match:
                        try:
                            ep = int(match.group(1))
                            if ep > max_episode:
                                max_episode = ep
                        except ValueError:
                            continue

                if max_episode == 0:
                    print(f"Error: No model files found for {self.game_name}")
                    return

                load_episode = max_episode
                print(f"Auto-detected latest episode: {load_episode}")

            print(f"Loading network from episode {load_episode}...")
            policy_network.load_network(load_episode)

            # Create minimal hyperparams for testing (no training)
            hyperparams = HyperParameters(
                learning_rate=1e-4,  # Not used in testing
                decay_rate=0.99,
                gamma=0.99,
                batch_size=1,  # Not used in testing
                save_interval=10000,
            )

            agent = self.setup_agent(policy_network, hyperparams)

            # Call game-specific initialization if needed
            if self.game_config.init_func:
                self.game_config.init_func(game)

            # Initialize game-specific variables
            if self.game_name == "breakout":
                game.previous_lives = 5  # Initialize for Breakout fire logic

            print("Starting testing loop...")
            print("Press Ctrl+C to stop testing")

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

        try:
            episode_count = 0
            total_reward = 0

            while True:
                state = self.preprocess_state(game)

                # Use the agent's action sampling logic (but with minimal exploration for testing)
                if self.game_name == "pacman":
                    # For multi-action agents, use the policy directly
                    action_probs, _ = agent.policy_network.forward_pass(state)
                    action = np.argmax(action_probs)  # Take best action
                else:
                    # For binary action agents, use the agent's sampling logic
                    # But override the random sampling to always take the best action
                    action_prob, _ = agent.policy_network.forward_pass(state)
                    action = 2 if action_prob > 0.5 else 3  # DOWN or UP

                observation, reward, done, info = game.step(action)
                total_reward += reward
                game.update_episode_stats(reward)  # Update game's episode stats

                # Call game-specific post-step logic
                if self.game_config.post_step_func:
                    self.game_config.post_step_func(game, info)

                if done:
                    episode_count += 1
                    print(f"Episode {episode_count}: Total Reward = {total_reward:.1f}")
                    total_reward = 0

                    # Call game-specific post-episode logic
                    if self.game_config.post_episode_func:
                        self.game_config.post_episode_func(game, agent)

                    game.end_episode()

                    # Reset frame for next episode (for games that need it)
                    if self.game_name == "pacman":
                        self.previous_frame = None

        except KeyboardInterrupt:
            print(f"\nTesting stopped by user")
            print(f"Total episodes tested: {episode_count}")

        except Exception as e:
            print(f"Error during testing: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Policy Gradient Tester")
    parser.add_argument(
        "game", choices=["pong", "breakout", "pacman"], help="Game to test"
    )
    parser.add_argument(
        "--load-episode",
        type=int,
        help="Episode number to load from (defaults to latest)",
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()

    config = {
        "render": not args.no_render,
        "load_episode": args.load_episode,
    }

    tester = PolicyGradientTester(args.game, config)
    tester.test()


if __name__ == "__main__":
    main()
