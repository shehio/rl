#!/usr/bin/env python3
"""
DQN Tester for Atari Games
Loads trained models and runs them for testing/evaluation.
"""

import gymnasium as gym
import ale_py  # This registers the ALE environments
import os
import numpy as np
import torch
import sys
import time
import json
import argparse
from typing import Dict, Any

# Add atari/algorithms to path so the `dqn` package resolves
algorithms_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, algorithms_root)

from dqn_game_configs import get_dqn_hyperparameters, get_dqn_game_config


class DQNTester:
    """Tester for DQN algorithms."""

    def __init__(self, game_name: str, config: Dict[str, Any]):
        self.game_name = game_name
        self.config = config
        self.game_config = get_dqn_game_config(game_name)

    def test(self):
        """Main testing loop."""
        try:
            print(f"Initializing {self.game_config.name} DQN for testing...")

            # Get hyperparameters with config overrides
            hyperparams = get_dqn_hyperparameters(self.game_name, self.config)

            # Create environment
            if hyperparams.environment.render_game_window:
                env = gym.make(hyperparams.environment.environment, render_mode="human")
            else:
                env = gym.make(
                    hyperparams.environment.environment, render_mode="rgb_array"
                )

            # Create agent
            agent = self.game_config.agent_class(env, hyperparams)

            # Load the specified model or find the latest
            load_episode = self.config.get("load_episode")
            if load_episode is None:
                # Find the latest episode automatically
                model_dir = os.path.dirname(hyperparams.model.model_path)
                base_name = os.path.basename(hyperparams.model.model_path)

                max_episode = 0
                for fname in os.listdir(model_dir):
                    if fname.startswith(base_name) and fname.endswith(".pkl"):
                        try:
                            ep = int(
                                fname[len(base_name) : -4]
                            )  # Remove base_name and .pkl
                            if ep > max_episode:
                                max_episode = ep
                        except ValueError:
                            continue

                if max_episode == 0:
                    print(f"Error: No model files found in {model_dir}")
                    return

                load_episode = max_episode
                print(f"Auto-detected latest episode: {load_episode}")

            model_path = f"{hyperparams.model.model_path}{load_episode}.pkl"
            epsilon_path = f"{hyperparams.model.model_path}{load_episode}.json"

            if os.path.exists(model_path):
                agent.online_model.load_state_dict(
                    torch.load(model_path, map_location=hyperparams.environment.device)
                )
                print(f"Model loaded from {model_path}")

                # Load epsilon value if JSON file exists
                if os.path.exists(epsilon_path):
                    with open(epsilon_path, "r") as outfile:
                        param = json.load(outfile)
                        agent.epsilon = param.get("epsilon", agent.epsilon)
                    print(f"Epsilon loaded: {agent.epsilon}")
                else:
                    print(
                        f"Epsilon file {epsilon_path} not found, using default epsilon"
                    )
            else:
                print(f"Model file {model_path} not found!")
                return

            print(f"Starting testing from episode {load_episode}")
            print(f"Device: {hyperparams.environment.device}")
            print(f"Epsilon: {agent.epsilon}")
            print("Press Ctrl+C to stop testing")

            # Testing loop
            episode_count = 0
            total_rewards = []

            while True:
                start_time = time.time()
                state, _ = env.reset()
                state = agent.preProcess(state)

                # Stack state: Every state contains 4 time continuous frames
                state = np.stack((state, state, state, state))

                total_reward = 0
                step_count = 0

                for step in range(hyperparams.training.max_step):
                    # Use epsilon-greedy with very low epsilon for testing
                    if np.random.random() <= 0.01:  # 1% exploration
                        action = np.random.randint(0, env.action_space.n)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.tensor(
                                state,
                                dtype=torch.float,
                                device=hyperparams.environment.device,
                            ).unsqueeze(0)
                            q_values = agent.online_model.forward(state_tensor)
                            action = torch.argmax(q_values).item()

                    # Take action
                    next_state, reward, done, truncated, info = env.step(action)
                    next_state = agent.preProcess(next_state)

                    # Stack next state
                    next_state = np.stack((next_state, state[0], state[1], state[2]))

                    state = next_state
                    total_reward += float(reward)
                    step_count += 1

                    if done or truncated:
                        break

                episode_count += 1
                total_rewards.append(total_reward)

                # Episode completed
                current_time = time.time()
                time_passed = current_time - start_time
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())

                avg_reward = (
                    np.mean(total_rewards[-100:])
                    if len(total_rewards) >= 100
                    else np.mean(total_rewards)
                )

                out_str = "Test Episode:{} Time:{} Reward:{:.2f} Avg_100_Rew:{:.3f} Duration:{:.2f} Steps:{}".format(
                    episode_count,
                    current_time_format,
                    total_reward,
                    avg_reward,
                    time_passed,
                    step_count,
                )

                print(out_str)

        except KeyboardInterrupt:
            print(f"\nTesting stopped by user")
            print(f"Total episodes tested: {episode_count}")
            if total_rewards:
                print(f"Average reward: {np.mean(total_rewards):.3f}")
                print(f"Best reward: {np.max(total_rewards):.3f}")
            env.close()

        except Exception as e:
            print(f"Error during testing: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="DQN Tester")
    parser.add_argument("game", choices=["pong", "pacman"], help="Game to test")
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

    tester = DQNTester(args.game, config)
    tester.test()


if __name__ == "__main__":
    main()
