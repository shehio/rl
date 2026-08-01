#!/usr/bin/env python3
"""
Unified DQN Trainer for Atari Games
Supports Pong and Ms. Pacman with automatic model loading and configuration.
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
from collections import deque
from typing import Dict, Any

# Add atari/algorithms to path so the `dqn` package resolves
algorithms_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, algorithms_root)

from dqn_game_configs import get_dqn_hyperparameters, get_dqn_game_config

try:
    import wandb
except ImportError:
    wandb = None


class DQNTrainer:
    """Unified trainer for DQN algorithms."""

    def __init__(self, game_name: str, config: Dict[str, Any]):
        self.game_name = game_name
        self.config = config
        self.game_config = get_dqn_game_config(game_name)
        self.wandb_run = None

    def init_wandb(self, hyperparams):
        """Start a wandb run, or leave tracking off when disabled/unavailable."""
        if self.config.get("no_wandb") or wandb is None:
            if wandb is None and not self.config.get("no_wandb"):
                print("wandb not installed; continuing without experiment tracking")
            return

        self.wandb_run = wandb.init(
            entity="shehio",
            project="rl-atari",
            config={
                "algorithm": "dqn-custom",
                "env": hyperparams.environment.environment,
                "game": self.game_name,
                "learning_rate": hyperparams.learning.alpha,
                "gamma": hyperparams.learning.gamma,
                "batch_size": hyperparams.training.batch_size,
                "max_episode": hyperparams.training.max_episode,
                "max_step": hyperparams.training.max_step,
                "max_memory_len": hyperparams.training.max_memory_len,
                "min_memory_len": hyperparams.training.min_memory_len,
                "epsilon_start": hyperparams.exploration.epsilon_start,
                "epsilon_decay": hyperparams.exploration.epsilon_decay,
                "epsilon_minimum": hyperparams.exploration.epsilon_minimum,
                "device": str(hyperparams.environment.device),
            },
            tags=["dqn-custom", self.game_name],
        )

    def train(self):
        """Main training loop."""
        try:
            print(f"Initializing {self.game_config.name} DQN training...")

            # Get hyperparameters with config overrides
            hyperparams = get_dqn_hyperparameters(self.game_name, self.config)
            self.init_wandb(hyperparams)

            # Print hyperparameters
            print(f"Learning rate (alpha): {hyperparams.learning.alpha}")
            print(f"Gamma: {hyperparams.learning.gamma}")
            print(f"Batch size: {hyperparams.training.batch_size}")
            print(f"Epsilon start: {hyperparams.exploration.epsilon_start}")

            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(hyperparams.model.model_path), exist_ok=True)

            # Create environment
            if hyperparams.environment.render_game_window:
                env = gym.make(hyperparams.environment.environment, render_mode="human")
            else:
                env = gym.make(
                    hyperparams.environment.environment, render_mode="rgb_array"
                )

            # Validate hyperparameters before creating agent
            if hyperparams.learning.alpha is None:
                raise ValueError(
                    "Learning rate (alpha) cannot be None. Check hyperparameter configuration."
                )

            # Create agent
            agent = self.game_config.agent_class(env, hyperparams)

            # Load model if specified
            if hyperparams.model.load_model_from_file:
                model_path = f"{hyperparams.model.model_path}{hyperparams.model.load_file_episode}.pkl"
                epsilon_path = f"{hyperparams.model.model_path}{hyperparams.model.load_file_episode}.json"

                if os.path.exists(model_path):
                    agent.online_model.load_state_dict(
                        torch.load(
                            model_path, map_location=hyperparams.environment.device
                        )
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

            # Determine starting episode
            if hyperparams.model.load_model_from_file:
                start_episode = hyperparams.model.load_file_episode + 1
            else:
                start_episode = 0

            print(f"Starting training from episode {start_episode}")
            print(f"Device: {hyperparams.environment.device}")
            print(f"Batch size: {hyperparams.training.batch_size}")
            print(f"Learning rate: {hyperparams.learning.alpha}")
            print(f"Epsilon: {agent.epsilon}")

            # Training loop
            last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
            total_step = 1  # Cumulative sum of all steps in episodes

            for episode in range(start_episode, hyperparams.training.max_episode):
                start_time = time.time()  # Keep time
                state, _ = env.reset()
                state = agent.preProcess(state)

                # Stack state: Every state contains 4 time continuous frames
                # We stack frames like 4 channel image
                state = np.stack((state, state, state, state))

                total_max_q_val = 0  # Total max q vals
                total_reward = 0  # Total reward for each episode
                total_loss = 0  # Total loss for each episode

                for step in range(hyperparams.training.max_step):
                    action = agent.act(state)

                    # Take action
                    next_state, reward, done, truncated, info = env.step(action)
                    next_state = agent.preProcess(next_state)

                    # Stack next state: Every state contains 4 time continuous frames
                    # We stack frames like 4 channel image
                    next_state = np.stack((next_state, state[0], state[1], state[2]))

                    # Store experience
                    agent.storeResults(state, action, reward, next_state, done)

                    # Train if enough memory
                    if hyperparams.model.train_model:
                        loss, max_q = agent.train()
                    else:
                        loss, max_q = [0, 0]

                    total_loss += loss
                    total_max_q_val += max_q
                    state = next_state
                    total_reward += float(reward)
                    total_step += 1

                    # Should this be based on episode or step?
                    if total_step % 1000 == 0:
                        agent.adaptiveEpsilon()

                    if done or truncated:
                        break

                # Update target network periodically
                if episode % 100 == 0:
                    agent.target_model.load_state_dict(agent.online_model.state_dict())

                # Episode completed - detailed logging
                current_time = time.time()  # Keep current time
                time_passed = current_time - start_time  # Find episode duration
                current_time_format = time.strftime(
                    "%H:%M:%S", time.gmtime()
                )  # Get current dateTime as HH:MM:SS
                epsilon_dict = {
                    "epsilon": agent.epsilon
                }  # Create epsilon dict to save model as file

                if (
                    hyperparams.model.save_models
                    and episode % hyperparams.model.save_model_interval == 0
                ):  # Save model as file
                    weights_path = f"{hyperparams.model.model_path}{episode}.pkl"
                    epsilon_path = f"{hyperparams.model.model_path}{episode}.json"

                    torch.save(agent.online_model.state_dict(), weights_path)
                    with open(epsilon_path, "w") as outfile:
                        json.dump(epsilon_dict, outfile)

                if hyperparams.model.train_model:
                    agent.target_model.load_state_dict(
                        agent.online_model.state_dict()
                    )  # Update target model

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step if step > 0 else 0

                out_str = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
                    episode,
                    current_time_format,
                    total_reward,
                    total_loss,
                    np.mean(last_100_ep_reward),
                    avg_max_q_val,
                    agent.epsilon,
                    time_passed,
                    step,
                    total_step,
                )

                print(out_str)

                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "episode": episode,
                            "episode_reward": total_reward,
                            "episode_loss": total_loss,
                            "last_100_avg_reward": float(np.mean(last_100_ep_reward)),
                            "avg_max_q": avg_max_q_val,
                            "epsilon": agent.epsilon,
                            "episode_steps": step,
                            "total_steps": total_step,
                            "episode_duration_sec": time_passed,
                        }
                    )

                if hyperparams.model.save_models:
                    output_path = (
                        f"{hyperparams.model.model_path}out.txt"  # Save outStr to file
                    )
                    with open(output_path, "a") as outfile:
                        outfile.write(out_str + "\n")

            env.close()
            self.finish_wandb(episode, last_100_ep_reward)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print(f"Final episode: {episode}")
            if last_100_ep_reward:
                print(f"Final running reward: {np.mean(last_100_ep_reward):.3f}")
            env.close()
            self.finish_wandb(episode, last_100_ep_reward)

        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def finish_wandb(self, episode, last_100_ep_reward):
        """Record final results as summary metrics and close the run."""
        if not self.wandb_run:
            return
        self.wandb_run.summary["final_episode"] = episode
        if last_100_ep_reward:
            self.wandb_run.summary["final_last_100_avg_reward"] = float(
                np.mean(last_100_ep_reward)
            )
        self.wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Unified DQN Trainer")
    parser.add_argument("game", choices=["pong", "pacman"], help="Game to train on")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--no-load-network", action="store_true", help="Don't load pre-trained network"
    )
    parser.add_argument(
        "--load-episode",
        type=int,
        help="Episode number to load from (defaults to latest)",
    )
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--save-interval", type=int, help="Save interval")
    parser.add_argument(
        "--epsilon-decay", type=float, help="Epsilon decay rate (e.g. 0.99)"
    )
    parser.add_argument(
        "--memory-size", type=int, help="Replay buffer capacity (max_memory_len)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases experiment tracking",
    )

    args = parser.parse_args()

    config = {
        "render": args.render,
        "no_load_network": args.no_load_network,
        "load_episode": args.load_episode,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "save_interval": args.save_interval,
        "epsilon_decay": args.epsilon_decay,
        "memory_size": args.memory_size,
        "no_wandb": args.no_wandb,
    }

    trainer = DQNTrainer(args.game, config)
    trainer.train()


if __name__ == "__main__":
    main()
