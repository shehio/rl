from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py  # This registers the ALE environments
import os
import glob
import re
import argparse
import sys

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train RL agents on Atari games using Stable Baselines3"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["ppo", "dqn", "a2c", "a2c_cnn"],
        default="ppo",
        help="Algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=str,
        default="100000",
        help='Number of timesteps to train (use "infinite" for continuous training)',
    )
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default="ALE/Pong-v5",
        help="Environment to train on (default: ALE/Pong-v5)",
    )
    parser.add_argument(
        "--n_envs",
        "-n",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated wandb tags (e.g. smoke-ab,baseline)",
    )

    return parser.parse_args()


def init_wandb(args, policy):
    """Start a wandb run, or return None when disabled/unavailable."""
    if args.no_wandb or wandb is None:
        if wandb is None and not args.no_wandb:
            print("wandb not installed; continuing without experiment tracking")
        return None

    tags = args.wandb_tags.split(",") if args.wandb_tags else [args.algorithm, "sb3"]
    return wandb.init(
        entity="shehio",
        project="rl-atari",
        config={
            "algorithm": args.algorithm,
            "env": args.env,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "policy": policy,
        },
        tags=tags,
        sync_tensorboard=True,  # Stream SB3's tensorboard metrics to wandb
    )


def get_env_name(env_string):
    """Extract a clean environment name from the full environment string"""
    # Remove ALE/ prefix and -v5 suffix
    env_name = (
        env_string.replace("ALE/", "")
        .replace("-v5", "")
        .replace("-v4", "")
        .replace("-v0", "")
    )
    return env_name.lower()


def ensure_model_dir(model_dir):
    """Ensure the model directory exists"""
    os.makedirs(model_dir, exist_ok=True)


def find_latest_model(base_path):
    pattern = f"{base_path}_*.zip"
    model_files = glob.glob(pattern)

    if not model_files:
        return None, 0

    # Extract timesteps from filenames and find the latest
    latest_model = None
    latest_timesteps = 0

    for file in model_files:
        # Extract timesteps from filename (e.g., pong_ppo_cnn_100000.zip -> 100000)
        match = re.search(r"_(\d+)\.zip$", file)
        if match:
            timesteps = int(match.group(1))
            if timesteps > latest_timesteps:
                latest_timesteps = timesteps
                latest_model = file.replace(".zip", "")

    return latest_model, latest_timesteps


def main():
    args = parse_arguments()

    # Create environment
    env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames

    # Get clean environment name
    env_name = get_env_name(args.env)

    # Map algorithm choice to class and configuration
    algorithm_map = {
        "ppo": (PPO, f"{env_name}_ppo_cnn", "CnnPolicy"),
        "dqn": (DQN, f"{env_name}_dqn_cnn", "CnnPolicy"),
        "a2c": (A2C, f"{env_name}_a2c_mlp", "MlpPolicy"),
        "a2c_cnn": (A2C, f"{env_name}_a2c_cnn", "CnnPolicy"),
    }

    algorithm, base_model_path, policy = algorithm_map[args.algorithm]
    print(f"Selected: {algorithm.__name__} with {policy} on {args.env}")

    run = init_wandb(args, policy)
    tensorboard_log = "./tb_logs" if run else None
    callback = WandbCallback(verbose=2) if run else None

    # Set up model directory
    model_dir = f"../models/baselines"
    ensure_model_dir(model_dir)

    # Find and load the most recent model
    latest_model, current_timesteps = find_latest_model(
        os.path.join(model_dir, base_model_path)
    )

    if latest_model:
        print(f"Found latest model: {latest_model} ({current_timesteps:,} timesteps)")
        print("Loading latest model...")
        model = algorithm.load(
            latest_model, env=env, verbose=1, tensorboard_log=tensorboard_log
        )
    else:
        print("No existing models found. Starting fresh...")
        model = algorithm(policy, env, verbose=1, tensorboard_log=tensorboard_log)
        current_timesteps = 0

    # Handle training duration
    training_input = args.timesteps.lower()

    if training_input == "infinite" or training_input == "inf":
        print("Starting infinite training... (Press Ctrl+C to stop)")
        try:
            while True:
                # Train in chunks of 100000 timesteps
                model.learn(
                    total_timesteps=100000,
                    reset_num_timesteps=False,
                    callback=callback,
                )
                current_timesteps += 100000

                # Save checkpoint every 100000 timesteps
                save_path = os.path.join(
                    model_dir, f"{base_model_path}_{current_timesteps}"
                )
                model.save(save_path)
                print(
                    f"Checkpoint saved: {save_path} ({current_timesteps:,} total timesteps)"
                )

        except KeyboardInterrupt:
            print(f"\nTraining stopped by user at {current_timesteps:,} timesteps")
            final_save_path = os.path.join(
                model_dir, f"{base_model_path}_{current_timesteps}_final"
            )
            model.save(final_save_path)
            print(f"Final model saved as: {final_save_path}")
    else:
        # Finite training
        try:
            training_timesteps = int(training_input)
            print(f"Starting training for {training_timesteps:,} timesteps...")
            model.learn(
                total_timesteps=training_timesteps,
                reset_num_timesteps=False,
                callback=callback,
            )

            # Save the final model
            final_timesteps = current_timesteps + training_timesteps
            save_path = os.path.join(model_dir, f"{base_model_path}_{final_timesteps}")
            model.save(save_path)
            print(f"Training completed! Model saved as: {save_path}")

        except ValueError:
            print(
                f"Invalid timesteps value: {training_input}. Defaulting to 100,000 timesteps..."
            )
            model.learn(
                total_timesteps=100000,
                reset_num_timesteps=False,
                callback=callback,
            )
            final_timesteps = current_timesteps + 100000
            save_path = os.path.join(model_dir, f"{base_model_path}_{final_timesteps}")
            model.save(save_path)
            print(f"Training completed! Model saved as: {save_path}")

    if run:
        run.summary["total_timesteps_trained"] = model.num_timesteps
        run.finish()


if __name__ == "__main__":
    main()
