import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3
import os
import argparse
import glob
import re


def get_algorithm_class(algorithm_name):
    """Get the algorithm class based on name"""
    algorithms = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}
    return algorithms.get(algorithm_name.lower(), A2C)


def find_latest_model(models_dir, algorithm):
    """Find the most recent model for the specified algorithm"""
    print(f"Looking for models in: {os.path.abspath(models_dir)}")

    if not os.path.exists(models_dir):
        print(f"Directory does not exist: {models_dir}")
        return None, 0

    # Pattern to match model files: lunarlander_algorithm_timesteps.zip
    pattern = f"lunarlander_{algorithm}_*.zip"
    model_files = glob.glob(os.path.join(models_dir, pattern))

    print(f"Found {len(model_files)} model files matching pattern: {pattern}")

    if not model_files:
        print(f"No model files found in {models_dir}")
        return None, 0

    # Extract timesteps from filenames and find the latest
    latest_model = None
    max_timesteps = 0

    for model_file in model_files:
        filename = os.path.basename(model_file)
        print(f"  Checking file: {filename}")
        # Extract timesteps from filename like "lunarlander_ppo_11000000.zip"
        match = re.search(rf"lunarlander_{algorithm}_(\d+)\.zip", filename)
        if match:
            timesteps = int(match.group(1))
            print(f"    -> Timesteps: {timesteps:,}")
            if timesteps > max_timesteps:
                max_timesteps = timesteps
                latest_model = model_file
                print(f"    -> New best model: {filename}")

    if latest_model:
        print(
            f"Selected latest model: {os.path.basename(latest_model)} ({max_timesteps:,} timesteps)"
        )
    else:
        print("No valid models found")

    return latest_model, max_timesteps


def main():
    parser = argparse.ArgumentParser(
        description="Train Lunar Lander with infinite training"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="a2c",
        choices=["a2c", "ppo", "sac", "td3"],
        help="Algorithm to use (default: a2c)",
    )
    parser.add_argument(
        "--save-interval",
        "-s",
        type=int,
        default=100_0000,
        help="Save interval in timesteps (default: 1000000)",
    )
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="MlpPolicy",
        help="Policy type (default: MlpPolicy)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Force training from scratch, ignoring existing models",
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    # Since we're running from atari/baselines/, go up one level to reach atari/
    models_dir = f"../models/baselines/lunarlander"
    os.makedirs(models_dir, exist_ok=True)

    env = gym.make("LunarLander-v3")

    # Get algorithm class
    algorithm_class = get_algorithm_class(args.algorithm)
    print(f"Using algorithm: {args.algorithm.upper()}")
    print(f"Using policy: {args.policy}")

    # Check for existing models
    latest_model, existing_timesteps = find_latest_model(models_dir, args.algorithm)

    if latest_model and not args.from_scratch:
        print(f"Found existing model: {os.path.basename(latest_model)}")
        print(f"Continuing training from {existing_timesteps:,} timesteps")

        # Load the existing model
        model = algorithm_class.load(latest_model, env=env)
        total_timesteps = existing_timesteps
        print(f"Model loaded successfully!")
    else:
        if args.from_scratch:
            print("Starting training from scratch (--from-scratch specified)")
        else:
            print("No existing models found, starting from scratch")

        # Create new model
        model = algorithm_class(args.policy, env, verbose=1)
        total_timesteps = 0

    # Training parameters
    save_interval = args.save_interval

    print(f"Starting infinite training... Press Ctrl+C to stop")
    print(f"Save interval: {save_interval:,} timesteps")
    print(f"Models will be saved to: {models_dir}")

    try:
        while True:
            # Train for save_interval steps
            model.learn(total_timesteps=save_interval)
            total_timesteps += save_interval

            # Save the model
            model_filename = f"lunarlander_{args.algorithm}_{total_timesteps}.zip"
            model_path = os.path.join(models_dir, model_filename)
            model.save(model_path)
            print(f"Model saved: {model_filename} ({total_timesteps:,} timesteps)")

            print(f"Continuing training... Total timesteps: {total_timesteps:,}")
            print("-" * 50)

    except KeyboardInterrupt:
        print(f"\nTraining stopped by user. Total timesteps: {total_timesteps:,}")

        # Save final model
        final_model_filename = (
            f"lunarlander_{args.algorithm}_{total_timesteps}_final.zip"
        )
        final_model_path = os.path.join(models_dir, final_model_filename)
        model.save(final_model_path)
        print(f"Final model saved: {final_model_filename}")

        print("Training complete!")
        print(f"All models saved in: {os.path.abspath(models_dir)}")


if __name__ == "__main__":
    main()
