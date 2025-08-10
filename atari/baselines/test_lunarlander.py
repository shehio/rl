import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3
import os
import argparse
import glob


def get_algorithm_class(algorithm_name):
    """Get the algorithm class based on name"""
    algorithms = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}
    return algorithms.get(algorithm_name.lower(), A2C)


def list_available_models():
    """List all available lunar lander models"""
    # Since we're running from atari/baselines/, go up one level to reach atari/
    models_dir = "../models/baselines/lunarlander"
    if not os.path.exists(models_dir):
        print(f"No models directory found at {models_dir}")
        return []

    model_files = glob.glob(os.path.join(models_dir, "*.zip"))
    if not model_files:
        print(f"No model files found in {models_dir}")
        return []

    print("Available models:")
    for i, model_file in enumerate(sorted(model_files)):
        filename = os.path.basename(model_file)
        print(f"  {i+1}. {filename}")

    return model_files


def test_model(
    model_path, algorithm, episodes=5, render=True, policy="MlpPolicy", verbose=False
):
    """Test a specific model"""
    print(f"Testing model: {os.path.basename(model_path)}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {episodes}")
    print(f"Rendering: {'Yes' if render else 'No'}")

    # Create environment
    try:
        if render:
            env = gym.make("LunarLander-v3", render_mode="human")
        else:
            env = gym.make("LunarLander-v3")
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return None

    # Load the model
    try:
        algorithm_class = get_algorithm_class(algorithm)
        model = algorithm_class.load(model_path, env=env)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return None

    total_reward = 0
    episode_rewards = []

    print(f"\nStarting evaluation for {episodes} episodes...")
    print("=" * 60)

    for ep in range(episodes):
        print(f"\n--- Episode {ep+1}/{episodes} ---")

        try:
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            steps = 0

            print(f"  Starting episode {ep+1}...")

            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, truncated, info = env.step(action)
                episode_reward += rewards
                steps += 1

                # Print step-by-step rewards if verbose
                if verbose and steps <= 20:
                    print(
                        f"    Step {steps}: reward = {rewards:.3f}, total = {episode_reward:.3f}"
                    )
                elif verbose and steps == 21:
                    print(f"    ... (continuing with step-by-step rewards hidden)")

                # Print periodic updates
                if steps % 50 == 0:
                    print(f"    Step {steps}: cumulative reward = {episode_reward:.3f}")

            # Episode completed
            print(f"  Episode {ep+1} COMPLETED:")
            print(f"    Final reward: {episode_reward:.3f}")
            print(f"    Total steps: {steps}")
            print(f"    Done: {done}, Truncated: {truncated}")

            total_reward += episode_reward
            episode_rewards.append(episode_reward)

        except Exception as e:
            print(f"  ERROR during episode {ep+1}: {e}")
            episode_rewards.append(0)  # Add 0 for failed episode
            continue

    env.close()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    if episode_rewards:
        successful_episodes = len([r for r in episode_rewards if r != 0])
        avg_reward = total_reward / len(episode_rewards)

        print(f"Episodes completed: {successful_episodes}/{len(episode_rewards)}")
        print(f"Total reward across all episodes: {total_reward:.3f}")
        print(f"Average reward per episode: {avg_reward:.3f}")
        print(f"Best episode reward: {max(episode_rewards):.3f}")
        print(f"Worst episode reward: {min(episode_rewards):.3f}")

        if successful_episodes > 0:
            successful_rewards = [r for r in episode_rewards if r != 0]
            successful_avg = sum(successful_rewards) / len(successful_rewards)
            print(f"Average reward (successful episodes only): {successful_avg:.3f}")

        print(f"\nEpisode-by-episode breakdown:")
        for i, reward in enumerate(episode_rewards):
            status = "✓" if reward != 0 else "✗"
            print(f"  Episode {i+1}: {reward:.3f} {status}")
    else:
        print("No episodes completed successfully")

    return avg_reward if episode_rewards else None


def main():
    parser = argparse.ArgumentParser(description="Test trained Lunar Lander models")
    parser.add_argument(
        "--model", "-m", type=str, help="Path to specific model file (optional)"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="a2c",
        choices=["a2c", "ppo", "sac", "td3"],
        help="Algorithm used in the model (default: a2c)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=5,
        help="Number of episodes to test (default: 5)",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering (faster evaluation)"
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="MlpPolicy",
        help="Policy type (default: MlpPolicy)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed step-by-step output"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    # If no specific model provided, show available models
    if not args.model:
        print("No model specified. Available models:")
        model_files = list_available_models()
        if not model_files:
            return

        print(f"\nUse --model to specify a specific model file.")
        print(
            f"Example: python test_lunarlander.py --model {os.path.basename(model_files[0])}"
        )
        return

    # Check if model file exists
    if not os.path.exists(args.model):
        # Try to find it in the models directory
        # Since we're running from atari/baselines/, go up one level to reach atari/
        models_dir = "../models/baselines/lunarlander"
        potential_path = os.path.join(models_dir, args.model)
        if os.path.exists(potential_path):
            args.model = potential_path
        else:
            print(f"Model file not found: {args.model}")
            print("Available models:")
            list_available_models()
            return

    # Test the model
    render = not args.no_render
    test_model(
        args.model, args.algorithm, args.episodes, render, args.policy, args.verbose
    )


if __name__ == "__main__":
    main()
