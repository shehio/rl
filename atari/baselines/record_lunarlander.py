#!/usr/bin/env python3
"""
Script to automatically record LunarLander gameplay using the most recent model.
Usage: python record_lunarlander.py [--algorithm a2c|ppo|sac|td3]
Example: python record_lunarlander.py --algorithm ppo
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3
import cv2
import numpy as np


def get_algorithm_class(algorithm_name):
    """Get the algorithm class based on name"""
    algorithms = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}
    return algorithms.get(algorithm_name.lower(), A2C)


def find_most_recent_model(algorithm):
    """
    Find the most recent model for the specified algorithm.
    Returns the model path without .zip extension.
    """
    # Since we're running from atari/baselines/, go up one level to reach atari/
    models_dir = f"../models/baselines/lunarlander"

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Pattern to match model files: lunarlander_algorithm_timesteps.zip
    pattern = f"lunarlander_{algorithm}_*.zip"
    model_files = glob.glob(os.path.join(models_dir, pattern))

    if not model_files:
        raise FileNotFoundError(f"No models found for algorithm '{algorithm}'")

    model_timesteps = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        match = re.search(rf"lunarlander_{algorithm}_(\d+)\.zip$", filename)
        if match:
            timesteps = int(match.group(1))
            model_timesteps.append((model_path, timesteps))

    if not model_timesteps:
        raise ValueError(
            f"Could not extract timesteps from model filenames for algorithm '{algorithm}'"
        )

    # Sort by timesteps (descending) and return the most recent
    model_timesteps.sort(key=lambda x: x[1], reverse=True)
    most_recent_model = model_timesteps[0][0]

    # Remove .zip extension
    model_path_without_zip = most_recent_model[:-4]

    print(
        f"Found most recent {algorithm.upper()} model: {os.path.basename(most_recent_model)}"
    )
    print(f"Timesteps: {model_timesteps[0][1]:,}")

    return model_path_without_zip


def extract_timesteps_from_model(model_path):
    """Extract timesteps from model path for naming the output files."""
    filename = os.path.basename(model_path)
    match = re.search(r"_(\d+)", filename)
    if match:
        timesteps = int(match.group(1))
        # Format timesteps for display
        if timesteps >= 1000000:
            millions = timesteps / 1000000
            return f"{millions:.1f}M".replace(".0M", "M")
        elif timesteps >= 1000:
            thousands = timesteps / 1000
            return f"{thousands:.1f}k".replace(".0k", "k")
        else:
            return str(timesteps)
    return "unknown"


def record_lunarlander_gameplay(model_path, algorithm, video_path, episodes=1):
    """Record LunarLander gameplay to video"""
    print(f"Loading {algorithm.upper()} model from: {model_path}")

    # Create environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Load model
    algorithm_class = get_algorithm_class(algorithm)
    model = algorithm_class.load(model_path, env=env)

    # Setup video recording
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (600, 400))

    print(f"Recording LunarLander gameplay to: {video_path}")
    print(f"Running {episodes} episode(s)...")

    total_reward = 0

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

        # Run one episode
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Capture frame
            frame = env.render()
            if frame is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            # Limit episode length
            if steps >= 1000:
                print(f"  Episode {episode + 1} reached maximum steps (1000)")
                break

        total_reward += episode_reward
        print(
            f"  Episode {episode + 1} completed: {steps} steps, reward: {episode_reward:.2f}"
        )

    env.close()
    video_writer.release()

    avg_reward = total_reward / episodes
    print(f"\nAll episodes completed!")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Video saved to: {video_path}")

    return video_path


def convert_to_gif(video_path):
    """
    Convert the recorded video to GIF using ffmpeg.
    """
    if not video_path or not os.path.exists(video_path):
        print("Video file not found, skipping GIF conversion")
        return None

    # Generate GIF filename
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.basename(video_path)
    gif_filename = video_basename.replace(".mp4", ".gif")
    gif_path = os.path.join(video_dir, gif_filename)

    print(f"Converting to GIF: {gif_path}")

    # ffmpeg command to convert to GIF
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=10,scale=320:-1:flags=lanczos",
        gif_path,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = os.system(" ".join(cmd))

    if result != 0:
        print(f"Error converting to GIF (ffmpeg not found or failed)")
        return None

    print("GIF conversion completed successfully!")
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description="Record LunarLander gameplay using the most recent model"
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
        "--episodes",
        "-e",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF conversion",
    )

    args = parser.parse_args()

    try:
        # Find the most recent model
        model_path = find_most_recent_model(args.algorithm)

        # Create assets/videos directory if it doesn't exist
        videos_dir = Path("../../assets/videos")
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        timesteps = extract_timesteps_from_model(model_path)
        video_filename = f"lunarlander_{args.algorithm}_{timesteps}_gameplay.mp4"
        video_path = videos_dir / video_filename

        # Record gameplay
        record_lunarlander_gameplay(
            model_path, args.algorithm, video_path, args.episodes
        )

        # Convert to GIF if requested
        if not args.no_gif:
            gif_path = convert_to_gif(str(video_path))
            if gif_path:
                print(f"\nSuccessfully created gameplay recording!")
                print(f"Video: {video_path}")
                print(f"GIF: {gif_path}")
            else:
                print(f"\nVideo recorded successfully: {video_path}")
                print("GIF conversion failed (ffmpeg may not be installed)")
        else:
            print(f"\nVideo recorded successfully: {video_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
