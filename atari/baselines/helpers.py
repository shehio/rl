# helpers.py

import gymnasium as gym
import ale_py  # This registers the ALE environments
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os
import time
from stable_baselines3.common.callbacks import BaseCallback

class SaveEveryStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, game_name, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.game_name = game_name

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/{self.game_name}_step_{self.n_calls}")
        return True

def create_atari_environment(name: str, render: bool, render_fps = 60):
    render_mode = 'human' if render else None
    env = gym.make(name, render_mode=render_mode)
    env = AtariWrapper(env)
    # if render:
    #     env.metadata['render_fps'] = render_fps
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def create_environment(name: str, render: bool, render_fps = 60):
    render_mode = 'human' if render else None
    env = gym.make(name, render_mode=render_mode)
    if render:
        env.metadata['render_fps'] = render_fps
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def train(model, timesteps, epochs, tensorboard_log_name, model_directory, callback=None):
    start_time = time.time()
    for i in range(epochs):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=tensorboard_log_name, callback=callback)
        model.save(f"{model_directory}/{timesteps * i}")

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60} minutes")
    return model

def render_model(env, model, timestep: int, frame_skip: int = 10):
    obs = env.reset()
    for i in range(timestep):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i % frame_skip == 0:
            env.render()
    env.close()

def create_directory_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
