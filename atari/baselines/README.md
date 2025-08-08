# Stable Baselines3 Implementation

This directory contains implementations using Stable Baselines3 for Atari games, providing production-ready reinforcement learning algorithms.

> **📚 For general Atari information, see [atari/README.md](../README.md)**

## 🚀 Quick Start

```bash
# Train on any Atari game
python atari_baseline_train.py --algorithm ppo --env ALE/Pong-v5 --timesteps 1000000

# Test trained model with rendering
python atari_baseline_test.py --model pong_ppo_cnn_1000000

# Record gameplay video
python atari_baseline_test.py --model pong_dqn_cnn_6000000 --record

# Train with infinite training (Ctrl+C to stop)
python atari_baseline_train.py --algorithm dqn --env ALE/Breakout-v5 --timesteps infinite
```

## Files

- `atari_baseline_train.py` - Generic training script for all algorithms
- `atari_baseline_test.py` - Testing script with rendering
- `helpers.py` - Training utilities and callbacks
- `README.md` - This documentation

## Command Line Arguments

### Training Script (`atari_baseline_train.py`)

```bash
python atari_baseline_train.py [OPTIONS]

Options:
  --algorithm, -a    Algorithm: ppo, dqn, a2c, a2c_cnn [default: ppo]
  --timesteps, -t    Training timesteps or "infinite" [default: 100000]
  --env, -e          Environment name [default: ALE/Pong-v5]
  --n_envs, -n       Number of parallel environments [default: 1]
  --seed, -s         Random seed [default: 0]
```

### Testing Script (`atari_baseline_test.py`)

```bash
python atari_baseline_test.py [OPTIONS]

Options:
  --model, -m        Model path (without .zip extension) [required]
  --env, -e          Environment name [default: ALE/Pong-v5]
  --episodes, -n     Number of episodes to run [default: 3]
  --algorithm, -a    Algorithm type: auto, ppo, dqn, a2c [default: auto]
  --delay, -d        Frame delay in seconds [default: 0.01]
  --record, -r       Record video of gameplay [default: False]
  --video_path, -v   Custom video output path [default: auto-generated]
```

## Environment Options

### Popular Atari Games
```bash
# Classic games
ALE/Pong-v5          # Pong (2 actions)
ALE/Breakout-v5      # Breakout (4 actions)
ALE/MsPacman-v5      # Ms. Pacman (9 actions)
ALE/SpaceInvaders-v5 # Space Invaders (6 actions)
ALE/Asteroids-v5     # Asteroids (18 actions)

# Other games
ALE/Boxing-v5        # Boxing (18 actions)
ALE/Enduro-v5        # Enduro (9 actions)
ALE/Seaquest-v5      # Seaquest (18 actions)
```

### Environment Configuration
- **Frame Stacking**: Automatically stacks 4 frames
- **No Frame Skip**: Uses `frameskip=1` for precise control
- **Repeat Action Probability**: Set to 0.0 for deterministic actions
- **Rendering**: Available in test mode with `render_mode="human"`

## 📊 Training Modes

### Finite Training
```bash
# Train for specific number of timesteps
python atari_baseline_train.py --algorithm ppo --timesteps 1000000
```

### Infinite Training
```bash
# Train continuously with periodic saving
python atari_baseline_train.py --algorithm ppo --timesteps infinite

# Press Ctrl+C to stop and save final model
```

### Model Continuation
```bash
# Automatically loads latest model and continues training
python atari_baseline_train.py --algorithm ppo --timesteps 500000
```

## 💾 Model Management

### Automatic Model Loading
The training script automatically:
1. Searches for existing models in `../models/baselines/`
2. Loads the latest model by timesteps
3. Continues training from where it left off

### Model Naming Convention
```
{game}_{algorithm}_{policy}_{timesteps}.zip

Examples:
pong_ppo_cnn_1000000.zip
breakout_dqn_cnn_500000.zip
pacman_a2c_cnn_200000.zip
```

### Model Storage
- **Location**: `../models/baselines/`
- **Format**: `.zip` files (Stable Baselines3 format)
- **Contents**: Model weights, hyperparameters, training state

## 📈 Performance Monitoring

### Training Metrics
- **Episode Rewards**: Average, best, worst
- **Training Loss**: Policy and value losses
- **Exploration**: Epsilon values (for DQN)
- **Timesteps**: Progress tracking

### Testing Metrics
```bash
# Run multiple episodes and get statistics
python atari_baseline_test.py --model pong_ppo_cnn_1000000 --episodes 10

# Output includes:
# - Average reward
# - Best episode reward
# - Worst episode reward
# - Standard deviation
```

### Video Recording
```bash
# Record gameplay with auto-generated filename
python atari_baseline_test.py --model pong_dqn_cnn_6000000 --record

# Record with custom filename
python atari_baseline_test.py --model pong_dqn_cnn_6000000 --record --video_path my_gameplay.mp4

# Record longer gameplay with slower playback
python atari_baseline_test.py --model pong_dqn_cnn_6000000 --record --episodes 5 --delay 0.05

# Video specifications:
# - Format: MP4
# - Frame rate: 30 FPS
# - Resolution: Auto-detected from environment
# - Codec: H.264 (mp4v)
# - Location: Current directory (atari/baselines/)
```

## 🔧 Advanced Configuration

### Parallel Environments
```bash
# Train with multiple parallel environments (faster)
python atari_baseline_train.py --n_envs 4 --algorithm ppo
```

### Custom Seeds
```bash
# Reproducible training
python atari_baseline_train.py --seed 42 --algorithm dqn
```

### Algorithm-Specific Settings
```python
# PPO settings (in helpers.py)
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# DQN settings
DQN_CONFIG = {
    "learning_rate": 1e-4,
    "buffer_size": 1000000,
    "learning_starts": 50000,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}
```

## 🎯 Best Practices

### Algorithm Selection
- **PPO**: Best all-around choice, stable training
- **DQN**: Good for discrete actions, experience replay
- **A2C**: Fast training, good for simple environments

### Training Tips
1. **Start Small**: Use 100k timesteps for testing
2. **Monitor Progress**: Watch episode rewards
3. **Save Regularly**: Use infinite training for long runs
4. **Test Frequently**: Validate models with rendering

### Performance Optimization
- **GPU Training**: Automatically uses CUDA if available
- **Parallel Environments**: Use `--n_envs 4` for faster training
- **Memory Management**: Monitor GPU memory usage

## 🔍 Troubleshooting

### Common Issues
```bash
# Model not found
Error: Model file not found!
Solution: Check ../models/baselines/ directory

# CUDA out of memory
RuntimeError: CUDA out of memory
Solution: Reduce batch_size or use CPU

# Environment not found
gym.error.UnregisteredEnv
Solution: Install ale-py: pip install ale-py
```

### Performance Issues
- **Slow Training**: Increase `n_envs` or use GPU
- **Poor Results**: Try different algorithms or hyperparameters
- **Memory Issues**: Reduce batch size or buffer size

## 🔗 Related Documentation

- **[atari/README.md](../README.md)** - General Atari information
- **[algorithms/dqn/README.md](../algorithms/dqn/README.md)** - Custom DQN implementation
- **[algorithms/pg/README.md](../algorithms/pg/README.md)** - Custom Policy Gradient implementation
- **[Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)** - Official documentation
