# DQN (Deep Q-Network) Implementation

This directory contains a custom implementation of Deep Q-Network (DQN) for Atari games.

> **📚 For general Atari information, see [atari/README.md](../README.md)**

## Structure

```
dqn/
├── scripts/           # Training scripts
│   ├── pong-dqn.py    # Train DQN on Pong
│   ├── pacman-dqn.py  # Train DQN on Ms. Pacman
│   ├── dqn_trainer.py # Generic DQN trainer
│   └── dqn_tester.py  # Test trained DQN models
└── src/               # Source code
    ├── agent.py       # DQN agent implementation
    ├── model.py       # Neural network architecture
    └── config/        # Configuration classes
        ├── environment_config.py
        ├── model_config.py
        ├── training_config.py
        ├── learning_config.py
        ├── exploration_config.py
        └── image_config.py
```

## 🚀 Quick Start

```bash
# Train on Pong
python scripts/pong-dqn.py

# Train on Ms. Pacman
python scripts/pacman-dqn.py

# Use generic trainer
python scripts/dqn_trainer.py pong

# Test trained model (loads the latest checkpoint by default)
python scripts/dqn_tester.py pong --load-episode 900
```

## 🧠 DQN Features

- **Dueling CNN Architecture**: Separate value and advantage streams
- **Experience Replay**: Stores and samples past experiences
- **Target Networks**: Stabilizes training with separate target network
- **Frame Stacking**: Uses 4 consecutive frames as state representation
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation

## Architecture

The DQN uses a Dueling CNN architecture:
1. **Convolutional Layers**: Feature extraction from game pixels
2. **Dueling Streams**: Separate value and advantage computation
3. **Fully Connected**: Final Q-value computation
4. **Output**: Q-values for each action

### Network Details
- **Input**: 80x80 preprocessed grayscale frames
- **Conv1**: 32 filters, 8x8 kernel, stride 4
- **Conv2**: 64 filters, 4x4 kernel, stride 2
- **Conv3**: 64 filters, 3x3 kernel, stride 1
- **Fully Connected**: 512 hidden units
- **Output**: Q-values for each action

## Configuration

The DQN implementation uses a modular configuration system:

- **Environment Config**: Game-specific settings
- **Model Config**: Neural network architecture
- **Training Config**: Learning parameters
- **Exploration Config**: Epsilon-greedy settings
- **Image Config**: Frame preprocessing

### Key Hyperparameters
```python
# Learning
learning_rate = 0.00025
gamma = 0.99  # Discount factor

# Experience Replay
replay_buffer_size = 100000
batch_size = 32

# Exploration
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Target Network
target_update_freq = 1000
```

## 📊 Performance

- **Pong**: 21+ average reward after 1M timesteps
- **Pacman**: Complex maze navigation with ghost avoidance
- **Training Time**: 1-2 hours for 1M timesteps on GPU
- **Memory Usage**: ~2GB for replay buffer

## 💾 Model Storage

Trained models are saved in:
- `../../models/dqn/pong/` for Pong models
- `../../models/dqn/pacman/` for Pacman models

### File Types
- **`.pkl`**: Model weights
- **`.json`**: Epsilon values and training state
- **`.txt`**: Training logs and performance metrics

## 🔧 Dependencies

```bash
pip install torch numpy gymnasium[atari] ale-py
```

## 📈 Training Process

1. **Environment Setup**: Preprocess frames to 80x80 grayscale
2. **Experience Collection**: Store (state, action, reward, next_state, done) tuples
3. **Training Loop**: Sample batches and update Q-network
4. **Target Network**: Periodically update target network for stability
5. **Exploration**: Gradually reduce epsilon for exploitation

## Game-Specific Implementations

### Pong
- **Action Space**: 2 actions (UP/DOWN)
- **Reward**: +1 for winning, -1 for losing
- **Preprocessing**: Frame differencing for motion detection

### Ms. Pacman
- **Action Space**: 9 actions (8 directions + NOOP)
- **Reward**: Points for eating dots, power pellets, ghosts
- **Preprocessing**: Color-aware processing for ghost detection

## 🔍 Key Differences from Policy Gradient

| Aspect | DQN | Policy Gradient |
|--------|-----|-----------------|
| **Learning Type** | Off-policy | On-policy |
| **Action Space** | Discrete only | Both |
| **Policy Type** | Deterministic | Stochastic |
| **Sample Efficiency** | Higher (replay) | Lower |
| **Training Stability** | More stable | Less stable |
| **Implementation** | More complex | Simpler |

## 🔗 Related Documentation

- **[atari/README.md](../README.md)** - General Atari information
- **[baselines/README.md](../baselines/README.md)** - Stable Baselines3 DQN
- **[pg/README.md](../pg/README.md)** - Policy Gradient comparison
