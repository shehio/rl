# Policy Gradient (REINFORCE) Implementation

This directory contains a custom implementation of Policy Gradient (REINFORCE) algorithm for Atari games.

> **📚 For general Atari information, see [atari/README.md](../README.md)**

## Structure

```
pg/
├── scripts/               # Training scripts
│   ├── pg_trainer.py     # Generic PG trainer (Pong, Breakout, Ms. Pacman)
│   ├── pg_tester.py      # Test trained PG models
│   └── game_configs.py   # Game-specific configuration
└── src/                   # Source code
    ├── agent.py          # PG agent implementation
    ├── game.py           # Game environment wrapper
    ├── memory.py         # Episode memory buffer
    ├── mlp_torch.py      # PyTorch MLP implementation
    ├── mlp.py            # NumPy MLP (legacy)
    ├── hyperparameters.py # Training hyperparameters
    └── pacman/           # Pacman-specific implementations
        ├── multi_action_agent.py
        ├── cnn_torch_multiaction.py
        ├── mlp_torch_multiaction.py
        └── preprocess_pacman.py
```

## 🚀 Quick Start

```bash
# Train on Pong (binary actions)
python scripts/pg_trainer.py pong

# Train on Breakout (binary actions)
python scripts/pg_trainer.py breakout

# Train on Ms. Pacman (9 actions)
python scripts/pg_trainer.py pacman

# Train with rendering enabled
python scripts/pg_trainer.py pong --render

# Test trained model (loads the latest checkpoint by default)
python scripts/pg_tester.py pong --load-episode 90000
```

## 🧠 Policy Gradient Features

- **REINFORCE Algorithm**: Direct policy optimization
- **PyTorch Implementation**: Modern MLP with GPU support
- **CNN Support**: Convolutional networks for Pacman
- **Episode Memory**: Stores complete episodes for training
- **Multi-Action Support**: 9-action space for Pacman
- **Frame Preprocessing**: Optimized image processing

## Architecture

### MLP Architecture (Pong/Breakout)
- **Input**: 6400 units (80x80 preprocessed frames)
- **Hidden**: 200 units with ReLU activation
- **Output**: 1 unit with Sigmoid activation
- **Policy**: Binary action selection (UP/DOWN or LEFT/RIGHT)

### CNN Architecture (Pacman)
- **Input**: 7 channels at 80x80 resolution
- **Conv1**: 32 filters, 8x8 kernel, stride 4
- **Conv2**: 64 filters, 4x4 kernel, stride 2
- **Conv3**: 64 filters, 3x3 kernel, stride 1
- **Fully Connected**: 200 hidden units
- **Output**: 9 action probabilities

## Configuration

Key hyperparameters in `src/hyperparameters.py`:
```python
# Training
learning_rate = 1e-3
batch_size = 10
max_episodes = 100000

# Network
input_size = 6400
hidden_size = 200
output_size = 1  # or 9 for Pacman

# Exploration
temperature = 1.0
```

## 📊 Performance

- **Pong**: 20+ average reward after 70k episodes
- **Breakout**: Good paddle control and ball tracking
- **Pacman**: Complex maze navigation with ghost avoidance
- **Training Time**: 2-4 hours for 100k episodes on CPU

## 💾 Model Storage

Trained models are saved in:
- `../../models/pg/pong/` for Pong models
- `../../models/pg/breakout/` for Breakout models
- `../../models/pg/pacman/` for Pacman models

### Model Naming Convention
```
{base_name}_{game_name}_i{input_size}_h{hidden_size}_o{output_size}_{episode}
```

## 🔧 Dependencies

```bash
pip install torch numpy gymnasium[atari] ale-py
```

## 📈 Training Process

1. **Episode Collection**: Run complete episodes with current policy
2. **Reward Calculation**: Compute discounted rewards for each step
3. **Policy Update**: Update network weights using REINFORCE
4. **Exploration**: Use temperature scheduling for exploration
5. **Model Saving**: Save checkpoints every N episodes

## Game-Specific Implementations

### Pong & Breakout
- **Action Space**: Binary (UP/DOWN for Pong, LEFT/RIGHT for Breakout)
- **Network**: MLP with 6400 input units, 200 hidden units
- **Policy**: Sigmoid output for binary action selection

### Ms. Pacman
- **Action Space**: 9 discrete actions (8 directions + NOOP)
- **Network**: CNN with 7 input channels (color masks + grayscale)
- **Policy**: Softmax output for multi-action selection
- **Preprocessing**: Color-aware preprocessing for ghost detection

## 🔍 Key Differences from DQN

| Aspect | Policy Gradient | DQN |
|--------|----------------|-----|
| **Learning Type** | On-policy | Off-policy |
| **Action Space** | Both | Discrete only |
| **Policy Type** | Stochastic | Deterministic |
| **Sample Efficiency** | Lower | Higher |
| **Training Stability** | Less stable | More stable |
| **Implementation** | Simpler | More complex |
| **Best For** | Continuous actions | Discrete actions |

## 🎯 REINFORCE Algorithm

The REINFORCE algorithm directly optimizes the policy:

1. **Collect Episode**: Run policy π(s) for complete episode
2. **Compute Returns**: Calculate discounted rewards R(t)
3. **Update Policy**: ∇θ J(θ) = E[∇θ log π(a|s) R(t)]
4. **Repeat**: Continue until convergence

## 🔗 Related Documentation

- **[atari/README.md](../README.md)** - General Atari information
- **[dqn/README.md](../dqn/README.md)** - DQN comparison
- **[baselines/README.md](../baselines/README.md)** - Stable Baselines3 A2C
