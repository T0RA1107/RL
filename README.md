# RL Project - JAX/Flax/rlax Reinforcement Learning Framework

A simple, modular reinforcement learning framework built with JAX, Flax, rlax, and Gymnax. Features Hydra configuration management and Weights & Biases integration for experiment tracking.

## Features

- **JAX-native implementation**: Leverages JAX for JIT compilation and automatic differentiation
- **Modular design**: Easy to extend with new algorithms, environments, and network architectures
- **Configuration-driven**: All hyperparameters managed via Hydra YAML files
- **DQN algorithm**: Deep Q-Network with experience replay and target network
- **Gymnax environments**: Fast, JAX-based gym environments (CartPole, etc.)
- **Experiment tracking**: Weights & Biases integration for metrics and visualization
- **Clean codebase**: Well-structured with clear separation of concerns

## Project Structure

```
RL/
├── configs/                      # Hydra configuration files
│   ├── train.yaml               # Main training config
│   ├── environment/             # Environment configs
│   ├── agent/                   # Agent/algorithm configs
│   ├── network/                 # Network architecture configs
│   ├── optimizer/               # Optimizer configs
│   ├── logger/                  # Logger configs
│   └── hydra/                   # Hydra framework configs
├── src/                         # Source code
│   ├── agents/                  # RL agent implementations
│   │   ├── base.py             # Abstract base agent
│   │   └── dqn.py              # DQN agent
│   ├── environments/            # Environment wrappers
│   │   └── gymnax_wrapper.py  # Gymnax wrapper
│   ├── networks/                # Neural network architectures
│   │   └── mlp.py              # MLP Q-network
│   ├── buffers/                 # Replay buffers
│   │   └── replay_buffer.py    # Experience replay
│   ├── utils/                   # Utility functions
│   │   ├── logger.py           # Logging setup
│   │   └── wandb_logger.py     # WandB utilities
│   └── train.py                 # Training orchestration
├── train.py                     # Root entry point
├── pyproject.toml              # Project dependencies
└── README.md                    # This file
```

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Using pip

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Dependencies

- JAX >= 0.8.2
- Flax >= 0.12.2
- rlax >= 0.1.8
- optax >= 0.2.6
- Gymnax >= 0.0.8
- Hydra-core >= 1.3.2
- WandB >= 0.18.11
- Loguru >= 0.7.3
- Chex >= 0.1.86

## Quick Start

### Basic Training

Train DQN on CartPole with default settings:

```bash
python train.py
```

### Configuration Override

Override specific parameters from command line:

```bash
# Train for more episodes
python train.py training.num_episodes=1000

# Change epsilon decay
python train.py agent.epsilon_decay=0.99

# Disable WandB
python train.py wandb.enabled=false

# Change learning rate
python train.py optimizer.learning_rate=5e-4
```

### Using Different Configurations

Switch between config files:

```bash
# Use different environment (when added)
python train.py environment=pendulum

# Use different agent (when added)
python train.py agent=ppo
```

### Hyperparameter Sweeps

Run multiple experiments with Hydra multirun:

```bash
python train.py -m agent.epsilon_decay=0.99,0.995,0.999
```

## Configuration System

The project uses Hydra for hierarchical configuration management. Configs are organized into modular groups:

### Main Config (configs/train.yaml)

Defines the overall training setup:

```yaml
defaults:
  - environment: cartpole
  - agent: dqn
  - network: mlp
  - optimizer: adam
  - logger: default
  - hydra: default

training:
  num_episodes: 500
  max_steps_per_episode: 500
  eval_frequency: 50

wandb:
  enabled: true
  project: "rl-simple-demo"
```

### Environment Config (configs/environment/cartpole.yaml)

Specifies the environment:

```yaml
environment:
  _target_: src.environments.gymnax_wrapper.GymnaxWrapper
  env_name: "CartPole-v1"
```

### Agent Config (configs/agent/dqn.yaml)

Defines algorithm hyperparameters:

```yaml
agent:
  _target_: src.agents.dqn.DQNAgent
  discount_gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  batch_size: 64
  buffer_size: 10000
```

## Weights & Biases Setup

1. Install WandB and login:

```bash
pip install wandb
wandb login
```

2. Set your entity in config:

```yaml
# configs/train.yaml
wandb:
  enabled: true
  entity: "your-username"  # Your WandB username
  project: "rl-simple-demo"
```

3. Or disable WandB:

```bash
python train.py wandb.enabled=false
```

## Adding New Algorithms

The framework is designed for easy extensibility. To add a new algorithm:

### 1. Create Agent Class

Create `src/agents/your_algorithm.py`:

```python
from src.agents.base import BaseAgent

class YourAgent(BaseAgent):
    def __init__(self, observation_dim, num_actions, **kwargs):
        # Initialize your agent
        pass

    def select_action(self, observation, rng, training=True):
        # Implement action selection
        pass

    def update(self, batch):
        # Implement learning update
        pass
```

### 2. Create Config File

Create `configs/agent/your_algorithm.yaml`:

```yaml
agent:
  _target_: src.agents.your_algorithm.YourAgent
  # Your hyperparameters
  learning_rate: 1e-3
  discount: 0.99
```

### 3. Run Training

```bash
python train.py agent=your_algorithm
```

## Adding New Environments

### 1. Create Environment Config

Create `configs/environment/your_env.yaml`:

```yaml
environment:
  _target_: src.environments.gymnax_wrapper.GymnaxWrapper
  env_name: "Pendulum-v1"  # Any Gymnax environment
```

### 2. Run Training

```bash
python train.py environment=your_env
```

## Adding New Network Architectures

### 1. Create Network Module

Create `src/networks/your_network.py`:

```python
from flax import linen as nn

class YourNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        # Define your architecture
        return q_values
```

### 2. Create Config File

Create `configs/network/your_network.yaml`:

```yaml
network:
  _target_: src.networks.your_network.YourNetwork
  # Network-specific params
```

### 3. Reference in Agent Config

```yaml
# configs/agent/dqn.yaml
agent:
  network_config: ${network}
```

## Outputs

Training outputs are organized by Hydra:

```
outputs/
├── YYYY-MM-DD/              # Date
│   └── HH-MM-SS/           # Time
│       ├── .hydra/         # Hydra config snapshots
│       ├── train.log       # Training logs
│       └── checkpoints/    # Model checkpoints
```

## Tips and Best Practices

### JAX-Specific

- **RNG Keys**: Always split RNG keys properly for reproducibility
- **JIT Compilation**: Performance-critical functions are JIT-compiled
- **Pure Functions**: Keep JIT-compiled functions pure (no side effects)

### Training

- Start with default hyperparameters
- Monitor epsilon decay - agent should gradually shift to exploitation
- Check buffer size - ensure it's filling up before major training
- Evaluate periodically to detect overfitting

### Debugging

- Use `wandb.enabled=false` for quick local testing
- Reduce `num_episodes` for faster iteration
- Check logs in `outputs/YYYY-MM-DD/HH-MM-SS/train.log`
- Print resolved config: `python train.py --cfg job`

## Current Implementation

- **Algorithms**: DQN
- **Environments**: CartPole-v1 (via Gymnax)
- **Networks**: MLP Q-Network

## Roadmap

Future improvements:

- [ ] Add more algorithms (PPO, SAC, A2C)
- [ ] Support for more environments
- [ ] CNN networks for visual observations
- [ ] Distributed training
- [ ] Multi-environment training
- [ ] Advanced exploration strategies
- [ ] Model checkpointing with best model selection

## Troubleshooting

### Import Errors

Ensure you're in the project root and virtual environment is activated:

```bash
source .venv/bin/activate
```

### JAX GPU Support

For GPU support, install JAX with CUDA:

```bash
pip install --upgrade "jax[cuda12]"
```

### WandB Authentication

If WandB isn't working:

```bash
wandb login
# Or disable it
python train.py wandb.enabled=false
```

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [rlax Documentation](https://github.com/google-deepmind/rlax)
- [Gymnax Documentation](https://github.com/RobertTLange/gymnax)
- [Hydra Documentation](https://hydra.cc/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## License

This project is open source and available under the MIT License.
