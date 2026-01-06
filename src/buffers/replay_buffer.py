"""Experience replay buffer for off-policy RL algorithms."""
import numpy as np
from typing import Tuple, Dict


class ReplayBuffer:
    """Simple experience replay buffer for storing and sampling transitions."""

    def __init__(self, buffer_size: int, observation_dim: int):
        """Initialize replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            observation_dim: Dimension of observation space
        """
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.position = 0
        self.size = 0

        # Preallocate arrays for efficiency
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batch of transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer has at least min_size samples
        """
        return self.size >= min_size
