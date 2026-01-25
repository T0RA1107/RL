"""Experience replay buffer for off-policy RL algorithms."""
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool

from .base import BaseBuffer


class ReplayBuffer(BaseBuffer):
    """Simple experience replay buffer for storing and sampling transitions."""

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
    ):
        """Initialize replay buffer.

        Args:
            n_env: Number of parallel environments
            buffer_size: Maximum number of transitions to store
            observation_dim: Observation dimension
            action_dim: Action dimension
        """
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Preallocate arrays for efficiency
        self.observations = jnp.zeros((n_env, buffer_size, observation_dim))
        self.actions = jnp.zeros((n_env, buffer_size, action_dim))
        self.rewards = jnp.zeros((n_env, buffer_size))
        self.next_observations = jnp.zeros((n_env, buffer_size, observation_dim))
        self.dones = jnp.zeros((n_env, buffer_size))

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size

    def add(
        self,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
        reward: Float[Array, " n_env"],
        next_observation: Float[Array, "n_env ..."],
        done: Bool[Array, " n_env"],
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.observations = self.observations.at[:, self.position].set(observation)
        self.actions = self.actions.at[:, self.position].set(action)
        self.rewards = self.rewards.at[:, self.position].set(reward)
        self.next_observations = self.next_observations.at[:, self.position].set(next_observation)
        self.dones = self.dones.at[:, self.position].set(done)

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batch of transitions
        """
        indices = jnp.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[:, indices],
            "actions": self.actions[:, indices],
            "rewards": self.rewards[:, indices],
            "next_observations": self.next_observations[:, indices],
            "dones": self.dones[:, indices],
        }

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples.

        Args:
            min_size: Minimum number of samples required

        Returns:
            True if buffer has at least min_size samples
        """
        return self.size >= min_size
