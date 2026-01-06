"""Rollout buffer for on-policy algorithms (A2C)."""
import jax.numpy as jnp
from typing import Dict, Union


class RolloutBuffer:
    """Buffer for storing episode rollouts (on-policy).

    Stores: observations, actions, rewards, next_observations, dones

    """

    def __init__(
        self,
        max_episode_length: int,
        observation_dim: int,
        action_dim: int,
    ):
        """Initialize rollout buffer.

        Args:
            max_episode_length: Maximum episode length
            observation_dim: Observation dimension
            action_dim: Action dimension
        """
        self.max_episode_length = max_episode_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Pre-allocate JAX arrays
        self.observations = jnp.zeros((max_episode_length, observation_dim))
        self.actions = jnp.zeros((max_episode_length, action_dim))
        self.rewards = jnp.zeros(max_episode_length)
        self.next_observations = jnp.zeros((max_episode_length, observation_dim))
        self.dones = jnp.zeros(max_episode_length)

        self.position = 0

    def add(
        self,
        observation: Union[jnp.ndarray, list, tuple],
        action: Union[jnp.ndarray, float],
        reward: float,
        next_observation: Union[jnp.ndarray, list, tuple],
        done: bool
    ):
        """Add a transition to the buffer.

        Args:
            observation: Observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
        """
        self.observations = self.observations.at[self.position].set(jnp.array(observation))
        self.actions = self.actions.at[self.position].set(jnp.array(action))
        self.rewards = self.rewards.at[self.position].set(float(reward))
        self.next_observations = self.next_observations.at[self.position].set(jnp.array(next_observation))
        self.dones = self.dones.at[self.position].set(float(done))

        self.position += 1

    def get_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Get all stored transitions as a batch.

        Args:
            batch_size: Size of the batch to return

        Returns:
            Dictionary with observations, actions, rewards, next_observations, dones
        """
        batch = {
            "observations": self.observations[self.position - batch_size:self.position],
            "actions": self.actions[self.position - batch_size:self.position],
            "rewards": self.rewards[self.position - batch_size:self.position],
            "next_observations": self.next_observations[self.position - batch_size:self.position],
            "dones": self.dones[self.position - batch_size:self.position],
        }
        self.observations = jnp.roll(self.observations, -batch_size, axis=0)
        self.actions = jnp.roll(self.actions, -batch_size, axis=0)
        self.rewards = jnp.roll(self.rewards, -batch_size, axis=0)
        self.next_observations = jnp.roll(self.next_observations, -batch_size, axis=0)
        self.dones = jnp.roll(self.dones, -batch_size, axis=0)
        self.position -= batch_size
        return batch

    def reset(self):
        """Clear buffer for next episode."""
        self.position = 0

    def __len__(self) -> int:
        """Return current number of transitions."""
        return self.position
