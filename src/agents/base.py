"""Base agent abstract class for RL algorithms."""
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from typing import Any, Dict, Tuple, Union


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    @abstractmethod
    def select_action(
        self,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = True
    ) -> Tuple[Union[int, jnp.ndarray], Union[Dict[str, Any], None], jax.random.PRNGKey]:
        """Select an action given an observation.

        Args:
            observation: Current observation
            rng: JAX random number generator key
            training: Whether in training mode (for exploration)

        Returns:
            Tuple of (action, info_dict, updated_rng)
            - action: int for discrete, jnp.ndarray for continuous
            - info_dict: Optional additional info (e.g., log_prob, value)
            - updated_rng: Updated RNG key
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Update agent parameters using a batch of transitions.

        Args:
            batch: Dictionary containing transitions with keys:
                - observations: (batch_size, observation_dim)
                - actions: (batch_size,)
                - rewards: (batch_size,)
                - next_observations: (batch_size, observation_dim)
                - dones: (batch_size,)

        Returns:
            Dictionary of metrics (e.g., loss, q_values)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent parameters to file.

        Args:
            path: File path to save parameters
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent parameters from file.

        Args:
            path: File path to load parameters from
        """
        pass
