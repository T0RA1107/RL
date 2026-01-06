"""Gymnax environment wrapper for consistent interface."""
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
from typing import Tuple, Any, Union


class GymnaxWrapper:
    """Wrapper for Gymnax environments providing a consistent interface.

    Supports both discrete and continuous action spaces.
    """

    def __init__(self, env_name: str, env_kwargs: dict = None):
        """Initialize the Gymnax environment wrapper.

        Args:
            env_name: Name of the Gymnax environment (e.g., "CartPole-v1")
            env_kwargs: Additional keyword arguments for environment creation
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}

        # Create the Gymnax environment
        self.env, self.env_params = gymnax.make(env_name, **self.env_kwargs)

    def reset(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Reset the environment.

        Args:
            rng: JAX random number generator key

        Returns:
            Tuple of (observation, environment_state)
        """
        obs, state = self.env.reset(rng, self.env_params)
        return obs, state

    def step(
        self,
        state: Any,
        action: Union[int, jnp.ndarray],
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Any, float, bool, dict]:
        """Execute one step in the environment.

        Args:
            state: Current environment state
            action: Action to take (int for discrete, jnp.ndarray for continuous)
            rng: JAX random number generator key

        Returns:
            Tuple of (next_observation, next_state, reward, done, info)
        """
        next_obs, next_state, reward, done, info = self.env.step(
            rng, state, action, self.env_params
        )
        return next_obs, next_state, reward, done, info

    def get_observation_space(self) -> int:
        """Get the observation space dimension.

        Returns:
            Dimension of observation space
        """
        return self.env.observation_space(self.env_params).shape[0]

    def get_action_space(self) -> int:
        """Get the action space dimension.

        Returns:
            Number of discrete actions
        """
        return self.env.action_space(self.env_params).n

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""
        return self.get_observation_space()

    @property
    def is_continuous_action(self) -> bool:
        """Check if action space is continuous."""
        return isinstance(self.env.action_space(self.env_params), gymnax.environments.spaces.Box)

    @property
    def action_dim(self) -> int:
        """Get action dimension (for continuous action spaces).

        Returns:
            Action dimension
        """
        if self.is_continuous_action:
            space = self.env.action_space(self.env_params)
            return int(np.prod(space.shape))
        else:
            return self.get_action_space()

    @property
    def action_low(self) -> jnp.ndarray:
        """Get lower bounds for continuous actions.

        Returns:
            Lower bounds as JAX array

        Raises:
            TypeError: If action space is not continuous
        """
        if self.is_continuous_action:
            low = jnp.array(self.env.action_space(self.env_params).low)
            # Ensure it's at least 1D
            if low.ndim == 0:
                low = jnp.array([low])
            return low
        raise TypeError("action_low is only for continuous action spaces")

    @property
    def action_high(self) -> jnp.ndarray:
        """Get upper bounds for continuous actions.

        Returns:
            Upper bounds as JAX array

        Raises:
            TypeError: If action space is not continuous
        """
        if self.is_continuous_action:
            high = jnp.array(self.env.action_space(self.env_params).high)
            # Ensure it's at least 1D
            if high.ndim == 0:
                high = jnp.array([high])
            return high
        raise TypeError("action_high is only for continuous action spaces")

    def scale_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Scale action from [-1, 1] to environment's action bounds.

        Args:
            action: Action in range [-1, 1]

        Returns:
            Scaled action within environment's action bounds
        """
        if not self.is_continuous_action:
            raise TypeError("scale_action is only for continuous action spaces")

        low = self.action_low
        high = self.action_high
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        return jnp.clip(scaled_action, low, high)

    def normalize_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Normalize action from environment's action bounds to [-1, 1].

        Args:
            action: Action within environment's action bounds

        Returns:
            Normalized action in range [-1, 1]
        """
        if not self.is_continuous_action:
            raise TypeError("normalize_action is only for continuous action spaces")

        low = self.action_low
        high = self.action_high
        normalized_action = 2.0 * (action - low) / (high - low) - 1.0
        return jnp.clip(normalized_action, -1.0, 1.0)
