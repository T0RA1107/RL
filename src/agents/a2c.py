"""A2C (Advantage Actor-Critic) agent for continuous action spaces."""
import hydra
import jax
import jax.numpy as jnp
import optax
import pickle
from flax.training import train_state
from typing import Dict, Tuple
from omegaconf import DictConfig

from src.agents.base import BaseAgent
from src.buffers.rollout_buffer import RolloutBuffer


class A2CAgent(BaseAgent):
    """A2C agent for continuous action spaces."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        batch_size: int = 64,
        discount_gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        max_episode_length: int = 200,
        seed: int = 42,
    ):
        """Initialize A2C agent.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            network_config: Network configuration
            optimizer_config: Optimizer configuration
            batch_size: Batch size for updates
            discount_gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            max_episode_length: Maximum episode length
            seed: Random seed
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # RNG
        self.rng = jax.random.PRNGKey(seed)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            max_episode_length, observation_dim, action_dim
        )

        # Networks
        self.actor_network = hydra.utils.instantiate(
            network_cfg.actor,
            action_dim=action_dim
        )
        self.critic_network = hydra.utils.instantiate(
            network_cfg.critic
        )

        # Initialize networks
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)
        dummy_obs = jnp.zeros((1, observation_dim))

        actor_params = self.actor_network.init(actor_rng, dummy_obs)
        critic_params = self.critic_network.init(critic_rng, dummy_obs)

        # Optimizers
        actor_tx = hydra.utils.instantiate(optimizer_cfg.actor)
        actor_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            actor_tx,
        )
        critic_tx = hydra.utils.instantiate(optimizer_cfg.critic)
        critic_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            critic_tx,
        )

        # TrainStates
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

    def select_action(
        self,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = True
    ) -> Tuple[jnp.ndarray, None, jax.random.PRNGKey]:
        """Select action from Gaussian policy.

        Args:
            observation: Current observation
            rng: JAX random key
            training: Whether in training mode

        Returns:
            Tuple of (action, None, updated_rng)
        """
        rng, sample_rng = jax.random.split(rng)

        obs_batch = observation[None, ...]

        # Get policy distribution
        mean, log_std = self.actor_state.apply_fn(self.actor_state.params, obs_batch)
        std = jnp.exp(log_std)

        # Sample action
        z = jax.random.normal(sample_rng, mean.shape)
        action = mean + std * z

        return action[0], None, rng

    @staticmethod
    def _gaussian_log_prob(action, mean, log_std):
        """Compute log probability of action under Gaussian.

        Args:
            action: Action taken
            mean: Mean of Gaussian
            log_std: Log standard deviation

        Returns:
            Log probability
        """
        var = jnp.exp(2 * log_std)
        log_prob = -0.5 * (
            jnp.square(action - mean) / var
            + 2 * log_std
            + jnp.log(2 * jnp.pi)
        )
        return jnp.sum(log_prob, axis=-1)

    def update(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Update actor and critic using collected rollout.

        Args:
            batch: Batch with observations, actions, rewards, next_observations, dones

        Returns:
            Dictionary of training metrics
        """
        observations = jnp.array(batch["observations"])
        actions = jnp.array(batch["actions"])
        rewards = jnp.array(batch["rewards"])
        next_observations = jnp.array(batch["next_observations"])
        dones = jnp.array(batch["dones"])

        # Update
        self.actor_state, self.critic_state, losses = self._update_step(
            self.actor_state,
            self.critic_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            self.discount_gamma,
            self.entropy_coef,
            self.value_loss_coef,
        )

        return {
            "loss/total": float(losses["loss"]),
            "loss/actor": float(losses["actor_loss"]),
            "loss/critic": float(losses["critic_loss"]),
            "loss/entropy": float(losses["entropy"]),
        }

    @staticmethod
    @jax.jit
    def _update_step(
        actor_state,
        critic_state,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        discount_gamma,
        entropy_coef,
        value_loss_coef):
        """JIT-compiled update step.

        Args:
            actor_state: Actor training state
            critic_state: Critic training state
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            discount_gamma: Discount factor
            entropy_coef: Entropy coefficient
            value_loss_coef: Value loss coefficient

        Returns:
            Tuple of (updated_actor_state, updated_critic_state, losses)
        """
        # Compute values for current and next states
        values = critic_state.apply_fn(critic_state.params, observations).squeeze()
        next_values = critic_state.apply_fn(critic_state.params, next_observations).squeeze()

        td_target = rewards + discount_gamma * next_values * (1.0 - dones)
        advantages = td_target - values

        def actor_loss_fn(actor_params):
            mean, log_std = actor_state.apply_fn(actor_params, observations)
            log_probs = A2CAgent._gaussian_log_prob(actions, mean, log_std)

            policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages))

            entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))

            actor_loss = policy_loss - entropy_coef * entropy

            return actor_loss, entropy

        def critic_loss_fn(critic_params):
            values = critic_state.apply_fn(critic_params, observations).squeeze()

            critic_loss = jnp.mean(jnp.square(values - jax.lax.stop_gradient(td_target)))

            return critic_loss

        # Actor gradients and update
        (actor_loss, entropy), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        # Critic gradients and update
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        losses = {
            "loss": actor_loss + value_loss_coef * critic_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }

        return actor_state, critic_state, losses

    def decay_epsilon(self):
        """Placeholder for compatibility with BaseAgent."""
        pass

    def save(self, path: str):
        """Save agent parameters.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str):
        """Load agent parameters.

        Args:
            path: Path to load checkpoint from
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.actor_state = self.actor_state.replace(
            params=checkpoint["actor_params"],
        )
        self.critic_state = self.critic_state.replace(
            params=checkpoint["critic_params"],
        )
