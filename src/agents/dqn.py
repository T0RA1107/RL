"""DQN agent implementation using JAX, Flax, and rlax."""
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool
import rlax
from flax.training import train_state
from omegaconf import DictConfig
import pickle

from src.agents.base import BaseAgent
from src.buffers.replay_buffer import ReplayBuffer


class TrainState(train_state.TrainState):
    """Extended TrainState for DQN with target network."""
    target_params: dict[str, jnp.ndarray] = None


class DQNAgent(BaseAgent):
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        discount_gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        learning_starts: int = 1000,
        target_update_frequency: int = 100,
        seed: int = 42,
    ):
        """Initialize DQN agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of discrete actions
            network_cfg: Configuration for Q-network
            optimizer_cfg: Configuration for optimizer
            discount_gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            buffer_size: Replay buffer size
            learning_starts: Steps before starting training
            target_update_frequency: Steps between target network updates
            seed: Random seed
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount_gamma = discount_gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_frequency = target_update_frequency

        # Initialize RNG
        self.rng = jax.random.PRNGKey(seed)

        # Create replay buffer
        self.buffer = ReplayBuffer(buffer_size, observation_dim)

        # Initialize networks
        self.rng, network_rng = jax.random.split(self.rng)
        self.q_network = hydra.utils.instantiate(network_cfg, action_dim=action_dim)

        # Initialize network parameters
        dummy_obs = jnp.zeros((1, observation_dim))
        params = self.q_network.init(network_rng, dummy_obs)

        # Create optimizer
        optimizer = hydra.utils.instantiate(optimizer_cfg)

        # Create train state with target network
        self.state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=params,
            target_params=params,
            tx=optimizer,
        )

        # Training step counter
        self.training_steps = 0

    def select_action(
        self,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = True
    ):
        """Select action using epsilon-greedy policy.

        Args:
            observation: Current observation
            rng: JAX random number generator key
            training: Whether in training mode

        Returns:
            Tuple of (action, info_dict, updated_rng)
        """
        rng, epsilon_rng, action_rng = jax.random.split(rng, 3)

        if training and jax.random.uniform(epsilon_rng) < self.epsilon:
            # Random action (exploration)
            action = jax.random.randint(action_rng, (), 0, self.action_dim)
        else:
            # Greedy action (exploitation)
            obs_batch = observation[None, ...]  # Add batch dimension
            q_values = self.state.apply_fn(self.state.params, obs_batch)
            action = jnp.argmax(q_values[0])

        return int(action), None, rng

    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """Update Q-network using a batch of transitions.

        Args:
            batch: Dictionary containing transitions

        Returns:
            Dictionary of training metrics
        """
        # Convert numpy arrays to JAX arrays
        observations = jnp.array(batch["observations"])
        actions = jnp.array(batch["actions"])
        rewards = jnp.array(batch["rewards"])
        next_observations = jnp.array(batch["next_observations"])
        dones = jnp.array(batch["dones"])

        # Update Q-network
        self.state, loss, q_values = self._update_step(
            self.state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_frequency == 0:
            self.state = self.state.replace(target_params=self.state.params)

        return {
            "loss/total": float(loss),
            "mean_q_value": float(q_values.mean()),
            "epsilon": self.epsilon,
        }

    @staticmethod
    @jax.jit
    def _update_step(
        state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
        """JIT-compiled training step.

        Args:
            state: Current training state
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags

        Returns:
            Tuple of (updated_state, loss, q_values)
        """
        def loss_fn(params):
            # Compute Q-values for current observations
            q_values = state.apply_fn(params, observations)

            # Compute target Q-values using target network
            target_q_values = state.apply_fn(state.target_params, next_observations)

            # Use rlax q_learning for TD error computation (vectorized)
            td_errors = jax.vmap(rlax.q_learning)(
                q_tm1=q_values,
                a_tm1=actions,
                r_t=rewards,
                discount_t=(1.0 - dones) * 0.99,
                q_t=target_q_values
            )

            # Mean squared TD error
            loss = jnp.mean(td_errors ** 2)

            return loss, q_values

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, q_values), grads = grad_fn(state.params)

        # Apply gradients
        state = state.apply_gradients(grads=grads)

        return state, loss, q_values

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save agent parameters to file.

        Args:
            path: File path to save parameters
        """
        checkpoint = {
            "params": self.state.params,
            "target_params": self.state.target_params,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str) -> None:
        """Load agent parameters from file.

        Args:
            path: File path to load parameters from
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.state = self.state.replace(
            params=checkpoint["params"],
            target_params=checkpoint["target_params"],
        )
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
