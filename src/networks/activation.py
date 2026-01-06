import jax
import jax.numpy as jnp
import flax.linen as nn


@jax.jit
def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation function.

    Args:
        x: Input array

    Returns:
        Activated array
    """
    return x * jnp.tanh(nn.softplus(x))
