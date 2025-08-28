### 
from jaxtyping import Float, Array
from typing import Tuple
from jax import numpy as jnp, jit, random as jr
from functools import partial
import optax
import numpy as np
import equinox as eqx

DTYPE=jnp.float32


def is_trainable(x):
    return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

def shuffle(x,y, seed=1):
    np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    return x,y


def create_lifted_module(base_layer, lift_dim, key):
    keys = jr.split(key, lift_dim)
    return eqx.filter_vmap(lambda key: base_layer(key=key))(keys)

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = jnp.mean(x, axis=0)
        self.std = jnp.std(x, axis=0)
        self.eps = eps

    @partial(jit, static_argnums=(0,))
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    @partial(jit, static_argnums=(0,))
    def decode(self, x):
        std = self.std + self.eps  # n
        mean = self.mean
        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x
    

def cosine_annealing(
    total_steps,
    init_value=1e-4,
    warmup_frac=0.3,
    peak_value=3e-4,
    end_value=1e-4,
    num_cycles=6,
    gamma=0.9,
):
    decay_steps = total_steps / num_cycles
    schedules = []
    boundaries = []
    boundary = 0
    for cycle in range(num_cycles):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            warmup_steps=decay_steps * warmup_frac,
            peak_value=peak_value,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=2,
        )
        boundary = decay_steps + boundary
        boundaries.append(boundary)
        init_value = end_value
        peak_value = peak_value * gamma
        schedules.append(schedule)
        
    return optax.join_schedules(schedules=schedules, boundaries=boundaries)
