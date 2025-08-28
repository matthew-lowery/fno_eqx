
from jax import random as jr
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import List
from jaxtyping import Float, Array
from fno_layers import *


class FNO(eqx.Module):
    spectral_layers: List[eqx.Module]
    pointwise_layers: List[eqx.Module]
    lift_layer: eqx.Module
    proj_layers: List[eqx.Module]
    activation: Callable
    transposes: List[List[int]]
    lift_dim: int
    ndims: int

    def __init__(self, 
                 modes: List[int], ## length of this list specifies dimension? 
                 lift_dim: int,
                 activation: Callable,
                 depth: int,
                 in_feats: int,
                 *,
                 key,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.activation = activation

        keys = jr.split(key, depth)
        ndims = len(modes)

        if ndims == 1:
            self.spectral_layers = [SpectralConv1d(modes[0], lift_dim, lift_dim, key=key) for key in keys]
        elif ndims == 2:
            self.spectral_layers = [SpectralConv2d(modes, lift_dim, lift_dim, key=key) for key in keys]
        elif ndims == 3:
            self.spectral_layers = [SpectralConv3d(modes, lift_dim, lift_dim, key=key) for key in keys]
        else:
            raise 'spectral conv not implemented for dimensions > 3'
        
        key,_ = jr.split(keys[0])
        self.lift_layer =  eqx.nn.Linear(in_feats+ndims, lift_dim, key=key)

        keys = jr.split(key)
        self.proj_layers = [eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]), eqx.nn.Linear(lift_dim, 1, key=keys[1])]

        keys = jr.split(keys[0], depth)
        self.pointwise_layers = [eqx.nn.Conv(ndims, lift_dim, lift_dim, 1, key=key) for key in keys]

        ### move channel dim back and forth
        self.transposes = [[ndims] + list(range(ndims)), list(range(1,ndims+1)) + [0]]

        self.lift_dim = lift_dim
        self.ndims = ndims
        
    def __call__(self, 
                 f_x:  Float[Array, "x_1 x_2 x_ndims 1"], 
                 x_grid: Float[Array, "x_1 x_2 x_ndims ndims"],
                 ) -> Float[Array, "x_1 x_2 x_ndims 1"]:
        f_x = jnp.concatenate((f_x, x_grid), axis=-1)
        f_x = jax.vmap(self.lift_layer)(f_x.reshape(-1,f_x.shape[-1]))
        f_x = f_x.reshape(*x_grid.shape[:self.ndims],self.lift_dim)

        for i in range(len(self.spectral_layers[:-1])):

            ### conv wants channel dim first
            f_x_prev = self.pointwise_layers[i](f_x.transpose(self.transposes[0])).transpose(self.transposes[1])

            f_x = self.spectral_layers[i](f_x)
            f_x = self.activation(f_x_prev + f_x)

        f_x_prev = self.pointwise_layers[-1](f_x.transpose(self.transposes[0])).transpose(self.transposes[1])
        f_x = f_x_prev + self.spectral_layers[-1](f_x)
        
        
        f_x = self.activation(jax.vmap(self.proj_layers[0])(f_x.reshape(-1,f_x.shape[-1])))
        f_x = jax.vmap(self.proj_layers[1])(f_x)
        f_x = f_x.reshape(*x_grid.shape[:self.ndims], 1)
        return f_x