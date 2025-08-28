from jax import random as jr, numpy as jnp
import jax
import equinox as eqx
from typing import List, Callable, Tuple
from jaxtyping import Float, Array 
from functools import partial



class SpectralConv1d(eqx.Module):
    mul: callable
    mode: int
    out_channels: int
    weights: jax.Array
    ndims: int

    def __init__(self, 
                 mode,
                 in_channels: int,
                 out_channels: int,
                 key,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.mode = mode
        self.ndims = 1
        scale = 1 / (in_channels*out_channels)

        key,key1 = jr.split(key)
        
        self.weights = jr.uniform(key, (2, mode, in_channels, out_channels)) * scale
        self.mul = partial(jnp.einsum, 
                           f'ic,ico->io')
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "x in_channels"]) -> Float[Array, "x out_channels"]: ### x,y,z,channels
        x_ft = jnp.fft.rfftn(x, axes=list(range(self.ndims))) ### no batch dim, of shape x,y,z//2+1,channels
        out_ft = jnp.zeros((*x_ft.shape[:-1], self.out_channels), dtype=jnp.complex64)
        out = self.mul(x_ft[:self.mode], self.weights[0]+self.weights[1]*1j)
        out_ft = out_ft.at[:self.mode,].set(out)
        return jnp.fft.irfftn(out_ft, axes=list(range(self.ndims)))
    
    
class SpectralConv2d(eqx.Module):
    ndims: int
    mul: callable
    modes: List[int]
    out_channels: int
    weights: jax.Array
    ndims: int
    def __init__(self, 
                 modes: List[int], 
                 in_channels: int,
                 out_channels: int,
                 key,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        assert len(modes) == 2
        self.modes = modes
        self.ndims = 2
        scale = 1 / (in_channels*out_channels)
        key,_ = jr.split(key)

        num_weights = (2**(self.ndims-1))
        key,key1 = jr.split(key)

        self.weights = [jr.uniform(k, (2, *modes, in_channels, out_channels)) * scale for k in jr.split(key1, num_weights)]

        self.mul = partial(jnp.einsum, 
                           f'ijc,ijco->ijo')
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "x y in_channels"]) -> Float[Array, "x y out_channels"]: ### x,y,z,channels
        modes = self.modes
        x_ft = jnp.fft.rfftn(x, axes=list(range(self.ndims))) ### no batch dim, of shape x,y,z//2+1,channels

        out_ft = jnp.zeros((*x_ft.shape[:-1], self.out_channels), dtype=jnp.complex64)

        out = self.mul(x_ft[:modes[0], :modes[1]], self.weights[0][0]+self.weights[0][1]*1j)
        out_ft = out_ft.at[:modes[0],:modes[1]].set(out)

        out = self.mul(x_ft[-modes[0]:, :modes[1]], self.weights[1][0]+self.weights[1][1]*1j)
        out_ft = out_ft.at[-modes[0]:,:modes[1]].set(out)
        
        return jnp.fft.irfftn(out_ft, s=(x.shape[0], x.shape[1]), axes=list(range(self.ndims)))
        
        
class SpectralConv3d(eqx.Module):
    ndims: int
    mul: callable
    modes: List[int]
    out_channels: int
    weights: jax.Array
    ndims: int
    def __init__(self, 
                modes: List[int], 
                in_channels: int,
                out_channels: int,
                key,
                **kwargs
                ):
        super().__init__(**kwargs)
        assert len(modes) == 3
        self.modes = modes
        self.ndims = 3
        scale = 1 / (in_channels*out_channels)
        key,_ = jr.split(key)

        scale = 1 / (in_channels*out_channels)
        key,_ = jr.split(key)

        num_weights = (2**(self.ndims-1))
        key,key1 = jr.split(key)

        self.weights = [jr.uniform(k, (2, *modes, in_channels, out_channels)) * scale for k in jr.split(key1, num_weights)]
        

        self.mul = partial(jnp.einsum, 
                        f'ijkc,ijkco->ijko')
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "x y z in_channels"]) -> Float[Array, "x y z in_channels"]: ### x,y,z,channels
        modes = self.modes
        x_ft = jnp.fft.rfftn(x, axes=list(range(self.ndims))) ### no batch dim, of shape x,y,z//2+1,channels
        out_ft = jnp.zeros((*x_ft.shape[:-1], self.out_channels), dtype=jnp.complex64)

        out = self.mul(x_ft[:modes[0], :modes[1], :modes[2]], self.weights[0][0]+self.weights[0][1]*1j)
        out_ft = out_ft.at[:modes[0],:modes[1], :modes[2]].set(out)

        out = self.mul(x_ft[-modes[0]:, :modes[1], :modes[2]], self.weights[1][0]+self.weights[1][1]*1j)
        out_ft = out_ft.at[-modes[0]:,:modes[1], :modes[2]].set(out)

        out = self.mul(x_ft[:modes[0], -modes[1]:, :modes[2]], self.weights[2][0]+self.weights[2][1]*1j)
        out_ft = out_ft.at[:modes[0],-modes[1]:, :modes[2]].set(out)

        out = self.mul(x_ft[-modes[0]:, -modes[1]:, :modes[2]], self.weights[3][0]+self.weights[3][1]*1j)
        out_ft = out_ft.at[-modes[0]:,-modes[1]:, :modes[2]].set(out)

        return jnp.fft.irfftn(out_ft, s=(x.shape[0], x.shape[1], x.shape[2]), axes=list(range(self.ndims)))