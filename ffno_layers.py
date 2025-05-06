from jax import random as jr, numpy as jnp
import jax
import equinox as eqx
from typing import List, Callable, Tuple
from jaxtyping import Float, Array 
from functools import partial


class SpectralConv2d(eqx.Module):
    ndims: int
    mulx: callable
    muly: callable
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
        key,_ = jr.split(key)
        self.weights = [jax.nn.initializers.xavier_normal()(k, (2, modes[i], in_channels, out_channels)) for i,k in enumerate(jr.split(key, self.ndims))] ### xavier normal? 
        self.mulx = partial(jnp.einsum, 
                           f'ijc,ico->ijo')
        self.muly = partial(jnp.einsum, 
                           f'ijc,jco->ijo')
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "x y in_channels"]) -> Float[Array, "x y out_channels"]: ### x,y,z,channels
        modes = self.modes
        
        ### 
        x_fty = jnp.fft.rfftn(x, axes=(1,), norm='ortho') 
        out_ft = jnp.zeros((*x_fty.shape[:-1], self.out_channels), dtype=jnp.complex64)
        out = self.muly(x_fty[:, :modes[1], :], self.weights[1][0]+self.weights[1][1]*1j)
        out_ft = out_ft.at[:, :modes[1], :].set(out)
        second_dim = jnp.fft.irfftn(out_ft, s=(x.shape[1],), axes=(1,), norm='ortho')
        
        
        x_ftx = jnp.fft.rfftn(x, axes=(0,), norm='ortho') 
        out_ft = jnp.zeros((*x_ftx.shape[:-1], self.out_channels), dtype=jnp.complex64)
        out = self.mulx(x_ftx[:modes[0], :, :], self.weights[0][0]+self.weights[0][1]*1j)
        out_ft = out_ft.at[:modes[0], :, :].set(out)
        first_dim = jnp.fft.irfftn(out_ft, s=(x.shape[0],), axes=(0,), norm='ortho')
        return first_dim + second_dim
    

class SpectralConv3d(eqx.Module):
    ndims: int
    mulx: callable
    muly: callable
    mulz: callable
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
        key,_ = jr.split(key)
        self.weights = [jax.nn.initializers.xavier_normal()(k, (2, modes[i], in_channels, out_channels)) for i,k in enumerate(jr.split(key, self.ndims))] ### xavier normal? 
        self.mulx = partial(jnp.einsum, 
                           f'ijkc,ico->ijko')
        self.muly = partial(jnp.einsum, 
                           f'ijkc,jco->ijko')
        self.mulz = partial(jnp.einsum, 
                           f'ijkc,kco->ijko')
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "x y z in_channels"]) -> Float[Array, "x y z out_channels"]: ### x,y,z,channels
        modes = self.modes
        
        
        x_ftz = jnp.fft.rfftn(x, axes=(2,), norm='ortho') 
        out_ft = jnp.zeros((*x_ftz.shape[:-1], self.out_channels), dtype=jnp.complex64) ### x,y,z,c
        ### 
        out = self.mulz(x_ftz[:, :, :modes[2], :], self.weights[2][0]+self.weights[2][1]*1j)
        out_ft = out_ft.at[:, :, :modes[2], :].set(out)
        third_dim = jnp.fft.irfftn(out_ft, s=(x.shape[2],), axes=(2,), norm='ortho')
        
        
        ### 
        x_fty = jnp.fft.rfftn(x, axes=(1,), norm='ortho') 
        out_ft = jnp.zeros((*x_fty.shape[:-1], self.out_channels), dtype=jnp.complex64)
        out = self.muly(x_fty[:, :modes[1], :, :], self.weights[1][0]+self.weights[1][1]*1j)
        out_ft = out_ft.at[:, :modes[1], :, :].set(out)
        second_dim = jnp.fft.irfftn(out_ft, s=(x.shape[1],), axes=(1,), norm='ortho')
        
        
        x_ftx = jnp.fft.rfftn(x, axes=(0,), norm='ortho') 
        out_ft = jnp.zeros((*x_ftx.shape[:-1], self.out_channels), dtype=jnp.complex64)
        out = self.mulx(x_ftx[:modes[0], :, :, :], self.weights[0][0]+self.weights[0][1]*1j)
        out_ft = out_ft.at[:modes[0], :, :, :].set(out)
        first_dim = jnp.fft.irfftn(out_ft, s=(x.shape[0],), axes=(0,), norm='ortho')
        return first_dim + second_dim + third_dim