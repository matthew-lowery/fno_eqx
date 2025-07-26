from jax import random as jr, numpy as jnp
import jax
import equinox as eqx
from typing import List,Callable
from utils import create_lifted_module as clm
from functools import partial
import inspect
from abc import ABC, abstractmethod

class KernelBaseClass(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass     

    @abstractmethod
    def eval(self, x, y):
        pass
 
    def __call__(self, 
                x, 
                y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(Y,X)
        return k_xy



class HarmonyMixture(eqx.Module, KernelBaseClass):
    centers: jax.Array
    num_centers: int
    means: jax.Array
    B_LowerT: jax.Array
    
class RationalQuadraticKernel(eqx.Module, KernelBaseClass):
    s1: jax.Array
    s2: jax.Array
    a: jax.Array

    def __init__(self, *, key):
        keys = jr.split(key,3)
        self.s1 = jr.uniform(keys[0],minval=-3.,maxval=0.0)
        self.s2 = jr.uniform(keys[1],minval=-3.,maxval=0.0)
        self.a = jr.uniform(keys[2],minval=-3.,maxval=0.0)

    def eval(self, x, y,):
        a = jax.nn.softplus(self.a)
        s1 = jax.nn.softplus(self.s1)
        s2 = jax.nn.softplus(self.s2)
        r = ((x - y) ** 2).sum()
        return (1 + (r *s1)/a)**-a
    
class GaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, key, **kwargs):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)

    ### one pair of pts
    def eval(self, x, y,):
        return (jnp.exp(- 0.5*(x - y)**2 / jnp.exp(2*self.scale))).sum() 

class MaternC1Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)
   
    ### one pair of pts
    def eval(self, x, y,):
        scale = jax.nn.softplus(self.scale)
        r_scaled = jnp.linalg.norm(x - y) * scale
        return (1 + jnp.sqrt(3) * r_scaled * jnp.exp(-jnp.sqrt(3) * r_scaled))
         
class MaternC2Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, init_realm=-2, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)
    ### one pair of pts
    def eval(self, x, y,):
        scale = jax.nn.softplus(self.scale)
        r_scaled = jnp.linalg.norm(x - y) * scale
        sq5 = jnp.sqrt(5)
        return (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
        
class NonstationaryMaternC2Kernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, 1, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims

    ### one pair of pts
    def eval(self, x, y,):
        sx = self.scale(x)
        sy = self.scale(y)
        scale = sx + sy
        r = jnp.linalg.norm(x - y)
        r_scaled = r * scale.squeeze()
        sq5 = jnp.sqrt(5)
        return (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
    
class AnisotropicMaternC2Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, ndims, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,(ndims,ndims), minval=-3.,maxval=0.0)
    ### one pair of pts
    def eval(self, x, y,):
        scale = jax.nn.softplus(self.scale)
        r_scaled = jnp.sqrt((x-y) @ scale @ (x-y))
        sq5 = jnp.sqrt(5)
        return (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
        
class NonstationaryAnisotropicMaternC2Kernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    place: Callable
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, int(ndims*(ndims+1)/2), key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)

    def eval(self, x, y):
        raw_Lx = self.scale(x)
        raw_Ly = self.scale(y)
        Lx = self.place(raw_Lx)
        Ly = self.place(raw_Ly)
        scale = Lx @ Lx.T + Ly @ Ly.T 
        diff = x - y
        r2 = diff @ scale @ diff
        r_scaled = jnp.sqrt(jnp.maximum(r2, 1e-10))
        sq5 = jnp.sqrt(5)
        return (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
    
class AnisotropicGaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    ndims: int
    place: Callable
    def __init__(self, *, ndims, key):
        keys = jr.split(key) 
        self.scale = jr.uniform(keys[0], (int(ndims*(ndims+1)/2),), minval=-3., maxval=0.)
        self.ndims = ndims
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)

    ### one pair of pts
    def eval(self, x, y,):
        L = self.place(self.scale)
        scale = L @ L.T
        r_scaled = (x-y) @ scale @ (x-y)
        return jnp.exp(-1/2 * r_scaled)
    
class NonstationaryGaussianKernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, 1, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims
    ### one pair of pts
    def eval(self, x, y,):
        sx = self.scale(x)
        sy = self.scale(y)
        scale = sx + sy
        r_scaled = (x-y)@(x-y) * scale.squeeze()
        return jnp.exp(-1/2 * r_scaled)
    
    ### make kernel matrix
    def __call__(self, 
                 x, 
                 y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(Y,X)
        return k_xy
    
class NonstationaryAnisotropicGaussianKernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    place: Callable
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim,int(ndims*(ndims+1)/2), key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims 
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)

    ### one pair of pts
    def eval(self, x, y,):
        Lx = self.place(self.scale(x))
        Ly = self.place(self.scale(y))
        scale = (Lx @ Lx.T) + (Ly @ Ly.T)
        r_scaled = (x-y) @ scale @ (x-y)
        return jnp.exp(-1/2 * r_scaled)
    
class MaternC0Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, init_realm=-2, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)

    ### one pair of pts
    def eval(self, x, y,):
        scale = jax.nn.softplus(self.scale)
        r_scaled = jnp.linalg.norm(x - y) * scale
        return jnp.exp(-r_scaled)

class MaternC3Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)
   
    ### one pair of pts
    def eval(self, x, y,):
            scale = jax.nn.softplus(self.scale)
            tau = jnp.linalg.norm(x-y)
            z = (3*tau) / scale
            k_xy = 1+z+ \
                    (27*tau**2)/(7*scale**2)+\
                    (18*tau**3)/(7*scale**3)+\
                    (27*tau**4)/(35*scale**4)
            k_xy *= jnp.exp(-z)
            return k_xy
    
class MaternC6Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, init_realm=-2, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)
    ### one pair of pts
    def eval(self, x, y,):
        psp = jax.nn.softplus(self.scale)
        r = jnp.linalg.norm(x - y)
        r_scaled = r * psp
        phi = (1 / 945) * jnp.exp(-(r_scaled)) * ((r_scaled)**5 + (15 * (r_scaled)**4) + (105 * (r_scaled)**3) + (420 * (r_scaled)**2) + (945 * (r_scaled)) + 945)
        return phi

class NonstationaryMaternC6Kernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim,1, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims 

    ### one pair of pts
    def eval(self, x, y,):
        sx = self.scale(x)
        sy = self.scale(y)
        scale = sx + sy
        r = jnp.linalg.norm(x - y)
        r_scaled = r * scale.squeeze()
        phi = (1 / 945) * jnp.exp(-(r_scaled)) * ((r_scaled)**5 + (15 * (r_scaled)**4) + (105 * (r_scaled)**3) + (420 * (r_scaled)**2) + (945 * (r_scaled)) + 945)
        return phi
    
class AnisotropicMaternC6Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, *, ndims, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,(ndims,ndims), minval=-3.,maxval=0.0)

    ### one pair of pts
    def eval(self, x, y,):
        scale = jax.nn.softplus(self.scale)
        r_scaled = jnp.sqrt((x-y) @ scale @ (x-y))
        phi = (1 / 945) * jnp.exp(-(r_scaled)) * ((r_scaled)**5 + (15 * (r_scaled)**4) + (105 * (r_scaled)**3) + (420 * (r_scaled)**2) + (945 * (r_scaled)) + 945)
        return phi

class NonstationaryAnisotropicMaternC6Kernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    place: Callable
    def __init__(self, *, ndims, latent_dim, key):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim,int(ndims*(ndims+1)/2), key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims 
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)

    def eval(self, x, y):
        raw_Lx = self.scale(x)
        raw_Ly = self.scale(y)
        Lx = self.place(raw_Lx)
        Ly = self.place(raw_Ly)
        scale = Lx @ Lx.T + Ly @ Ly.T 
        diff = x - y
        r2 = diff @ scale @ diff
        r = jnp.sqrt(jnp.maximum(r2, 1e-10))
        poly = r**5 + 15*r**4 + 105*r**3 + 420*r**2 + 945*r + 945
        return (1 / 945) * jnp.exp(-r) * poly
    
    ### make kernel matrix
    def __call__(self, 
                 x, 
                 y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(Y,X)
        return k_xy
    

class SpectralMixtureKernel(eqx.Module, KernelBaseClass):
    q: int
    weights: jax.Array
    freqs: jax.Array
    base_kernel: eqx.Module ### could be a vanilla kernel, or an anisotropic one

    def __init__(self, base_kernel, q, ndims, *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.q = q

        self.weights = jr.uniform(key1, (q,1), maxval=0.1, minval=-3.)
        self.freqs =  jr.uniform(key2, (q,ndims), maxval=0.1, minval=-3.)
        ### only anisotropic kernels care about ndims
        if 'ndims' in inspect.signature(base_kernel).parameters:
            base_kernel = partial(base_kernel, ndims=ndims)
        self.base_kernel = clm(base_kernel, q, key=key3)

    def eval(self, x,y):
        weights = jax.nn.softplus(self.weights)
        freqs = jax.nn.softplus(self.freqs)
        tau = (x-y)
        cos = jnp.cos(freqs @ tau)
        mat = eqx.filter_vmap(lambda m: m(x,y))(self.base_kernel.eval)
        return jnp.sum(weights * cos * mat)
    
    def __call__(self, 
                 x, 
                 y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(Y,X)
        return k_xy

    
class AnisotropicWendlandC4Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, ndims, *, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key, (ndims,ndims), maxval=0.1, minval=-3.)

    def eval(self, x, y):
            scale = jax.nn.softplus(self.scale)
            r = (x-y)@scale@(x-y)
            r = jnp.sqrt(jnp.maximum(1e-10, r))
            return jnp.where(r < 1, ((1 - r) ** 6) * (3 + 18 * r + 35 * r**2), 0)

class WendlandC6Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, *, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key, maxval=0.1, minval=-3.)

    def eval(self, x, y):
        scale = jax.nn.softplus(self.scale)
        r = jnp.linalg.norm(x-y)
        return jnp.where(r < 1, ((1 - r) ** 8) * (1+8*r+25*r**2+32*r**3), 0)
    
class WendlandC2Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, *, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key, maxval=0.1, minval=-3.)

    def eval(self, x, y):
        scale = jax.nn.softplus(self.scale)
        r = jnp.linalg.norm(x-y) * scale
        return jnp.where(r < 1, ((1 - r) ** 4) * (1+4*r), 0)
    
class WendlandC4Kernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, *, key):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key, maxval=0.1, minval=-3.)

    def eval(self, x, y):
            scale = jax.nn.softplus(self.scale)
            r = jnp.linalg.norm(x-y) * scale
            return jnp.where(r < 1, ((1 - r) ** 6) * (3 + 18 * r + 35 * r**2), 0)
    

class NonstationaryAnisotropicMaternC2SpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int
    place: Callable

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q*int(ndims*(ndims+1)/2), key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)
    def eval(self, x, y):
        wx, wy = self.weights(x), self.weights(y)
        fx, fy = self.freqs(x).reshape(self.q, -1), self.freqs(y).reshape(self.q, -1)
        place = jax.vmap(self.place)
        Lx = place(self.scales(x).reshape(self.q,-1)) ## q,ndims,ndims
        Ly = place(self.scales(y).reshape(self.q,-1))
        r_scaled = jax.vmap(lambda Lx, Ly: (x-y)@((Lx @ Lx.T) + (Ly @ Ly.T))@(x-y), in_axes=(0,0))(Lx,Ly)
        sq5 = jnp.sqrt(5)
        mat = (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * mat * cosine).sum()  # sum over mixtures
        return k_xy

class NonstationaryMaternC2SpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )

    def eval(self, x, y):
        wx, wy = self.weights(x), self.weights(y)
        fx, fy = self.freqs(x).reshape(self.q, -1), self.freqs(y).reshape(self.q, -1)
        sx, sy = self.scales(x),self.scales(y)
        r_scaled = jnp.linalg.norm(x-y) * (sx + sy)
        sq5 = jnp.sqrt(5)
        mat = (1 + sq5 * r_scaled + (5/3) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * mat * cosine).sum()  # sum over mixtures
        return k_xy


class NonstationaryAnisotropicMaternC6SpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int
    place: Callable

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q*int(ndims*(ndims+1)/2), key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)
    def eval(self, x, y):
        wx, wy = self.weights(x), self.weights(y)
        fx, fy = self.freqs(x).reshape(self.q, -1), self.freqs(y).reshape(self.q, -1)
        place = jax.vmap(self.place)
        Lx = place(self.scales(x).reshape(self.q,-1)) ## q,ndims,ndims
        Ly = place(self.scales(y).reshape(self.q,-1))
        r_scaled = jax.vmap(lambda Lx, Ly: (x-y)@((Lx @ Lx.T) + (Ly @ Ly.T))@(x-y), in_axes=(0,0))(Lx,Ly)
        mat = (1 / 945) * jnp.exp(-(r_scaled)) * ((r_scaled)**5 + (15 * (r_scaled)**4) + (105 * (r_scaled)**3) + (420 * (r_scaled)**2) + (945 * (r_scaled)) + 945)
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * mat * cosine).sum()  # sum over mixtures
        return k_xy

class NonstationaryMaternC6SpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )

    def eval(self, x, y):
        wx, wy = self.weights(x), self.weights(y)
        fx, fy = self.freqs(x).reshape(self.q, -1), self.freqs(y).reshape(self.q, -1)
        sx, sy = self.scales(x),self.scales(y)
        r_scaled = jnp.linalg.norm(x-y) * (sx + sy)
        mat = (1 / 945) * jnp.exp(-(r_scaled)) * ((r_scaled)**5 + (15 * (r_scaled)**4) + (105 * (r_scaled)**3) + (420 * (r_scaled)**2) + (945 * (r_scaled)) + 945)
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * mat * cosine).sum()  # sum over mixtures
        return k_xy


class NonstationaryAnisotropicGaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int
    place: Callable

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q*int(ndims*(ndims+1)/2), key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)
    def eval(self, x, y):
        w,f = jax.checkpoint(lambda x: self.weights(x)), jax.checkpoint(lambda x: self.freqs(x))
        wx, wy = w(x), w(y)
        fx, fy = f(x).reshape(self.q, -1), f(y).reshape(self.q, -1)
        place = jax.vmap(self.place)
        Lx = place(self.scales(x).reshape(self.q,-1)) ## q,ndims,ndims
        Ly = place(self.scales(y).reshape(self.q,-1))
        r_scaled = jax.vmap(lambda Lx, Ly: (x-y)@((Lx @ Lx.T) + (Ly @ Ly.T))@(x-y), in_axes=(0,0))(Lx,Ly)
        gauss = jnp.exp(-1/2 * r_scaled)
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * gauss * cosine).sum()  # sum over mixtures
        return k_xy

class NonstationaryGaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    freqs: eqx.Module
    scales: eqx.Module
    q: int

    def __init__(
        self,
        ndims: int,
        q: int,
        latent_dim: int,
        *,
        key,):
        self.q = q
        key,_ = jr.split(key)
        shared_w = eqx.nn.Linear(ndims, latent_dim, key=key)
        keys = jr.split(key,3)
        self.weights = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[0]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )  
        self.freqs = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q * ndims, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.scales = eqx.nn.Sequential(
            [
                shared_w,
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, q, key=keys[2]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )

    def eval(self, x, y):
        wx, wy = self.weights(x), self.weights(y)
        fx, fy = self.freqs(x).reshape(self.q, -1), self.freqs(y).reshape(self.q, -1)
        sx, sy = self.scales(x),self.scales(y)
        k_gibbs = (jnp.sqrt(2 * sx * sy) / (sx**2 + sy**2)) * jnp.exp(
            -(jnp.sum((x - y) ** 2)) / (sx**2 + sy**2)
        )
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * k_gibbs * cosine).sum()  # sum over mixtures
        return k_xy

class AdditiveKernel(eqx.Module, KernelBaseClass):
    w: jax.Array
    ks: List[eqx.Module]

    def __init__(self, *, ks, key):
        keys = jr.split(key,)
        self.w = jnp.ones((len(ks,)))/len(ks)
        self.ks = [ks[i](key=k) for i,k in enumerate(jr.split(keys[1], len(ks)))]

    ### one pair of pts
    def eval(self, x, y,):
        w = jax.nn.softplus(self.w)
        acc = 0.
        for i,k in enumerate(range(len(self.ks))):
            acc+= w[i]*self.ks[i].eval(x,y)
        return acc
 
kernels = {'g': GaussianKernel,
           'a_g': AnisotropicGaussianKernel,
           'ns_g': partial(NonstationaryGaussianKernel, latent_dim=8),
           'ns_a_g': partial(NonstationaryAnisotropicGaussianKernel, latent_dim=8),
           'gsm': partial(SpectralMixtureKernel, base_kernel=GaussianKernel, q=2),
           'a_gsm': partial(SpectralMixtureKernel, base_kernel=AnisotropicGaussianKernel, q=2),
           'ns_gsm': partial(NonstationaryGaussianSpectralMixtureKernel, latent_dim=8, q=2),
           'ns_a_gsm': partial(NonstationaryAnisotropicGaussianSpectralMixtureKernel, latent_dim=8, q=2),
           'ns_a_gsm_no': NonstationaryAnisotropicGaussianSpectralMixtureKernel,
           'm2': MaternC2Kernel,
           'm6': MaternC6Kernel,
           'a_m2': AnisotropicMaternC2Kernel,
           'a_m6': AnisotropicMaternC6Kernel,
           'ns_m2': partial(NonstationaryMaternC2Kernel, latent_dim=8),
           'ns_m6': partial(NonstationaryMaternC6Kernel, latent_dim=8),
           'm2sm': partial(SpectralMixtureKernel, base_kernel=MaternC2Kernel, q=2),          
           'm6sm': partial(SpectralMixtureKernel, base_kernel=MaternC6Kernel, q=2),         
           'a_m2sm': partial(SpectralMixtureKernel, base_kernel=AnisotropicMaternC2Kernel, q=2),        
           'a_m6sm': partial(SpectralMixtureKernel, base_kernel=AnisotropicMaternC6Kernel, q=2),
           'ns_a_m6sm': partial(NonstationaryAnisotropicMaternC6SpectralMixtureKernel, latent_dim=8, q=2), 
           'ns_m6sm': partial(NonstationaryMaternC6SpectralMixtureKernel, latent_dim=8, q=2),
           'ns_a_m2sm': partial(NonstationaryAnisotropicMaternC2SpectralMixtureKernel, latent_dim=8, q=2), 
           'ns_m2sm': partial(NonstationaryMaternC2SpectralMixtureKernel, latent_dim=8, q=2), 
           'rq': RationalQuadraticKernel,
        
           }
