import jax
import optax
from jax import numpy as jnp
import jax.random as jr
from utils import *
import equinox as eqx
from model import FNO
import time
import pickle
import wandb
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=64)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--visc', type=str, default='0.001000')
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()

key = jax.random.PRNGKey(seed=args.seed)

def is_trainable(x):
    return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
    
## load data
from scipy.io import loadmat
data = loadmat(f'./datasets/burgers_1200_{args.visc}.mat')
# data = loadmat('/Users/mattlowery/Desktop/code/deeponet-fno/data/burgers/burgers_1200_0.001000')
data = data['output']
x = data[:,0]
y = data[:,1]
x_grid = jnp.linspace(0,1,8192)

# x,x_grid,y = data['x'].astype(jnp.float32), data['x_grid'].astype(jnp.float32), data['y'].astype(jnp.float32)

x = x.reshape(1200,-1,1)
y = y.reshape(1200,-1)
sub = 64
x,y = x[:,::sub], y[:,::sub]
x_grid = jnp.linspace(0,1,x.shape[1]).reshape(-1,1)
print(x.shape, y.shape, x_grid.shape)
ntrain = 1000
ntest = 200

# fp = '../datasets/burgers.npz'
# data = jnp.load(fp)
# dataset = fp.split('/')[-1].split('.')[0]
# x, x_grid, y, y_grid = data["x"].astype(DTYPE), data["x_grid"].astype(DTYPE), data["y"].astype(DTYPE), data["y_grid"].astype(DTYPE)
# y = y.reshape(1200, -1)
# ntrain = 1000
# ntest = 200

# from matplotlib import pyplot as plt
# plt.plot(x_grid.squeeze(), y[0])
# plt.show()

x_train, x_test = x[: ntrain], x[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
### data config 
train_batch_size = 100
num_train_batches = len(x_train) // train_batch_size

# def get_beijing(seed=0, normalization=True):
#     Ntr, Nte = 5000, 1000
#     with open('./datasets/beijing_data.pickle', 'rb') as handle:
#         d = pickle.load(handle)
#     X, Y = d["x"][:Ntr+Nte], d["y"][:Ntr+Nte]
#     X,Y=shuffle(X,Y)
#     Xtr, Xte = X[:Ntr], X[Ntr:]
#     Ytr, Yte = Y[:Ntr], Y[Ntr:]
#     return Xtr, Xte,Ytr,Yte


import os
os.environ["WANDB_MODE"] = "disabled"
wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project='fno')


## model config 
modes = [args.mode] ### list of modes, one per dim
depth = 4
activation = jax.nn.gelu
lift_dim= 64

model = FNO(modes, lift_dim, activation, depth, 1, key=key)

print(f'param count: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))}')

### optimizer config 
epochs = 10000

optimizer = optax.adamw(0.001)


### misc config
print_every = 1


### preprocess data
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)


### dataset is small enough to fully load onto gpu and slice
@jax.jit
def get_train_batch(
    i,
    key,
):
    xtr = jr.permutation(key, x_train)
    ytr = jr.permutation(key, y_train)
    
    x = jax.lax.dynamic_slice_in_dim(
        xtr,
        i * train_batch_size,
        train_batch_size,
    )
    y = jax.lax.dynamic_slice_in_dim(
        ytr,
        i * train_batch_size,
        train_batch_size,
    )
    return x, y


#### model init

optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, batch, optimizer_state):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,x_grid))(x).squeeze()
        y_pred = y_normalizer.decode(y_pred)
        return ((y-y_pred)**2).sum(axis=1).mean(), y_pred
    
    (train_loss, y_pred), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model)
    updates,optimizer_state = optimizer.update(grads, 
                                               optimizer_state, 
                                               eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    l2_loss = jnp.linalg.norm(y - y_pred, axis=1) / jnp.linalg.norm(y, axis=1)
    return model, optimizer_state, (train_loss, l2_loss.mean())


@eqx.filter_jit
def eval(model, batch,):
    x,y = batch
    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,x_grid,))(x).squeeze()
        y_pred = y_normalizer.decode(y_pred)
        return ((y-y_pred)**2).sum(axis=1).mean(), y_pred

    test_loss,y_pred = loss(model)
    l2_loss = jnp.linalg.norm(y - y_pred, axis=1) / jnp.linalg.norm(y,axis=1)
    return l2_loss.mean()
t1 = time.perf_counter()
for epoch in range(epochs):
    epoch_key,_ = jr.split(key)
    for batch_i in range(num_train_batches):
        batch = get_train_batch(batch_i, epoch_key)
        model, optimizer_state, (train_loss, train_l2) = train_step(model, batch, optimizer_state)
        
    if (epoch % print_every) == 0 or (epoch == epochs - 1):
        test_l2 = eval(model, (x_test, y_test))
        print(f"{epoch=}, train_loss: {train_loss.item():.3f}, train_l2: {train_l2.item()*100:.8f}, test_l2: {test_l2.item()*100:.8f}")
        wandb.log({"test_loss": test_l2.item()*100}, step=epoch)
print(time.perf_counter() - t1)
