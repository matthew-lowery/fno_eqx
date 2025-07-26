import jax
import optax
from jax import numpy as jnp
import jax.random as jr
from utils import *
import equinox as eqx
import wandb
from model import FNO
import os
from kernels import kernels

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--mode', type=int, default=2)
parser.add_argument('--res_1d', type=int, default=8)
parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--input-kernel', type=str, default='g')
parser.add_argument('--output-kernel', type=str, default='g')
parser.add_argument('--name', type=str, default='car_fno_interp')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--seed', type=int, default=42)


args = parser.parse_args()

print(args)
if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"


wandb.login(key='d612cda26a5690e196d092756d668fc2aee8525b')
wandb.init(project=args.name)
wandb.config.update(args)


key = jax.random.PRNGKey(seed=42)

### load data
### load data
data = np.load('../deep_gp_op/datasets/car_finished.npz')
x_grid = data['vertices']
y = data['pressure']
print(x_grid.shape, y.shape)
domain_dims = 3 
codomain_dims = 1
ntrain = 500
ntest = 111
x_grid_train, x_grid_test = x_grid[: ntrain], x_grid[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]

res_1d = args.res_1d
grid_1d = jnp.linspace(0,1,res_1d)
q_grid = jnp.asarray(jnp.meshgrid(grid_1d, grid_1d, grid_1d)).transpose(1,2,3,0).reshape(-1,3)
        
### data config 
train_batch_size = args.batch_size
num_train_batches = len(y_train) // train_batch_size

## kernel setup
domain_dims = 3
input_kernel = kernels[args.input_kernel]
output_kernel = kernels[args.output_kernel]
input_kernel = partial(input_kernel, ndims=domain_dims)
output_kernel = partial(output_kernel, ndims=domain_dims)

## model config 
modes = [args.mode,]*3
activation = jax.nn.gelu
model = FNO(modes, args.lift_dim, activation, args.depth, 0, res_1d, input_kernel, output_kernel, key=key)

### optimizer config 

optimizer = optax.adam(args.lr_max)


### misc config
print_every = 1


### preprocess data
# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
# y_normalizer = UnitGaussianNormalizer(y_train)


### dataset is small enough to fully load onto gpu and slice
@jax.jit
def get_train_batch(
    i,
    key,
):
    xtr = jr.permutation(key, x_grid_train)
    ytr = jr.permutation(key, y_train)
    x = jax.lax.dynamic_slice_in_dim(
        xtr,
        i * args.batch_size,
        args.batch_size,
    )

    y = jax.lax.dynamic_slice_in_dim(
        ytr,
        i * args.batch_size,
        args.batch_size,
    )
    return x, y

#### model init

print(f'param count: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))}')
optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, batch, optimizer_state):
    x_grid,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x_grid: model(x_grid,q_grid))(x_grid)
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        # y_pred = y_normalizer.decode(y_pred)
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
    x_grid,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x_grid: model(x_grid,q_grid))(x_grid)
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        # y_pred = y_normalizer.decode(y_pred)
        return ((y-y_pred)**2).sum(axis=1).mean(), y_pred

    test_loss,y_pred = loss(model)
    l2_loss = jnp.linalg.norm(y - y_pred, axis=1) / jnp.linalg.norm(y,axis=1)
    return l2_loss.mean()


for epoch in range(args.epochs):

    epoch_key,_ = jr.split(key)

    for batch_i in range(num_train_batches):
        batch = get_train_batch(batch_i, epoch_key)
        model, optimizer_state, (train_loss, train_l2) = train_step(model, batch, optimizer_state)
        wandb.log({"train_loss": train_l2.item()*100,})
    if (epoch % print_every) == 0 or (epoch == args.epochs - 1):
        test_l2 = eval(model, (x_grid_test, y_test))
        wandb.log({"train_loss": test_l2.item()*100,}, step=epoch)
        print(f"{epoch=}, train_loss: {train_loss.item():.3f}, train_l2: {train_l2.item()*100:.3f}, test_l2: {test_l2.item()*100:.3f}")
