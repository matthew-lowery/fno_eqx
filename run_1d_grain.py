import jax
import optax
from jax import numpy as jnp
import jax.random as jr
from utils import *
import equinox as eqx
from model import FNO
import grain.python as grain
from pathlib import Path
import numpy as np
import time

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(seed=42)
path = '../datasets/burgers.npz'

### load data
data = jnp.load('../datasets/burgers.npz')
x, x_grid, y, y_grid = data["x"], data["x_grid"], data["y"], data["y_grid"]
print(f'dataset dims: {x.shape=}, {x_grid.shape=}, {y.shape=}, {y_grid.shape=}')

ntrain = 1000
ntest = 200

x_train, x_test = x[: ntrain], x[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)
# print(x_train.shape, y_train.shape)

class BurgersTrain(grain.RandomAccessDataSource):
    def __init__(self,):
        self._data = x_train, y_train
    def __getitem__(self, idx):
        return (self._data[0][idx], self._data[1][idx])
    def __len__(self):
        return len(self._data[0])
    
train_batch_size = 20
test_batch_size = 200
num_train_batches = len(x_train) // train_batch_size

train_dataset = (
    grain.MapDataset.source(BurgersTrain())
    .shuffle(seed=42)
    .batch(batch_size=train_batch_size)
)
train_batched = train_dataset.to_iter_dataset(grain.ReadOptions(num_threads=10, prefetch_buffer_size=10))

## model config 
modes = [12] ### list of modes, one per dim
depth = 4
activation = jax.nn.gelu
lift_dim= 32

model = FNO(modes, lift_dim, activation, depth, key=key)

### optimizer config 
epochs = 10000
lr_schedule = cosine_annealing(
    total_steps = num_train_batches*epochs,
    init_value=1e-3,
    warmup_frac=0.3,
    peak_value=1e-3,
    end_value=1e-3,
    num_cycles=6,
    gamma=0.9,)

optimizer = optax.adamw(lr_schedule)

### misc config
print_every = 1

#### model init
print(f'param count: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))}')
optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, optimizer_state, batch,):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,x_grid))(x)
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
        y_pred = eqx.filter_vmap(lambda x: model(x,x_grid,))(x)
        y_pred = y_normalizer.decode(y_pred)
        return ((y-y_pred)**2).sum(axis=1).mean(), y_pred

    test_loss,y_pred = loss(model)
    l2_loss = jnp.linalg.norm(y - y_pred, axis=1) / jnp.linalg.norm(y,axis=1)
    return l2_loss.mean()

t1 = time.perf_counter()
for epoch in range(epochs):
    for batch in train_batched:
        model, optimizer_state, (train_loss, train_l2) = train_step(model, optimizer_state, batch)

    if (epoch % print_every) == 0 or (epoch == epochs - 1):
        test_l2 = eval(model, (x_test,y_test))
        print(f"{epoch=}, train_loss: {train_loss.item():.3f}, train_l2: {train_l2.item()*100:.3f}, test_l2: {test_l2.item()*100:.3f}")
print(time.perf_counter() - t1)