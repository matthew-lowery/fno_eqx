import jax
import optax
from jax import numpy as jnp
import jax.random as jr
from utils import *
import equinox as eqx
from model import FNO

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(seed=42)

### load data
data = jnp.load('./datasets/ns_pdebench_pres_t0_t10.npz') ### 
x, x_grid, y= data["x"], data["x_grid"], data["y"]
res = 128
n=100
ndims = 3
x = x.reshape(n,res,res,res,1)
x_grid = jnp.asarray(jnp.meshgrid(x_grid, x_grid,x_grid)).transpose(1,2,3,0)
y_grid = x_grid
y = y.reshape(n,-1) 
print(f'dataset dims: {x.shape=}, {x_grid.shape=}, {y.shape=}, {y_grid.shape=}')
ntrain = 90
ntest = 10

x_train, x_test = x[: ntrain], x[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]


### data config 
train_batch_size = 20
num_train_batches = len(x_train) // train_batch_size


## model config 
modes = [12,12,12]
depth = 3
activation = jax.nn.gelu
lift_dim= 32

model = FNO(modes, lift_dim, activation, depth, key=key)

### optimizer config 
epochs = 10000
lr_schedule = cosine_annealing(
    total_steps = num_train_batches*epochs,
    init_value=1e-3,
    warmup_frac=0.3,
    peak_value=3e-3,
    end_value=1e-5,
    num_cycles=5,
    gamma=0.7,)

optimizer = optax.adam(lr_schedule)


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

print(f'param count: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))}')
optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, batch, optimizer_state):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,x_grid))(x)
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
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
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        y_pred = y_normalizer.decode(y_pred)
        return ((y-y_pred)**2).sum(axis=1).mean(), y_pred

    test_loss,y_pred = loss(model)
    l2_loss = jnp.linalg.norm(y - y_pred, axis=1) / jnp.linalg.norm(y,axis=1)
    return l2_loss.mean()


for epoch in range(epochs):

    epoch_key,_ = jr.split(key)

    for batch_i in range(num_train_batches):
        batch = get_train_batch(batch_i, epoch_key)
        model, optimizer_state, (train_loss, train_l2) = train_step(model, batch, optimizer_state)
        
    if (epoch % print_every) == 0 or (epoch == epochs - 1):
        test_l2 = eval(model, (x_test, y_test))
        print(f"{epoch=}, train_loss: {train_loss.item():.3f}, train_l2: {train_l2.item()*100:.3f}, test_l2: {test_l2.item()*100:.3f}")
