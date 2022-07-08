import jax
import jax.numpy as jnp
import random
import torch

from taming.modules.losses.lpips import LPIPS as taming_LPIPS
from lpips_j.lpips import LPIPS

# Taming Transformers LPIPS in PyTorch

x = torch.rand([8, 3, 256, 256])
y = torch.rand([8, 3, 256, 256])

torch_lpips = taming_LPIPS()
result_t = torch_lpips(x, y)
print(result_t)

# JAX version

seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

x = jnp.array(x).transpose(0, 2, 3, 1)
y = jnp.array(y).transpose(0, 2, 3, 1)

lpips = LPIPS()
params = lpips.init(key, x, x)

result_j = lpips.apply(params, x, y)
print(result_j)

# For some reason the difference is in the order of 1e-3

import numpy as np
max_diff = (result_t - torch.Tensor(np.array(result_j))).abs().max().item()
print(f'Max diff: {max_diff}')
