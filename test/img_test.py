import jax
import jax.numpy as jnp
import random
import torch

from torchvision.transforms import PILToTensor
from taming.modules.losses.lpips import LPIPS as taming_LPIPS
from lpips_j.lpips import LPIPS
from PIL import Image

# Taming Transformers LPIPS in PyTorch

# The torch version appears not to be deterministic for some reason.
# Are all layers being initialized from pretrained weights?
torch.manual_seed(7667)

x = PILToTensor()(Image.open("img8.jpg")).unsqueeze(0)
y = PILToTensor()(Image.open("img8_edited.jpg")).unsqueeze(0)

x = 2 * (x / 255.) - 1
y = 2 * (y / 255.) - 1

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