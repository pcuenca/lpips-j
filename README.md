# LPIPS-J

This is a minimal JAX/Flax port of `lpips`, as implemented in:
* [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/lpips.py)
* [Taming Transformers](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py)

Only the essential features have been implemented. Our motivation is to support VQGAN training for [DALL•E Mini](https://github.com/borisdayma/dalle-mini).

It currently supports the `vgg16` backend, leveraging the implementation in [`flaxmodels`](https://github.com/matthias-wright/flaxmodels/blob/main/flaxmodels/vgg/vgg.py).

Pre-trained weights for the network and the linear layers are downloaded from the [🤗 Hugging Face hub](https://huggingface.co/pcuenq/lpips-jax).

## Installation

1. Install JAX for CUDA or TPU following the instructions at https://github.com/google/jax#installation.
2. Install this package:
   ```
   pip install lpips-j
   ```

## Use

Inputs must be in the range `[-1, 1]`, and _not_ already normalized with ImageNet stats. (They are internally converted to `[0, 1]` and then normalized by [the underlying flax model](https://github.com/matthias-wright/flaxmodels/tree/main/flaxmodels/vgg#1-important-note).

```Python
x = PILToTensor()(Image.open("img8.jpg")).unsqueeze(0)
y = PILToTensor()(Image.open("img8_edited.jpg")).unsqueeze(0)

x = 2 * (x / 255.) - 1
y = 2 * (y / 255.) - 1
x = jnp.array(x).transpose(0, 2, 3, 1)
y = jnp.array(y).transpose(0, 2, 3, 1)

lpips = LPIPS()
params = lpips.init(key, x, x)
loss = lpips.apply(params, x, y)
```