# LPIPS-J

This is a minimal JAX/Flax port of `lpips`, as implemented in:
* [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/pretrained_networks.py)
* [Taming Transformers](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py)

Only the essential features have been implemented. Our motivation is to support VQGAN training for [DALL•E Mini](https://github.com/borisdayma/dalle-mini).

It currently supports the `vgg16` backend, leveraging the implementation in [`flaxmodels`](https://github.com/matthias-wright/flaxmodels/blob/main/flaxmodels/vgg/vgg.py).

Pre-trained weights for the network and the linear layers are downloaded from the [🤗 Hugging Face hub](https://huggingface.co/pcuenq/lpips-jax).