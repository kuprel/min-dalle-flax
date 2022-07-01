# min(DALL·E) Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb) &nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)

This is a minimal implementation of Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini) in flax.  It has been stripped to the bare essentials necessary for doing inference.  This repository also contains code for converting the flax model to torch.

See [min(DALL·E)](https://github.com/kuprel/min-dalle) for the PyTorch version.

### Setup

Run `sh setup.sh` to install dependencies and download pretrained models.  The flax models can be manually downloaded here: 
[VQGan](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384), 
[DALL·E Mini](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mini-1/v0/files), 
[DALL·E Mega](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/v14/files)

### Usage

Use the python script `image_from_text.py` to generate images from the command line.  Note: the command line script loads the models and parameters each time.  To load a model once and generate multiple times, initialize `MinDalleFlax`, then call `generate_image` with some text and a seed.
# min-dalle-flax
