# MARVEL  -  A Multi-Depth Lookup-Free Quantization 1D Tokenizer


We introduced a residual tokenizer with a lookup-free quantization approach to modify the well-known 1D tokenizer scheme, Titok.


## 🚀 Contributions

#### We introduce a novel 1D image tokenization framework that breaks grid constraints existing in 2D tokenization methods, leading to a much more flexible and compact image latent representation.

#### The proposed 1D tokenizer can tokenize a 256 × 256 image into as few as 32 discrete tokens, leading to a significant speed-up (hundreds times faster than diffusion models) in generation process, while maintaining state-of-the-art generation quality.

#### We conduct a series of experiments to probe the properties of rarely studied 1D image tokenization, paving the path towards compact latent space for efficient and effective image representation.

## Model Zoo
| Model | Link | FID |
| ------------- | ------------- | ------------- |
| TiTok-L-32 Tokenizer | [checkpoint](https://huggingface.co/yucornetto/tokenizer_titok_l32_imagenet)| 2.21 (reconstruction) |
| TiTok-B-64 Tokenizer | [checkpoint](https://huggingface.co/yucornetto/tokenizer_titok_b64_imagenet) | 1.70 (reconstruction) |
| TiTok-S-128 Tokenizer | [checkpoint](https://huggingface.co/yucornetto/tokenizer_titok_s128_imagenet) | 1.71 (reconstruction) |
| MARVEL-L-32 Tokenizer | _| 2.21 (reconstruction) |
| MARVEL-L-32*2 Tokenizer | _ | 1.36 (reconstruction) |
| MARVEL-L-32*4 Tokenizer | _ | 1.06 (reconstruction) |

Please note that these models are trained only on limited academic dataset ImageNet, and they are only for research purposes.

## Installation
```shell
pip3 install -r requirements.txt
```

## Training Preparation
We use [webdataset](https://github.com/webdataset/webdataset) format for data loading. To begin with, it is needed to convert the dataset into webdataset format. An example script to convert ImageNet to wds format is provided [here](./data/convert_imagenet_to_wds.py).

Furthermore, the stage1 training relies on a pre-trained MaskGIT-VQGAN to generate proxy codes as learning targets. You can convert the [official Jax weight](https://github.com/google-research/maskgit) to PyTorch version using [this script](https://github.com/huggingface/open-muse/blob/main/scripts/convert_maskgit_vqgan.py). Alternatively, we provided a converted version at [HuggingFace](https://huggingface.co/fun-research/TiTok/blob/main/maskgit-vqgan-imagenet-f16-256.bin) and [Google Drive](https://drive.google.com/file/d/1DjZqzJrUt2hwpmUPkjGSBTFEJcOkLY-Q/view?usp=sharing). The MaskGIT-VQGAN's weight will be automatically downloaded when you run the training script.

You may also pretokenize the dataset for a training speedup, please refer to the [example pretokenization script](scripts/pretokenization.py).

## Training
We provide example commands to train TiTok as follows:
```bash
# Training for TiTok-l32*4
# Stage 1
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/TiTok/stage1/finetune_titok_l32.yaml \
    experiment.project="finetune_titok_l32_stage1" \
    experiment.name="finetune_titok_l32_stage1_run1" \
    experiment.output_dir="finetune_titok_l32_stage1_run1" \
    training.per_gpu_batch_size=32

# Stage 2
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/TiTok/stage2/finetune_titok_l32.yaml \
    experiment.project="finetune_titok_l32_stage2" \
    experiment.name="finetune_titok_l32_stage2_run1" \
    experiment.output_dir="finetune_titok_l32_stage2_run1" \
    training.per_gpu_batch_size=32 \
    experiment.init_weight=${PATH_TO_STAGE1_WEIGHT}
```
You may remove the flag "WANDB_MODE=offline" to support online wandb logging, if you have configured it.

The config _titok_b64.yaml_ can be replaced with _titok_s128.yaml_ or _titok_l32.yaml_ for other TiTok variants.

## Visualizations
<p>
<img src="assets/recon_w_model_size_num_token.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/random_vis_l32.png" alt="teaser" width=90% height=90%>
</p>



## Acknowledgement

[MaskGIT](https://github.com/google-research/maskgit)

[Taming-Transformers](https://github.com/CompVis/taming-transformers)

[Open-MUSE](https://github.com/huggingface/open-muse)

[MUSE-Pytorch](https://github.com/baaivision/MUSE-Pytorch)
