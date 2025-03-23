# [Transformers without Normalization](https://arxiv.org/abs/2503.10622)

Official PyTorch implementation of **DynamicTanh (DyT)**, from the following paper:

[Transformers with Normalization](https://arxiv.org/abs/2503.10622). CVPR 2025. \
[Jiachen Zhu](https://jiachenzhu.github.io), [Xinlei Chen](https://xinleic.xyz/), [Kaiming He](https://people.csail.mit.edu/kaiming/), [Yann LeCun](http://yann.lecun.com) and [Zhuang Liu](https://liuzhuang13.github.io) \
FAIR, NYU, MIT, Princeton \
[[`arXiv`](https://arxiv.org/abs/2503.10622)][[`project page`](https://jiachenzhu.github.io/DyT/)]

--- 

<p align="center">
<img src="https://raw.githubusercontent.com/jiachenzhu/jiachenzhu.github.io/refs/heads/master/DyT/webpage_assets/before_after.svg" width=100% height=100% 
class="center">
</p>

We propose **DynamicTanh(DyT)**, an element-wise operation defined as: DyT(***x***) = tanh($\alpha$***x***), where $\alpha$ is a learnable scaler.
DyT is designed to replace normalization layers in Transformers. Models with DyT achieves similar or better performance than their normalized counterparts.



## Installation
To reproduce our results, run the following commands to set up the Python environment:
```
conda create -n DyT python=3.12
conda activate DyT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

## Training

To reproduce our results on ImageNet-1K with ViT and ConvNeXt, run the following commands: \
(For results with LN, set `--dynamic_tanh` to `false`.)

<details>
<summary>
ViT-B
</summary>
    
```
torchrun --nnodes=8 --nproc_per_node=8 main.py \
    --model vit_base_patch16_224 \
    --drop_path 0.1 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --dynamic_tanh true
```
</details>
<details>
<summary>
ViT-L
</summary>
    
```
torchrun --nnodes=8 --nproc_per_node=8 main.py \
    --model vit_large_patch16_224 \
    --drop_path 0.4 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --opt_betas 0.9 0.95 \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --dynamic_tanh true
```
</details>
<details>
<summary>
ConvNeXt-B
</summary>
    
```
torchrun --nnodes=8 --nproc_per_node=8 main.py \
    --model convnext_base \
    --drop_path 0.5 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --dynamic_tanh true
```
</details>
<details>
<summary>
ConvNeXt-L
</summary>
    
```
torchrun --nnodes=8 --nproc_per_node=8 main.py \
    --model convnext_large \
    --drop_path 0.5 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --dynamic_tanh true
```
</details>

## ImageNet-1K Results

| name | acc@1 (LN) | acc@1 (DyT) |
|:---:|:---:|:---:|
| ViT-B | 82.3% | 82.5% | 
| ViT-L | 83.1% | 83.6% | 
| ConvNeXt-B | 83.7% | 83.7% |
| ConvNeXt-L | 84.3% | 84.4% |

## Other Tasks
To reproduce results for other tasks, follow the instructions in the respective folders:
- [MAE](other_tasks/MAE)
- [DINO](other_tasks/DINO)
- [DiT](other_tasks/DiT)
- [LLaMA](other_tasks/LLaMA)
- [wav2vec 2.0](other_tasks/wav2vec2)
- [DNA](other_tasks/DNA)

To apply DyT to your own models, see the [HowTo](other_tasks/HowTo) guide.

## Efficiency
To reproduce the computational efficiency results in *Section 6.1*, follow the instructions in the [Efficiency](other_tasks/Efficiency) folder.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{Zhu2025DyT,
  title={Transformers without Normalization},
  author={Zhu, Jiachen and Chen, Xinlei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
