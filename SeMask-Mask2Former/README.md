# SeMask Mask2Former

This repo contains the code for our paper **SeMask: Semantically Masked Transformers for Semantic Segmentation**. It is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## Contents
1. [Results](#1-results)
2. [Setup Instructions](#2-setup-instructions)
3. [Citing SeMask](#3-citing-semask)

## 1. Results

- &dagger; denotes the backbones were pretrained on ImageNet-22k and 384x384 resolution images.
- Pre-trained models can be downloaded following the instructions given [under tools](tools/README.md).

### ADE20K

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |
| SeMask-L Mask2Former | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 56.41  | 57.52 | 222M | [config](configs/ade20k/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) |

### Cityscapes

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |
| SeMask-L Mask2Former | SeMask Swin-L<sup>&dagger;</sup> | 512x1024 | 83.97  | 84.98 | 222M | [config](SeMask-Mask2Former/configs/cityscapes/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_90k.yaml) |

## 2. Setup Instructions

### Installation
- We developed the codebase using [Pytorch v1.9.0](https://pytorch.org/get-started/locally/) and python 3.8.
  ```
  pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```
- See [installation instructions](INSTALL.md).

### Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

## 3. Citing SeMask

```BibTeX
@article{jain2022semask,
  title={SeMask: Semantically Masking Transformer Backbones for Effective Semantic Segmentation},
  author={Jitesh Jain and Anukriti Singh and Nikita Orlov and Zilong Huang and Jiachen Li and Steven Walton and Humphrey Shi},
  journal={arXiv preprint arXiv:...},
  year={2022}
}
```
