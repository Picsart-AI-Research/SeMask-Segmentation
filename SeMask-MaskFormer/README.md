# SeMask-MaskFormer

This repo contains the code for our paper **SeMask: Semantically Masked Transformers for Semantic Segmentation**. It is based on [MaskFormer](https://github.com/facebookresearch/MaskFormer).

## Contents
1. [Results](#1-results)
2. [Setup Instructions](#2-setup-instructions)
3. [Citing SeMask](#3-citing-semask)

## 1. Results

- &dagger; denotes the backbones were pretrained on ImageNet-22k and 384x384 resolution images.
- Pre-trained models can be downloaded following the instructions given [under tools](tools/README.md).

### ADE20K

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |    :---:   |
| SeMask-L MaskFormer | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 54.75  | 56.15 | 219M | [config](configs/ade20k-150/semask_swin/maskformer_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | TBD |

## 2. Setup Instructions

### Installation

See [installation instructions](INSTALL.md).

### Getting Started

See [Preparing Datasets for MaskFormer](datasets/README.md).

See [Getting Started with MaskFormer](GETTING_STARTED.md).

## 3. Citing SeMask

```BibTeX
@article{jain2021semask,
  title={SeMask: Semantically Masking Transformer Backbones for Effective Semantic Segmentation},
  author={Jitesh Jain and Anukriti Singh and Nikita Orlov and Zilong Huang and Jiachen Li and Steven Walton and Humphrey Shi},
  journal={arXiv},
  year={2021}
}
```
