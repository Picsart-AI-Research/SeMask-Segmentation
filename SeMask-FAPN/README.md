# SeMask FaPN

This repo contains the code for our paper **SeMask: Semantically Masked Transformers for Semantic Segmentation**. It is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [FaPN-full](https://github.com/ShihuaHuang95/FaPN-full).

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
| SeMask-L Mask2Former MSFaPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 56.54  | 58.22 | 224M | [config](SeMask-Mask2Former/configs/ade20k/semantic-segmentation/semask_swin/msfapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | [checkpoint](https://drive.google.com/file/d/1w-DRGufIv3zpDO7rJFv2z5WeLx0pDTJe/view?usp=sharing) |
| SeMask-L Mask2Former FaPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | **56.97**  | **58.22**  | 227M | [config](SeMask-Mask2Former/configs/ade20k/semantic-segmentation/semask_swin/fapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | [checkpoint](https://drive.google.com/file/d/1DQ9KltSLDj47H2jYnCtVwyBf7KPR9SM_/view?usp=sharing) |


## 2. Setup Instructions

### Installation

- Build the [DCNv2](DCNv2) module which is compatible with [Pytorch v1.7.1](https://pytorch.org/get-started/locally/).

- Follow the installation instructions for [Mask2Former](SeMask-Mask2Former/INSTALL.md).

### Getting Started

See [Preparing Datasets for Mask2Former](SeMask-Mask2Former/datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

## 3. Citing SeMask

```BibTeX
@article{jain2021semask,
  title={SeMask: Semantically Masking Transformer Backbones for Effective Semantic Segmentation},
  author={Jitesh Jain and Anukriti Singh and Nikita Orlov and Zilong Huang and Jiachen Li and Steven Walton and Humphrey Shi},
  journal={arXiv},
  year={2021}
}
```