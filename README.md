# SeMask: Semantically Masked Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semask-semantically-masked-transformers-for-1/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=semask-semantically-masked-transformers-for-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semask-semantically-masked-transformers-for-1/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=semask-semantically-masked-transformers-for-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semask-semantically-masked-transformers-for-1/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=semask-semantically-masked-transformers-for-1)

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[Jitesh Jain](https://praeclarumjj3.github.io/), [Anukriti Singh](https://anukritisinghh.github.io/), [Nikita Orlov](https://www.linkedin.com/in/nukich74/), [Zilong Huang](https://speedinghzl.github.io/), [Jiachen Li](https://chrisjuniorli.github.io/), [Steven Walton](https://stevenwalton.github.io/about/), [Humphrey Shi](https://www.humphreyshi.com/home)

[[`arXiv`](https://arxiv.org/abs/2112.12782)] [[`pdf`](https://arxiv.org/pdf/2112.12782.pdf)] [[`BibTeX`](#3-citing-semask)]

This repo contains the code for our paper **SeMask: Semantically Masked Transformers for Semantic Segmentation**.

<img src="images/semask.svg" alt='semask' height='600px'>

## Contents
1. [Results](#1-results)
2. [Setup Instructions](#2-setup-instructions)
3. [Citing SeMask](#3-citing-semask)

## 1. Results

`Note:` &dagger; denotes the backbones were pretrained on ImageNet-22k and 384x384 resolution images.

### ADE20K

<!-- | SeMask-T FPN | SeMask Swin-T | 512x512 | 42.06  | 43.36 | 35M | [config](configs/semask_swin/ade20k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_ade20k.py) | [checkpoint](https://drive.google.com/file/d/1L0daUHWQGNGCXHF-cKWEauPSyBV0GLOR/view?usp=sharing) | -->

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |    :---:   |
| SeMask-T FPN | SeMask Swin-T | 512x512 | 42.11  | 43.16 | 35M | [config](configs/semask_swin/ade20k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_ade20k.py) | TBD |
| SeMask-S FPN | SeMask Swin-S | 512x512 | 45.92  | 47.63 | 56M | [config](SeMask-FPN/configs/semask_swin/ade20k/semfpn_semask_swin_small_patch4_window7_512x512_80k_ade20k.py) | [checkpoint](https://drive.google.com/file/d/1QhDG4SyGFtWL5kP9BbBoyPqTuFu7fH_y/view?usp=sharing) |
| SeMask-B FPN | SeMask Swin-B<sup>&dagger;</sup> | 512x512 | 49.35  | 50.98 | 96M | [config](SeMask-FPN/configs/semask_swin/ade20k/semfpn_semask_swin_base_patch4_window12_512x512_80k_ade20k.py) | [checkpoint](https://drive.google.com/file/d/1PXCEhrrUy5TJC4dUp7YDQvaapnMzGT6C/view?usp=sharing) |
| SeMask-L FPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 51.89  | 53.52 | 211M| [config](SeMask-FPN/configs/semask_swin/ade20k/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k.py) | [checkpoint](https://drive.google.com/file/d/1u5flfAQCiQJbMZbZPIlGUGTYBz9Ca7rE/view?usp=sharing) |
| SeMask-L MaskFormer | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 54.75  | 56.15 | 219M | [config](SeMask-MaskFormer/configs/ade20k-150/semask_swin/maskformer_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | [checkpoint](https://drive.google.com/file/d/1KgKQLGv9CcBqeEvOEDdxQ-O6lpMfHBLw/view?usp=sharing) |
| SeMask-L Mask2Former | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 56.41  | 57.52 | 222M | [config](SeMask-Mask2Former/configs/ade20k/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | [checkpoint](https://drive.google.com/file/d/1hN1I4Wv7_1FCPOsfA-5PELn6Xn3b_R8a/view?usp=sharing) |
| SeMask-L Mask2Former FAPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | **56.68**  | 58.00 | 227M | [config](SeMask-FAPN/SeMask-Mask2Former/configs/ade20k/semantic-segmentation/semask_swin/fapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | TBD |
| SeMask-L Mask2Former MSFAPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 56.54  | **58.22** | 224M | [config](SeMask-FAPN/SeMask-Mask2Former/configs/ade20k/semantic-segmentation/semask_swin/msfapn_maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml) | [checkpoint](https://drive.google.com/file/d/1w-DRGufIv3zpDO7rJFv2z5WeLx0pDTJe/view?usp=sharing) |

### Cityscapes

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |    :---:   |
| SeMask-T FPN | SeMask Swin-T | 768x768 | 74.92  | 76.56 | 34M | [config](SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_tiny_patch4_window7_768x768_80k_cityscapes.py) | [checkpoint](https://drive.google.com/file/d/1_JBOJQSUVes-CWs075XyPnuNfG5psELr/view?usp=sharing) |
| SeMask-S FPN | SeMask Swin-S | 768x768 | 77.13  | 79.14 | 56M | [config](SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_small_patch4_window7_768x768_80k_cityscapes.py) | [checkpoint](https://drive.google.com/file/d/1WyT207dZmdwETBUR6aeiqOVfQdUIV_fN/view?usp=sharing) |
| SeMask-B FPN | SeMask Swin-B<sup>&dagger;</sup> | 768x768 | 77.70  | 79.73 | 96M | [config](SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_base_patch4_window12_768x768_80k_cityscapes.py) | [checkpoint](https://drive.google.com/file/d/1-LzVB6XzD7IR0zzE5qmE0EM4ZTv429b4/view?usp=sharing) |
| SeMask-L FPN | SeMask Swin-L<sup>&dagger;</sup> | 768x768 | 78.53  | 80.39 | 211M| [config](SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_large_patch4_window12_768x768_80k_cityscapes.py) | [checkpoint](https://drive.google.com/file/d/1R9DDCmucQ_a_6ZkMGufEZCzJ-_qVMqCB/view?usp=sharing) |
| SeMask-L Mask2Former | SeMask Swin-L<sup>&dagger;</sup> | 512x1024 | **83.97**  | **84.98** | 222M | [config](SeMask-Mask2Former/configs/cityscapes/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_90k.yaml) | [checkpoint](https://drive.google.com/file/d/14fZQWQuBUu2qlpy3wyQ42xs6gQNg2DZX/view?usp=sharing) |

### COCO-Stuff 10k

| Method | Backbone | Crop Size | mIoU | mIoU (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---:     | :---:| :---:          | :---:   | :---:  |    :---:   |
| SeMask-T FPN | SeMask Swin-T | 512x512 | 37.53  | 38.88 | 35M | [config](SeMask-FPN/configs/semask_swin/coco_stuff10k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_coco10k.py) | [checkpoint](https://drive.google.com/file/d/1qhXsJ8H64JPI_DW7CNzhxpHSEG2sKaIl/view?usp=sharing) |
| SeMask-S FPN | SeMask Swin-S | 512x512 | 40.72  | 42.27 | 56M | [config](SeMask-FPN/configs/semask_swin/coco_stuff10k/semfpn_semask_swin_small_patch4_window7_512x512_80k_coco10k.py) | [checkpoint](https://drive.google.com/file/d/1ddXSMQu5ClkbLNMyQdyT0ATaOr86vIkL/view?usp=sharing) |
| SeMask-B FPN | SeMask Swin-B<sup>&dagger;</sup> | 512x512 | 44.63  | 46.30 | 96M | [config](SeMask-FPN/configs/semask_swin/coco_stuff10k/semfpn_semask_swin_base_patch4_window12_512x512_80k_coco10k.py) | [checkpoint](https://drive.google.com/file/d/1pGWI7U9bZJoe4ZaDx7ktWELx-uVN7rL0/view?usp=sharing) |
| SeMask-L FPN | SeMask Swin-L<sup>&dagger;</sup> | 640x640 | 47.47  | 48.54 | 211M| [config](SeMask-FPN/configs/semask_swin/coco_stuff10k/semfpn_semask_swin_large_patch4_window12_640x640_80k_coco10k.py) | [checkpoint](https://drive.google.com/file/d/1F6B9x9pX-SYEth7hdtxeNUeQ3XncOH7G/view?usp=sharing) |

<img src="SeMask-FPN/docs/demo.svg" alt='demo' height='600px'>

## 2. Setup Instructions

We provide the codebase with SeMask incorporated into various models. Please check the setup instructions inside the corresponding folders:

- SeMask-FPN: [Setup Instructions](SeMask-FPN/README.md#2-setup-instructions)
- SeMask-MaskFormer: [Setup Instructions](SeMask-MaskFormer/README.md#2-setup-instructions)
- SeMask-Mask2Former: [Setup Instructions](SeMask-Mask2Former/README.md#2-setup-instructions)
- SeMask-FAPN: [Setup Instructions](SeMask-FAPN/README.md#2-setup-instructions)

## 3. Citing SeMask

```BibTeX
@article{jain2021semask,
  title={SeMask: Semantically Masking Transformer Backbones for Effective Semantic Segmentation},
  author={Jitesh Jain and Anukriti Singh and Nikita Orlov and Zilong Huang and Jiachen Li and Steven Walton and Humphrey Shi},
  journal={arXiv},
  year={2021}
}
```

## Acknowledgements

Code is based heavily on the following repositories: [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MaskFormer](https://github.com/facebookresearch/MaskFormer) and [FaPN-full](https://github.com/ShihuaHuang95/FaPN-full).
