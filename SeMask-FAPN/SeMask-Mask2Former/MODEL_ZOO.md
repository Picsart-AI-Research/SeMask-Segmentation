# MaskFormer Model Zoo and Baselines

## Detectron2 ImageNet Pretrained Models

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available:

* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).
* [R-103.pkl](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers (a.k.a. ResNet101c in our paper). We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

Note: below are available pretrained models in Detectron2 that we do not use in our paper.
* [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
* [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
* [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.

#### Third-party ImageNet Pretrained Models

Our paper also uses ImageNet pretrained models that are not part of Detectron2, please refer to [tools](https://github.com/facebookresearch/MaskFormer/tree/master/tools) to get those pretrained models.