#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net.py --dist-url 'tcp://127.0.0.1:50162' --num-gpus 4 --config-file configs/ade20k/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_160k_res640.yaml