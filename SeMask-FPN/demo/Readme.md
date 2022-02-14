## Inference on a Single Image

You can test on your own images using the following command:

```.bash
python demo.py --img ${PATH_TO_IMAGE} \
        --config ${CONFIG_FILE} \
        --checkpoint ${CHECKPOINT_FILE} \
        [--palette ${PALETTE}] \
        [--device ${DEVICE}] \
        [--opacity ${OPACITY}]
```

Arguments:

- `--img ${PATH_TO_IMAGE}`: Path to the test image.
- `--config ${CONFIG_FILE}`: Path to the config file for model.
- `--palette ${PALETTE}`: Colour palette to be used for the output segmentation map. `Deafult: cityscapes`.
- `--device ${DEVICE}`: Device on which the inference process will run. `Deafult: cuda:0`.
- `--opacity ${OPACITY}`: Opacity in (0, 1] range of painted segmentation map. `Deafult: 0.5`.

### Test with Semask-Tiny FPN on Cityscapes

```.bash
python demo.py --img ${PATH_TO_IMAGE} \
        --config ../configs/semask_swin/cityscapes/semfpn_semask_swin_tiny_patch4_window7_768x768_80k_cityscapes.py \
        --checkpoint semask_tiny_fpn_cityscapes.pth
```
