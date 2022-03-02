import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
import os
import matplotlib.pyplot as plt

ignore_label= 255

id_to_trainid = {-1: -1, 0: ignore_label, 1: ignore_label, 2: ignore_label, 
                3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label, 
                7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4, 
                14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5, 
                18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 
                28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

class BaseSegmentor(nn.Module):
    """Base class for segmentors."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseSegmentor, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in segmentor.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info(f'load model from: {pretrained}')

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_inference_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

    def show_result(self,
                    i,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        
        assert len(self.CLASSES) in [19, 150, 171]
        
        img = mmcv.imread(img)
        img = img.copy()
        h, w = img.shape[:2]
        seg = result[0]
        seg = mmcv.imresize(seg, (w, h), interpolation='nearest')
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        pred = color_seg.copy()
        color_seg = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        color_seg = color_seg.astype(np.uint8)
        pred = pred.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        save_file_pred = os.path.join(out_file[0], str(i) + '_PRED.png')
        save_file_img = os.path.join(out_file[0], str(i) + '_IMG.png')
        save_file_gt = os.path.join(out_file[0], str(i) + '_GT.png')
        save_file_overlap = os.path.join(out_file[0], str(i) + '_OVERLAP.png')

        if len(self.CLASSES) == 19:
            gt_file = out_file[1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            gt_file = gt_file.replace('/leftImg8bit/', '/gtFine/')   
        elif len(self.CLASSES) == 150:
            gt_file = out_file[1].replace('/images/', '/annotations/')
            gt_file = gt_file.replace('.jpg', '.png')
        elif len(self.CLASSES) == 171:
            gt_file = out_file[1].replace('/images/', '/annotations/')
            gt_file = gt_file.replace('.jpg', '_labelTrainIds.png')
        
        gt = mmcv.imread(gt_file, flag='grayscale')
        gt = mmcv.imresize(gt, (w, h), interpolation='nearest')
        
        if len(self.CLASSES) == 19:
            gt = id2trainId(gt, id_to_trainid)
        elif len(self.CLASSES) == 150:
            gt = gt - 1
        elif len(self.CLASSES) == 171:
            gt = gt - 1
            
        color_gt = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_gt[gt == label, :] = color
        # convert to BGR
        color_gt = color_gt[..., ::-1]

        color_gt = color_gt.astype(np.uint8)

        if show:
            mmcv.imshow(color_seg, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(color_seg, save_file_overlap)
            mmcv.imwrite(img, save_file_img)
            mmcv.imwrite(color_gt, save_file_gt)
            mmcv.imwrite(pred, save_file_pred)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
    
    def grid_maps(self, feats):
        c = feats.shape[1]
        
        gh = int(np.sqrt(c))
        gw = c // gh
        idx = gw * gh

        max_num = torch.max(feats[:, :idx]).item()
        min_num = torch.min(feats[:, :idx]).item()
        feats = feats[:,:idx].cpu() * 255 / (max_num - min_num) 
        feats = np.asarray(feats, dtype=np.float32)
        feats = np.rint(feats).clip(0, 255).astype(np.uint8)

        _N, C, H, W = feats.shape

        feats = feats.reshape(gh, gw, 1, H, W)
        feats = feats.transpose(0, 3, 1, 4, 2)
        feats = feats.reshape(gh * H, gw * W, 1)

        return feats[:, :, 0], str(H)

    def save_maps(self,
                    i,
                    feat_maps,
                    sem_maps,
                    out_file=None):
        
        
        for ft in feat_maps:
            ft, dim = self.grid_maps(ft)
            save_file_ft = os.path.join(out_file, str(i) + f'_{dim}_FEAT.png')
            plt.imsave(save_file_ft, ft, cmap=plt.cm.viridis)

        for sem_ft in sem_maps:
            sem_ft, dim = self.grid_maps(sem_ft)
            save_file_ft = os.path.join(out_file, str(i) + f'_{dim}_SEM.png')
            plt.imsave(save_file_ft, sem_ft, cmap=plt.cm.viridis)

