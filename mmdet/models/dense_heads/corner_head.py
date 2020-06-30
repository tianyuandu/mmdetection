from math import ceil, log

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob

from mmdet.core import multi_apply
from mmdet.ops import CornerPool, soft_nms
from ..builder import HEADS, build_loss
from ..utils import gaussian_radius, gen_gaussian_target


class BCPool(nn.Module):
    """Bidirectional Corner Pooling Module (TopLeft, BottomRight, etc.)

    Args:
        in_channels (int): Input channels of module.
        pool_direction1 (str): direction of the first CornerPool.
        pool_direction2 (str): direction of the second CornerPool.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 pool_direction1,
                 pool_direction2,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(BCPool, self).__init__()
        self.pool1_conv = ConvModule(
            in_channels, 128, 3, padding=1, norm_cfg=norm_cfg)
        self.pool2_conv = ConvModule(
            in_channels, 128, 3, padding=1, norm_cfg=norm_cfg)

        self.p_conv = ConvModule(
            128, in_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)

        self.conv1 = ConvModule(
            in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            in_channels, in_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.pool1 = CornerPool(pool_direction1)
        self.pool2 = CornerPool(pool_direction2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tensor): Input feature of BCPool.
        Returns:
            conv2 (tensor): Output feature of BCPool.
        """
        pool1_conv = self.pool1_conv(x)
        pool2_conv = self.pool2_conv(x)
        pool1_feat = self.pool1(pool1_conv)
        pool2_feat = self.pool2(pool2_conv)
        p_conv = self.p_conv(pool1_feat + pool2_feat)
        conv1 = self.conv1(x)
        relu = self.relu(p_conv + conv1)
        conv2 = self.conv2(relu)
        return conv2


@HEADS.register_module()
class CornerHead(nn.Module):
    """Head of CornerNet: Detecting Objects as Paired Keypoints.

    Code is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp.py#L73>`_ .  # noqa: E501
    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_num_levels (int): Levels of feature from previous module.
            Default: 2 (HourglassNet-104 outputs the final feature and
            intermediate supervision feature, HourglassNet-52 only outputs
            the final feature, so feat_num_levels for HourglassNet-52 is 1).
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_hmp (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_emb (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_off (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_num_levels=2,
                 corner_emb_channels=1,
                 train_cfg=None,
                 test_cfg=None,
                 loss_hmp=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_emb=dict(
                     type='AssociativeEmbeddingLoss',
                     pull_weight=0.25,
                     push_weight=0.25),
                 loss_off=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)):
        super(CornerHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.corner_offset_channels = 2
        self.feat_num_levels = feat_num_levels
        self.loss_hmp = build_loss(loss_hmp) if loss_hmp is not None else None
        self.loss_emb = build_loss(loss_emb) if loss_emb is not None else None
        self.loss_off = build_loss(loss_off) if loss_off is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead.
        """
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, padding=1),
            nn.Conv2d(feat_channels, out_channels, (1, 1)))

    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers. Including corner heatmap branch
        and corner offset branch. Each branch has two parts: prefix `tl_` for
        top-left and `br_` for bottom-right.
        """
        self.tl_pool, self.br_pool = nn.ModuleList(), nn.ModuleList()
        self.tl_heat, self.br_heat = nn.ModuleList(), nn.ModuleList()
        self.tl_off, self.br_off = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.feat_num_levels):
            self.tl_pool.append(BCPool(self.in_channels, 'top', 'left'))
            self.br_pool.append(BCPool(self.in_channels, 'bottom', 'right'))

            self.tl_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.br_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))

            self.tl_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))
            self.br_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))

    def _init_corner_emb_layers(self):
        """Initialize corner embedding layers. Only include corner embedding
        branch with two parts: prefix `tl_` for top-left and `br_` for
        bottom-right.
        """
        self.tl_emb, self.br_emb = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.feat_num_levels):
            self.tl_emb.append(
                self._make_layers(
                    out_channels=self.corner_emb_channels,
                    in_channels=self.in_channels))
            self.br_emb.append(
                self._make_layers(
                    out_channels=self.corner_emb_channels,
                    in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for CornerHead.
        Including two parts: corner keypoint layers and corner embedding layers
        """
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.feat_num_levels):
            self.tl_heat[i][-1].bias.data.fill_(bias_init)
            self.br_heat[i][-1].bias.data.fill_(bias_init)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: Usually a tuple of corner heatmaps, offset heatmaps and
                    embedding heatmaps.
                tl_heats (list[Tensor]): Top-left corner heatmaps for all
                    levels, each is a 4D-tensor, the channels number is
                    num_classes.
                br_heats (list[Tensor]): Bottom-right corner heatmaps for all
                    levels, each is a 4D-tensor, the channels number is
                    num_classes.
                tl_embs (list[Tensor] | list[None]): Top-left embedding
                    heatmaps for all levels, each is a 4D-tensor or None.
                    If not None, the channels number is corner_emb_channels.
                br_embs (list[Tensor] | list[None]): Bottom-right embedding
                    heatmaps for all levels, each is a 4D-tensor or None.
                    If not None, the channels number is corner_emb_channels.
                tl_offs (list[Tensor]): Top-left offset heatmaps for all
                    levels, each is a 4D-tensor. The channels number is
                    corner_offset_channels.
                br_offs (list[Tensor]): Bottom-right offset heatmaps for all
                    levels, each is a 4D-tensor. The channels number is
                    corner_offset_channels.
        """
        lvl_ind = list(range(self.feat_num_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.
            return_pool (bool): Return corner pool feature or not.
        Returns:
            tuple:
                tl_heat (Tensor): Predicted top-left corner heatmap.
                br_heat (Tensor): Predicted bottom-right corner heatmap.
                tl_emb (Tensor | None): Predicted top-left embedding heatmap.
                    None for `self.with_corner_emb == False`.
                br_emb (Tensor | None): Predicted bottom-right embedding
                    heatmap. None for `self.with_corner_emb == False`.
                tl_off (Tensor): Predicted top-left offset heatmap.
                br_off (Tensor): Predicted bottom-right offset heatmap.
                tl_pool (Tensor): Top-left corner pool feature. Not must have.
                br_pool (Tensor): Bottom-right corner pool feature. Not must
                    have.
        """
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)

        tl_emb, br_emb = None, None
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)

        tl_off = self.tl_off[lvl_ind](tl_pool)
        br_off = self.br_off[lvl_ind](br_pool)

        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)

        return result_list

    def corner_target(self,
                      gt_bboxes,
                      gt_labels,
                      feat_shape,
                      img_shape,
                      with_corner_emb=False,
                      with_guiding_shift=False,
                      with_centripetal_shift=False):
        """Generate corner targets.

        Including corner heatmap, corner offset.
        Optional: corner embedding, corner guiding shift, centripetal shift.
        For CornerNet, we generate corner heatmap, corner offset and corner
            embedding from this function.
        For CentripetalNet, we generate corner heatmap, corner offset, guiding
            shift and centripetal shift from this function.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].
            with_corner_emb (bool): Generate corner embedding target or not.
                Default: False.
            with_guiding_shift (bool): Generate guiding shift target or not.
                Default: False.
            with_centripetal_shift (bool): Generate centripetal shift target or
                not. Default: False.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)

        gt_tl_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_br_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        if with_corner_emb:
            match = []

        if with_guiding_shift:
            gt_tl_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_br_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
        if with_centripetal_shift:
            gt_tl_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_br_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])

        for batch_id in range(batch_size):
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))

                width = ceil(scale_right - scale_left)
                height = ceil(scale_bottom - scale_top)

                radius = gaussian_radius((height, width), min_overlap=0.3)
                radius = max(0, int(radius))

                tl_heatmap = gt_tl_heatmap[batch_id, label]
                br_heatmap = gt_br_heatmap[batch_id, label]
                tl_heatmap = gen_gaussian_target(tl_heatmap,
                                                 [left_idx, top_idx], radius)
                br_heatmap = gen_gaussian_target(br_heatmap,
                                                 [right_idx, bottom_idx],
                                                 radius)

                left_offset = scale_left - left_idx
                top_offset = scale_top - top_idx
                right_offset = scale_right - right_idx
                bottom_offset = scale_bottom - bottom_idx

                gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
                gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
                gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
                gt_br_offset[batch_id, 1, bottom_idx,
                             right_idx] = bottom_offset

                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx],
                                         [bottom_idx, right_idx]])
                if with_guiding_shift:
                    gt_tl_guiding_shift[batch_id, 0, top_idx,
                                        left_idx] = scale_center_x - left_idx
                    gt_tl_guiding_shift[batch_id, 1, top_idx,
                                        left_idx] = scale_center_y - top_idx
                    gt_br_guiding_shift[batch_id, 0, bottom_idx,
                                        right_idx] = right_idx - scale_center_x
                    gt_br_guiding_shift[
                        batch_id, 1, bottom_idx,
                        right_idx] = bottom_idx - scale_center_y
                if with_centripetal_shift:
                    gt_tl_centripetal_shift[batch_id, 0, top_idx,
                                            left_idx] = log(scale_center_x -
                                                            scale_left)
                    gt_tl_centripetal_shift[batch_id, 1, top_idx,
                                            left_idx] = log(scale_center_y -
                                                            scale_top)
                    gt_br_centripetal_shift[batch_id, 0, bottom_idx,
                                            right_idx] = log(scale_right -
                                                             scale_center_x)
                    gt_br_centripetal_shift[batch_id, 1, bottom_idx,
                                            right_idx] = log(scale_bottom -
                                                             scale_center_y)

            if with_corner_emb:
                match.append(corner_match)

        result_list = [
            gt_tl_heatmap, gt_br_heatmap, gt_tl_offset, gt_br_offset
        ]

        if with_corner_emb:
            result_list.append(match)
        if with_guiding_shift:
            result_list.append(gt_tl_guiding_shift)
            result_list.append(gt_br_guiding_shift)
        if with_centripetal_shift:
            result_list.append(gt_tl_centripetal_shift)
            result_list.append(gt_br_centripetal_shift)

        return result_list

    def loss(self,
             tl_heats,
             br_heats,
             tl_embs,
             br_embs,
             tl_offs,
             br_offs,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        targets = self.corner_target(
            gt_bboxes,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'],
            with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.feat_num_levels)]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(
            self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs,
            br_offs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)
        if self.with_corner_emb:
            loss_dict.update({'pull_loss': pull_losses})
            loss_dict.update({'push_loss': push_losses})
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off,
                    targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            targets (list[Tensor]): Corner target generated by `corner_target`.

        Returns:
            det_loss (Tensor): Corner keypoint loss.
            pull_loss (Tensor): Part of AssociativeEmbedding loss.
            push_loss (Tensor): Another part of AssociativeEmbedding loss.
            off_loss (Tensor): Corner offset loss.
        """
        gt_tl_hmp, gt_br_hmp, gt_tl_off, gt_br_off, match = targets

        # Detection loss
        tl_det_loss = self.loss_hmp(
            tl_hmp.sigmoid(),
            gt_tl_hmp,
            avg_factor=max(1,
                           gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_hmp(
            br_hmp.sigmoid(),
            gt_br_hmp,
            avg_factor=max(1,
                           gt_br_hmp.eq(1).sum()))
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        # AssociativeEmbedding loss
        if self.with_corner_emb and self.loss_emb is not None:
            pull_loss, push_loss = self.loss_emb(tl_emb, br_emb, match)
        else:
            pull_loss, push_loss = None, None

        # Offset loss
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_hmp)
        tl_off_loss = self.loss_off(
            tl_off,
            gt_tl_off,
            tl_off_mask,
            avg_factor=max(1, tl_off_mask.sum()))
        br_off_loss = self.loss_off(
            br_off,
            gt_br_off,
            br_off_mask,
            avg_factor=max(1, br_off_mask.sum()))

        off_loss = (tl_off_loss + br_off_loss) / 2.0

        return det_loss, pull_loss, push_loss, off_loss

    def get_bboxes(self,
                   tl_heats,
                   br_heats,
                   tl_embs,
                   br_embs,
                   tl_offs,
                   br_offs,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl_heats[-1][img_id:img_id + 1, :],
                    br_heats[-1][img_id:img_id + 1, :],
                    tl_embs[-1][img_id:img_id + 1, :],
                    br_embs[-1][img_id:img_id + 1, :],
                    tl_offs[-1][img_id:img_id + 1, :],
                    br_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           tl_heat,
                           br_heat,
                           tl_emb,
                           br_emb,
                           tl_off,
                           br_off,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            tl_heat=tl_heat.sigmoid(),
            br_heat=br_heat.sigmoid(),
            tl_off=tl_off,
            br_off=br_off,
            tl_emb=tl_emb,
            br_emb=br_emb,
            img_meta=img_meta,
            K=self.test_cfg.corner_topk,
            kernel=self.test_cfg.local_maximum_kernel,
            distance_threshold=self.test_cfg.distance_threshold)

        if rescale:
            batch_bboxes /= img_meta['scale_factor']

        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view([-1, 1])
        clses = batch_clses.view([-1, 1])

        idx = scores.argsort(dim=0, descending=True)
        bboxes = bboxes[idx].view([-1, 4])
        scores = scores[idx].view(-1)
        clses = clses[idx].view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = (detections[:, -1] > -0.1)
        detections = detections[keepinds]
        labels = clses[keepinds]

        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels,
                                                  self.test_cfg,
                                                  self.num_classes)

        return detections, labels

    def _bboxes_nms(self, bboxes, labels, cfg, num_classes=80):
        out_bboxes = []
        out_labels = []
        for i in range(num_classes):
            keepinds = (labels == i)
            nms_detections = bboxes[keepinds]
            if nms_detections.size(0) == 0:
                continue
            nms_detections, _ = soft_nms(nms_detections, 0.5, 'gaussian')

            out_bboxes.append(nms_detections)
            out_labels += [i for _ in range(len(nms_detections))]

        if len(out_bboxes) > 0:
            out_bboxes = torch.cat(out_bboxes)
            out_labels = torch.Tensor(out_labels)
        else:
            out_bboxes = torch.Tensor(out_bboxes)
            out_labels = torch.Tensor(out_labels)

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.
        Return:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.
        Return:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
        Return:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, K=20):
        """Get top K positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            K (int): Target number. Default: 20.
        Return:
            topk_scores (Tensor): Max scores of each topk keypoint.
            topk_inds (Tensor): Indexes of each topk keypoint.
            topk_clses (Tensor): Categories of each topk keypoint.
            topk_ys (Tensor): Y-coord of each topk keypoint.
            topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(self,
                       tl_heat,
                       br_heat,
                       tl_off,
                       br_off,
                       tl_emb=None,
                       br_emb=None,
                       tl_centripetal_shift=None,
                       br_centripetal_shift=None,
                       img_meta=None,
                       K=100,
                       kernel=3,
                       distance_threshold=0.5,
                       num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_emb (Tensor | None): Top-left corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            br_emb (Tensor | None): Bottom-right corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift (Tensor | None): Top-left centripetal shift
                for current level with shape (N, 2, H, W).
            br_centripetal_shift (Tensor | None): Bottom-right centripetal
                shift for current level with shape (N, 2, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            K (int): Get top K corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            distance_threshold (float): Distance threshold. Top-left and
                bottom-right corner keypoints with feature distance less than
                the threshold will be regarded as keypoints from same object.
            num_dets (int): Num of raw boxes before doing nms.
        Return:
            bboxes (Tensor): Coords of each box.
            scores (Tensor): Scores of each box.
            clses (Tensor): Categories of each box.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = (
            tl_centripetal_shift is not None
            and br_centripetal_shift is not None)
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        tl_heat = self._local_maximum(tl_heat, kernel=kernel)
        br_heat = self._local_maximum(br_heat, kernel=kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, K=K)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, K=K)

        tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
        tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
        br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
        br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

        tl_off = self._transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, K, 1, 2)
        br_off = self._transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]

        if with_centripetal_shift:
            tl_centripetal_shift = self._transpose_and_gather_feat(
                tl_centripetal_shift, tl_inds).view(batch, K, 1, 2).exp()
            br_centripetal_shift = self._transpose_and_gather_feat(
                br_centripetal_shift, br_inds).view(batch, 1, K, 2).exp()

            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]

        # all possible boxes based on top k corners (ignoring class)
        tl_xs *= (inp_w / width)
        tl_ys *= (inp_h / height)
        br_xs *= (inp_w / width)
        br_ys *= (inp_h / height)

        if with_centripetal_shift:
            tl_ctxs *= (inp_w / width)
            tl_ctys *= (inp_h / height)
            br_ctxs *= (inp_w / width)
            br_ctys *= (inp_h / height)

        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]

        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off

        tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
        tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
        br_xs *= br_xs.gt(0.0).type_as(br_xs)
        br_ys *= br_ys.gt(0.0).type_as(br_ys)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()

        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off

            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)

            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys),
                                    dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()

            rcentral = torch.zeros_like(ct_bboxes)
            # magic nums from paper section 4.1
            mu = torch.ones_like(area_bboxes) / 2.4
            mu[area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu

            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) *
                             (rcentral[..., 3] - rcentral[..., 1])).abs()
            dists = area_ct_bboxes / area_rcentral

            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (
                ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (
                ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (
                ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (
                ct_bboxes[..., 3] >= rcentral[..., 3])

        if with_embedding:
            tl_emb = self._transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, K, 1)
            br_emb = self._transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, K)
            dists = torch.abs(tl_emb - br_emb)

        tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
        br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)

        scores = (tl_scores + br_scores) / 2  # scores for all possible boxes

        # tl and br should have same class
        tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
        br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
        cls_inds = (tl_clses != br_clses)

        # reject boxes based on distances
        dist_inds = dists > distance_threshold

        # reject boxes based on widths and heights
        width_inds = (br_xs <= tl_xs)
        height_inds = (br_ys <= tl_ys)

        scores[cls_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1
        scores[dist_inds] = -1
        if with_centripetal_shift:
            scores[tl_ctx_inds] = -1
            scores[tl_cty_inds] = -1
            scores[br_ctx_inds] = -1
            scores[br_cty_inds] = -1

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = self._gather_feat(bboxes, inds)

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = self._gather_feat(clses, inds).float()

        return bboxes, scores, clses
