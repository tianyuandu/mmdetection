import torch
import torch.nn as nn

from mmdet.core import multi_apply
from mmdet.core.corner import corner_target, decode_heatmap
from mmdet.ops import soft_nms, TopPool, BottomPool, LeftPool, RightPool
from mmcv.cnn import ConvModule

from ..builder import HEADS, build_loss


class BDPool(nn.Module):
    """Bi-Directional Pooling Module (TopLeft, BottomRight, etc.)
    """

    def __init__(self,
                 in_channels,
                 pool1,
                 pool2,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(BDPool, self).__init__()
        self.p1_conv = ConvModule(
            in_channels, 128, 3, padding=1, norm_cfg=norm_cfg)
        self.p2_conv = ConvModule(
            in_channels, 128, 3, padding=1, norm_cfg=norm_cfg)

        self.p_conv = ConvModule(
            128, in_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)

        self.conv1 = ConvModule(
            in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            in_channels, in_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        p1_conv = self.p1_conv(x)
        p2_conv = self.p2_conv(x)
        pool1 = self.pool1(p1_conv)
        pool2 = self.pool2(p2_conv)
        p_conv = self.p_conv(pool1 + pool2)
        conv1 = self.conv1(x)
        relu = self.relu(p_conv + conv1)
        conv2 = self.conv2(relu)
        return conv2


@HEADS.register_module()
class CornerHead(nn.Module):
    """CornerNet: Detecting Objects as Paired Keypoints.
    Official github repo: https://github.com/princeton-vl/CornerNet
    Paper : https://arxiv.org/abs/1808.01244
    """
    def __init__(
        self,
        num_classes,
        in_channels,
        corner_emb_channels=1,
        with_corner_offset=True,
        feat_num_levels=2,
        train_cfg=None,
        test_cfg=None,
        loss_hmp=dict(
            type='FocalLoss2D', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_emb=dict(
            type='AELoss', pull_weight=0.25, push_weight=0.25),
        loss_off=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1)):
        super(CornerHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.with_corner_offset = with_corner_offset
        self.corner_offset_channels = 2
        self.feat_num_levels = feat_num_levels
        self.loss_hmp = build_loss(loss_hmp) if loss_hmp is not None else None
        self.loss_emb = build_loss(loss_emb) if loss_emb is not None else None
        self.loss_off = build_loss(loss_off) if loss_off is not None else None
        self.train_cfg = train_cfg  # useless
        self.test_cfg = test_cfg

        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        if out_channels == 0:
            return nn.Sequential()
        else:
            return nn.Sequential(
                ConvModule(in_channels, feat_channels, 3, padding=1),
                nn.Conv2d(feat_channels, out_channels, (1, 1)))

    def _init_corner_kpt_layers(self):
        self.tl_pool, self.br_pool = nn.ModuleList(), nn.ModuleList()
        self.tl_heat, self.br_heat = nn.ModuleList(), nn.ModuleList()
        self.tl_off, self.br_off = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.feat_num_levels):
            self.tl_pool.append(BDPool(self.in_channels, TopPool, LeftPool))
            self.br_pool.append(BDPool(self.in_channels, BottomPool, RightPool))

            self.tl_heat.append(self._make_layers(out_channels=self.num_classes))
            self.br_heat.append(self._make_layers(out_channels=self.num_classes))

            if self.with_corner_offset:
                self.tl_off.append(
                    self._make_layers(out_channels=self.corner_offset_channels))
                self.br_off.append(
                    self._make_layers(out_channels=self.corner_offset_channels))

    def _init_corner_emb_layers(self):
        self.tl_emb, self.br_emb = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.feat_num_levels):
            self.tl_emb.append(
                self._make_layers(out_channels=self.corner_emb_channels))
            self.br_emb.append(
                self._make_layers(out_channels=self.corner_emb_channels))

    def _init_layers(self):
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()

    def init_weights(self):
        """
        -2.19 is the magic number from official github repo.
        Please refer to https://github.com/princeton-vl/CornerNet/issues/13
        """
        for i in range(self.feat_num_levels):
            self.tl_heat[i][-1].bias.data.fill_(-2.19)
            self.br_heat[i][-1].bias.data.fill_(-2.19)

    def forward(self, feats):
        lvl_ind = list(range(self.feat_num_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)

        tl_emb, br_emb = None, None
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)

        tl_off, br_off = None, None
        if self.with_corner_offset:
            tl_off = self.tl_off[lvl_ind](tl_pool)
            br_off = self.br_off[lvl_ind](br_pool)

        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)

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
        wh = img_metas[0]['pad_shape'][:2]

        targets = corner_target(gt_bboxes, gt_labels, tl_heats[-1], wh, self.num_classes)
        mlvl_targets = [targets for _ in range(len(tl_heats))]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(
            self.loss_single,
            tl_heats,
            br_heats,
            tl_embs,
            br_embs,
            tl_offs,
            br_offs,
            mlvl_targets)
        loss_dict = dict(det_loss=det_losses)
        if self.with_corner_emb:
            loss_dict.update({'pull_loss': pull_losses})
            loss_dict.update({'push_loss': push_losses})
        if self.with_corner_offset:
            loss_dict.update({'off_loss': off_losses})
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off, targets):
        gt_tl_hmp, gt_br_hmp, gt_tl_off, gt_br_off, match = targets

        # Detection loss
        tl_det_loss = self.loss_hmp(tl_hmp.sigmoid(), gt_tl_hmp)
        br_det_loss = self.loss_hmp(br_hmp.sigmoid(), gt_br_hmp)
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        # AE loss
        if self.with_corner_emb:
            pull_loss, push_loss = self.loss_emb(tl_emb, br_emb, match)
        else:
            pull_loss, push_loss = None, None

        # Offset loss
        if self.with_corner_offset:
            tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
                gt_tl_hmp)
            br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
                gt_br_hmp)
            tl_off_loss = self.loss_off(
                tl_off, gt_tl_off, tl_off_mask, avg_factor=tl_off_mask.sum())
            br_off_loss = self.loss_off(
                br_off, gt_br_off, br_off_mask, avg_factor=br_off_mask.sum())

            off_loss = (tl_off_loss + br_off_loss) / 2.0
        else:
            off_loss = None

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
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = decode_heatmap(
            tl_heat=tl_heat.sigmoid(),
            br_heat=br_heat.sigmoid(),
            tl_tag=tl_emb,
            br_tag=br_emb,
            tl_regr=tl_off,
            br_regr=br_off,
            img_meta=img_meta,
            K=self.test_cfg.nms_topk,
            kernel=self.test_cfg.nms_pool_kernel,
            ae_threshold=self.test_cfg.ae_threshold)

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
        keepinds = (detections[:, -1] > -0.1)  # 0.05
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
