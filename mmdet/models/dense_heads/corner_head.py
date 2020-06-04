import torch
import torch.nn as nn

from mmdet.core.corner import corner_target, decode_heatmap
from mmdet.ops import soft_nms, TopPool, BottomPool, LeftPool, RightPool
from mmcv.cnn import ConvModule

from ..builder import HEADS, build_loss


class BDPool(nn.Module):
    """ Bi-Directional Pooling Module (TopLeft, BottomRight, etc.)
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
    """ CornerNet: Detecting Objects as Paired Keypoints
    Official github repo: https://github.com/princeton-vl/CornerNet
    Paper : https://arxiv.org/abs/1808.01244
    """
    def __init__(
        self,
        num_classes,
        in_channels,
        emb_dim=1,
        off_dim=2,
        train_cfg=None,
        test_cfg=None,
        loss_hmp=dict(
            type='FocalLoss2D', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_emb=dict(
            type='AELoss', pull_weight=0.25, push_weight=0.25),
        loss_off=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1)):
        super(CornerHead, self).__init__()
        self.num_classes = num_classes - 1
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.offset_dim = off_dim
        self.tl_out_channels = self.num_classes + self.emb_dim + off_dim
        self.br_out_channels = self.num_classes + self.emb_dim + off_dim
        self.loss_hmp = build_loss(loss_hmp)
        self.loss_emb = build_loss(loss_emb)
        self.loss_off = build_loss(loss_off)
        self.train_cfg = train_cfg  # useless
        self.test_cfg = test_cfg

        self._init_layers()

    def _init_layers(self):

        def make_kp_layer(out_dim, cnv_dim=256, curr_dim=256):
            return nn.Sequential(
                ConvModule(cnv_dim, curr_dim, 3, padding=1),
                nn.Conv2d(curr_dim, out_dim, (1, 1)))

        self.tl_pool = BDPool(self.in_channels, TopPool, LeftPool)
        self.br_pool = BDPool(self.in_channels, BottomPool, RightPool)

        self.tl_heat = make_kp_layer(out_dim=self.num_classes)
        self.br_heat = make_kp_layer(out_dim=self.num_classes)

        self.tl_emb = make_kp_layer(out_dim=self.emb_dim)
        self.br_emb = make_kp_layer(out_dim=self.emb_dim)

        self.tl_off = make_kp_layer(out_dim=self.offset_dim)
        self.br_off = make_kp_layer(out_dim=self.offset_dim)

        # intermediate supervision
        self.tl_pool_is = BDPool(self.in_channels, TopPool, LeftPool)
        self.br_pool_is = BDPool(self.in_channels, BottomPool, RightPool)

        self.tl_heat_is = make_kp_layer(out_dim=self.num_classes)
        self.br_heat_is = make_kp_layer(out_dim=self.num_classes)

        self.tl_emb_is = make_kp_layer(out_dim=self.emb_dim)
        self.br_emb_is = make_kp_layer(out_dim=self.emb_dim)

        self.tl_off_is = make_kp_layer(out_dim=self.offset_dim)
        self.br_off_is = make_kp_layer(out_dim=self.offset_dim)

    def init_weights(self):
        """
        -2.19 is the magic number from official github repo.
        Please refer to https://github.com/princeton-vl/CornerNet/issues/13
        """
        self.tl_heat[-1].bias.data.fill_(-2.19)
        self.br_heat[-1].bias.data.fill_(-2.19)
        self.tl_heat_is[-1].bias.data.fill_(-2.19)
        self.br_heat_is[-1].bias.data.fill_(-2.19)

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)) and len(feats) == 2
        x_is, x = feats

        tl_pool = self.tl_pool(x)
        tl_heat = self.tl_heat(tl_pool)
        tl_emb = self.tl_emb(tl_pool)
        tl_off = self.tl_off(tl_pool)

        br_pool = self.br_pool(x)
        br_heat = self.br_heat(br_pool)
        br_emb = self.br_emb(br_pool)
        br_off = self.br_off(br_pool)

        tl_result = torch.cat([tl_heat, tl_emb, tl_off], 1)
        br_result = torch.cat([br_heat, br_emb, br_off], 1)

        tl_pool_is = self.tl_pool_is(x_is)
        tl_heat_is = self.tl_heat_is(tl_pool_is)
        tl_emb_is = self.tl_emb_is(tl_pool_is)
        tl_off_is = self.tl_off_is(tl_pool_is)

        br_pool_is = self.br_pool_is(x_is)
        br_heat_is = self.br_heat_is(br_pool_is)
        br_emb_is = self.br_emb_is(br_pool_is)
        br_off_is = self.br_off_is(br_pool_is)

        tl_result_is = torch.cat([tl_heat_is, tl_emb_is, tl_off_is], 1)
        br_result_is = torch.cat([br_heat_is, br_emb_is, br_off_is], 1)

        return tl_result, br_result, tl_result_is, br_result_is

    def loss(self, 
             tl,
             br,
             tl_is,
             br_is,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        wh = img_metas[0]['pad_shape'][:2]
        targets = corner_target(gt_bboxes, gt_labels, tl, wh, self.num_classes)

        gt_tl_hmp, gt_br_hmp, gt_tl_off, gt_br_off, match = targets

        pd_tl_hmp = tl[:, :self.num_classes, :, :].sigmoid().clamp(
            min=1e-4, max=1 - 1e-4)
        pd_tl_emb = tl[:,
                       self.num_classes:self.num_classes+self.emb_dim, :, :]
        pd_tl_off = tl[:, -self.offset_dim:, :, :]

        pd_br_hmp = br[:, :self.num_classes, :, :].sigmoid().clamp(
            min=1e-4, max=1 - 1e-4)
        pd_br_emb = br[:,
                       self.num_classes:self.num_classes+self.emb_dim, :, :]
        pd_br_off = br[:, -self.offset_dim:, :, :]

        pd_tl_hmp_is = tl_is[:, :self.num_classes, :, :].sigmoid().clamp(
            min=1e-4, max=1 - 1e-4)
        pd_tl_emb_is = tl_is[:, self.num_classes:self.num_classes +
                             self.emb_dim, :, :]
        pd_tl_off_is = tl_is[:, -self.offset_dim:, :, :]

        pd_br_hmp_is = br_is[:, :self.num_classes, :, :].sigmoid().clamp(
            min=1e-4, max=1 - 1e-4)
        pd_br_emb_is = br_is[:, self.num_classes:self.num_classes +
                             self.emb_dim, :, :]
        pd_br_off_is = br_is[:, -self.offset_dim:, :, :]

        # Detection loss
        tl_det_loss = self.loss_hmp(pd_tl_hmp, gt_tl_hmp) + self.loss_hmp(
            pd_tl_hmp_is, gt_tl_hmp)
        br_det_loss = self.loss_hmp(pd_br_hmp, gt_br_hmp) + self.loss_hmp(
            pd_br_hmp_is, gt_br_hmp)
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        # AE loss
        pull, push = self.loss_emb(pd_tl_emb, pd_br_emb, match)
        pull_is, push_is = self.loss_emb(pd_tl_emb_is, pd_br_emb_is, match)
        pull_loss = (pull + pull_is) / 2.0
        push_loss = (push + push_is) / 2.0

        # Offset loss
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_hmp)
        tl_off_loss, br_off_loss = 0, 0
        tl_off_loss += self.loss_off(
            pd_tl_off, gt_tl_off, tl_off_mask, avg_factor=tl_off_mask.sum())
        tl_off_loss += self.loss_off(
            pd_tl_off_is, gt_tl_off, tl_off_mask, avg_factor=tl_off_mask.sum())
        br_off_loss += self.loss_off(
            pd_br_off, gt_br_off, br_off_mask, avg_factor=br_off_mask.sum())
        br_off_loss += self.loss_off(
            pd_br_off_is, gt_br_off, br_off_mask, avg_factor=br_off_mask.sum())

        off_loss = (tl_off_loss + br_off_loss) / 2.0

        return dict(
            det_loss=det_loss,
            pull_loss=pull_loss,
            push_loss=push_loss,
            offset_loss=off_loss)

    def get_bboxes(self,
                   tl,
                   br,
                   tl_is,
                   br_is,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        assert tl.shape[0] == tl_is.shape[0] == len(img_metas)
        assert br.shape[0] == br_is.shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl[img_id:img_id + 1, :],
                    br[img_id:img_id + 1, :],
                    tl_is[img_id:img_id + 1, :],
                    br_is[img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           tl,
                           br,
                           tl_is,
                           br_is,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        tl_heat = tl[:, :self.num_classes, :, :]
        tl_tag = tl[:, self.num_classes:self.num_classes + self.emb_dim, :, :]
        tl_regr = tl[:, -self.offset_dim:, :, :]
        br_heat = br[:, :self.num_classes, :, :]
        br_tag = br[:, self.num_classes:self.num_classes + self.emb_dim, :, :]
        br_regr = br[:, -self.offset_dim:, :, :]

        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = decode_heatmap(
            tl_heat=tl_heat.sigmoid(),
            br_heat=br_heat.sigmoid(),
            tl_tag=tl_tag,
            br_tag=br_tag,
            tl_regr=tl_regr,
            br_regr=br_regr,
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
