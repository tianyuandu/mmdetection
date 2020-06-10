import torch
import torch.nn as nn

from mmdet.core import multi_apply
from mmdet.core.corner import corner_target, decode_heatmap
from mmdet.ops import DeformConv
from mmcv.cnn import ConvModule, normal_init

from .corner_head import CornerHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class CentripetalHead(CornerHead):
    """
    """

    def __init__(
        self,
        *args,
        loss_guiding_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
        loss_centripetal_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1),
        **kwargs):
        super(CentripetalHead, self).__init__(*args, **kwargs)
        self.loss_guiding_shift = build_loss(loss_guiding_shift)
        self.loss_centripetal_shift = build_loss(loss_centripetal_shift)

        self._init_layers()

    def _init_centripetal_layers(self):
        self.tl_fadp, self.br_fadp = nn.ModuleList(), nn.ModuleList()
        self.tl_dcn_off, self.br_dcn_off = nn.ModuleList(), nn.ModuleList()
        self.tl_guiding_shift = nn.ModuleList()
        self.br_guiding_shift = nn.ModuleList()
        self.tl_centripetal_shift = nn.ModuleList()
        self.br_centripetal_shift = nn.ModuleList()

        for _ in range(self.feat_num_levels):
            self.tl_fadp.append(
                DeformConv(self.in_channels, self.in_channels, 3, 1, 1))
            self.br_fadp.append(
                DeformConv(self.in_channels, self.in_channels, 3, 1, 1))

            self.tl_guiding_shift.append(self._make_layers(out_channels=2))
            self.br_guiding_shift.append(self._make_layers(out_channels=2))
            # 18 = 3 * 3 * 2
            self.tl_dcn_off.append(nn.Conv2d(2, 18, 1, bias=False))
            self.br_dcn_off.append(nn.Conv2d(2, 18, 1, bias=False))

            self.tl_centripetal_shift.append(self._make_layers(out_channels=2))
            self.br_centripetal_shift.append(self._make_layers(out_channels=2))


    def _init_layers(self):
        self._init_corner_kpt_layers()
        self._init_centripetal_layers()

    def init_weights(self):
        """
        -2.19 is the magic number from official github repo.
        Please refer to https://github.com/princeton-vl/CornerNet/issues/13
        """
        for i in range(self.feat_num_levels):
            self.tl_heat[i][-1].bias.data.fill_(-2.19)
            self.br_heat[i][-1].bias.data.fill_(-2.19)
            normal_init(self.tl_fadp[i], std=0.01)
            normal_init(self.br_fadp[i], std=0.01)
            normal_init(self.tl_dcn_off[i], std=0.1)
            normal_init(self.br_dcn_off[i], std=0.1)

    def forward_single(self, x, lvl_ind):
        tl_heat, br_heat, _, _, tl_off, br_off, tl_pool, br_pool = super(
            CentripetalHead, self).forward_single(x, lvl_ind, return_pool=True)

        tl_guiding_shift = self.tl_guiding_shift[lvl_ind](tl_pool)
        br_guiding_shift = self.br_guiding_shift[lvl_ind](br_pool)

        tl_dcn_offset = self.tl_dcn_off[lvl_ind](tl_guiding_shift.detach())
        br_dcn_offset = self.br_dcn_off[lvl_ind](br_guiding_shift.detach())

        tl_fadp = self.tl_fadp[lvl_ind](tl_pool, tl_dcn_offset)
        br_fadp = self.br_fadp[lvl_ind](br_pool, br_dcn_offset)

        tl_centripetal_shift = self.tl_centripetal_shift[lvl_ind](tl_fadp)
        br_centripetal_shift = self.br_centripetal_shift[lvl_ind](br_fadp)

        result_list = [
            tl_heat, br_heat, tl_off, br_off, tl_guiding_shift,
            br_guiding_shift, tl_centripetal_shift, br_centripetal_shift
        ]
        return result_list

    def loss(self,
             tl_heats,
             br_heats,
             tl_offs,
             br_offs,
             tl_guiding_shifts,
             br_guiding_shifts,
             tl_centripetal_shifts,
             br_centripetal_shifts,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        wh = img_metas[0]['pad_shape'][:2]

        targets = corner_target(
            gt_bboxes,
            gt_labels,
            tl_heats[-1],
            wh,
            self.num_classes,
            with_embedding=False,
            with_guiding_shift=True,
            with_centripetal_shift=True)

        mlvl_targets = [targets for _ in range(len(tl_heats))]
        det_losses, guiding_losses, centripetal_losses, off_losses = multi_apply(
            self.loss_single,
            tl_heats,
            br_heats,
            tl_offs,
            br_offs,
            tl_guiding_shifts,
            br_guiding_shifts,
            tl_centripetal_shifts,
            br_centripetal_shifts,
            mlvl_targets)
        loss_dict = dict(
            det_loss=det_losses,
            off_loss=off_losses,
            guiding_loss=guiding_losses,
            centripetal_loss=centripetal_losses)
        return loss_dict

    def loss_single(self,
                    tl_hmp,
                    br_hmp,
                    tl_off,
                    br_off,
                    tl_guiding_shift,
                    br_guiding_shift,
                    tl_centripetal_shift,
                    br_centripetal_shift,
                    target):

        base_target, centripetal_target = target[:4], target[4:]
        base_target.append(None)

        det_loss, _, _, off_loss = super(CentripetalHead, self).loss_single(
            tl_hmp, br_hmp, None, None, tl_off, br_off, base_target)

        (gt_tl_guiding_shift, gt_br_guiding_shift,
         gt_tl_centripetal_shift, gt_br_centripetal_shift) = centripetal_target

        gt_tl_heatmap, gt_br_heatmap = base_target[:2]
        tl_mask = gt_tl_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_heatmap)
        br_mask = gt_br_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_heatmap)

        # Guiding shift loss
        tl_guiding_loss = self.loss_guiding_shift(
            tl_guiding_shift,
            gt_tl_guiding_shift,
            tl_mask,
            avg_factor=tl_mask.sum())
        br_guiding_loss = self.loss_guiding_shift(
            br_guiding_shift,
            gt_br_guiding_shift,
            br_mask,
            avg_factor=br_mask.sum())
        guiding_loss = (tl_guiding_loss + br_guiding_loss) / 2.0
        # Centripetal shift loss
        tl_centripetal_loss = self.loss_centripetal_shift(
            tl_centripetal_shift,
            gt_tl_centripetal_shift,
            tl_mask,
            avg_factor=tl_mask.sum())
        br_centripetal_loss = self.loss_centripetal_shift(
            br_centripetal_shift,
            gt_br_centripetal_shift,
            br_mask,
            avg_factor=br_mask.sum())
        centripetal_loss = (tl_centripetal_loss + br_centripetal_loss) / 2.0

        return det_loss, off_loss, guiding_loss, centripetal_loss

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
