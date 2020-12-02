import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_bbox_coder,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SparseRCNNHead(CascadeRoIHead):
    """
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 num_proposals=100,
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign',
                        output_size=7,
                        sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                 bbox_head=dict(
                    type='DIIHead',
                    num_classes=80,
                    num_fcs=2,
                    num_heads=8,
                    num_cls_fcs=1,
                    num_reg_fcs=3,
                    feedforward_channels=2048,
                    hidden_channels=256,
                    dropout=0.0,
                    roi_feat_size=7,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    loss_cls=dict(),
                    loss_bbox=dict(),
                    loss_iou=dict()),
                 bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.]),
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        super(SparseRCNNHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.init_proposal_layers()
        self.init_proposal_weights()

    def init_weights(self):
        super(SparseRCNNHead, self).init_weights(pretrained=False)

    def init_proposal_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_proposal_weights(self):
        """Initialize the weights in proposal layers."""
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def forward_dummy(self, x):
        """Dummy forward function."""
        # bbox head
        outs = ()
        proposals = self._decode_init_proposals()
        rois = bbox2roi(proposals)
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, proposals, proposal_features):
        """Box head forward function used in both training and testing."""
        N = len(proposals)
        num_proposals = proposals[0].shape[0]
        rois = bbox2roi(proposals)

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_delta, proposal_features = bbox_head(bbox_feats,
                                                             proposal_features)

        bbox_pred = self.bbox_coder.decode(
            torch.cat(proposals, dim=0),
            bbox_delta.view(-1, 4),
            wh_ratio_clip=16 / 100000)
        bbox_pred = bbox_pred.view(N, num_proposals, -1).unbind(0)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            proposal_features=proposal_features)

        return bbox_results

    def _bbox_forward_train(self, stage, x, proposals, proposal_features,
                            gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        bbox_results = self._bbox_forward(stage, x, proposals, proposal_features)

        # TODO
        # bbox_targets = self.bbox_head[stage].get_targets(
        #     gt_bboxes, gt_labels)
        # loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
        #                                        bbox_results['bbox_pred'], rois,
        #                                        *bbox_targets)

        # bbox_results.update(
        #     loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _decode_init_proposals(self, img_metas=None):
        proposals = self.init_proposal_bboxes.weight.clone()
        x_center, y_center, w, h = proposals.unbind(-1)
        proposals = torch.stack(
            [(x_center - 0.5 * w), (y_center - 0.5 * h),
             (x_center + 0.5 * w), (y_center + 0.5 * h)], dim=-1)
        if img_metas is not None:
            if not isinstance(img_metas, list):
                img_metas = [img_metas]
            wh = img_metas[0]['batch_intput_shape'][:2]
            whwh = torch.tensor(wh + wh).type_as(proposals)
            proposals = proposals * whwh
        return proposals

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposals = self._decode_init_proposals(img_metas)
        proposals = [proposals.clone() for _ in range(num_imgs)]

        proposal_feats = self.init_proposal_features.weight
        proposal_feats = proposal_feats[None].repeat(num_imgs, 1, 1)

        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            lw = self.stage_loss_weights[i]

            if self.with_bbox or self.with_mask:
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, proposals,
                                                    proposal_feats,
                                                    gt_bboxes, gt_labels)

            # for name, value in bbox_results['loss_bbox'].items():
            #     losses[f's{i}.{name}'] = (
            #         value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            proposals = bbox_results['bbox_pred']
            proposal_feats = bbox_results['proposal_features']
        return losses

    def simple_test(self, x, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposals = self._decode_init_proposals(img_metas)
        proposals = [proposals.clone() for _ in range(num_imgs)]

        proposal_feats = self.init_proposal_features.weight
        proposal_feats = proposal_feats[None].repeat(num_imgs, 1, 1)

        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        for i in range(self.num_stages):
            self.current_stage = i
            # bbox head forward
            bbox_results = self._bbox_forward(i, x, proposals, proposal_feats)

            proposals = bbox_results['bbox_pred']
            proposal_feats = bbox_results['proposal_features']

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        import pdb
        pdb.set_trace()
        if self.bbox_head.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()

        '''
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage

        for i in range(self.num_stages):

            # split batch bbox prediction back to each image
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
           else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        '''

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        '''
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[[]
                                for _ in range(self.mask_head[-1].num_classes)]
                               ]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]
        '''
        raise NotImplementedError
