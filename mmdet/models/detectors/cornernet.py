import torch
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CornerNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CornerNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def aug_test(self, imgs, img_metas, rescale=False):
        import pdb
        pdb.set_trace()
        img_length = len(imgs)
        results, bboxes, labels = [], [], []
        for i in range(img_length // 2):
            imgs = torch.cat(imgs[2 * i:2 * (i + 1)])
            x = self.extract_feat(imgs)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas[2 * i:2 * (i + 1)], rescale=rescale)
            results.append(bbox_list)
            bboxes.append(bbox_list[0][0])
            labels.append(bbox_list[0][1])

