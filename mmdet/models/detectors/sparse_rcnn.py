from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class SparseRCNN(SingleStageDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self, *args, **kwargs):
        super(SparseRCNN, self).__init__(*args, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        bbox_results = self.bbox_head.simple_test(x, img_metas, rescale=rescale)

        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]
        return bbox_results
