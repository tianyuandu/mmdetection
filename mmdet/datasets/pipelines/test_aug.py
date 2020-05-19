import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale=None, ratio=None, flip=False):
        self.transforms = Compose(transforms)
        assert img_scale or ratio
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_if(self.img_scale, tuple)
        else:
            self.img_scale = ratio if isinstance(ratio, list) else [ratio]
            self.scale_key = 'scale_factor'

        self.flip = flip

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        return repr_str
