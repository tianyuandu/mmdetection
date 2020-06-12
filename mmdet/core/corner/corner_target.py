import torch
import numpy as np
from .kp_utils import gaussian_radius, draw_gaussian


def corner_target(gt_bboxes,
                  gt_labels,
                  feats,
                  imgscale,
                  num_classes=80,
                  with_embedding=True,
                  with_guiding_shift=False,
                  with_centripetal_shift=False):
    b, _, h, w = feats.size()
    im_h, im_w = imgscale

    width_ratio = float(w / im_w)
    height_ratio = float(h / im_h)

    gt_tl_heatmap = np.zeros([b, num_classes, h, w])
    gt_br_heatmap = np.zeros([b, num_classes, h, w])

    gt_tl_offsets = np.zeros([b, 2, h, w])
    gt_br_offsets = np.zeros([b, 2, h, w])

    match = []

    if with_guiding_shift:
        gt_tl_guiding_shift = np.zeros([b, 2, h, w])
        gt_br_guiding_shift = np.zeros([b, 2, h, w])

    if with_centripetal_shift:
        gt_tl_centripetal_shift = np.zeros([b, 2, h, w])
        gt_br_centripetal_shift = np.zeros([b, 2, h, w])

    for b_id in range(b):
        if with_embedding:
            corner_match = []
        for box_id in range(len(gt_labels[b_id])):
            tl_x, tl_y, br_x, br_y = gt_bboxes[b_id][box_id]
            ct_x = (tl_x + br_x) / 2.0
            ct_y = (tl_y + br_y) / 2.0
            label = gt_labels[b_id][box_id]

            ftlx = float(tl_x * width_ratio)
            fbrx = float(br_x * width_ratio)
            ftly = float(tl_y * height_ratio)
            fbry = float(br_y * height_ratio)
            fctx = float(ct_x * width_ratio)
            fcty = float(ct_y * height_ratio)

            tl_x_idx = int(min(ftlx, w - 1))
            br_x_idx = int(min(fbrx, w - 1))
            tl_y_idx = int(min(ftly, h - 1))
            br_y_idx = int(min(fbry, h - 1))

            width = int(fbrx - ftlx + 0.5)
            height = int(fbry - ftly + 0.5)

            radius = gaussian_radius((height, width), min_overlap=0.3)
            radius = max(0, int(radius))

            draw_gaussian(gt_tl_heatmap[b_id, label.long()],
                          [tl_x_idx, tl_y_idx], radius)
            draw_gaussian(gt_br_heatmap[b_id, label.long()],
                          [br_x_idx, br_y_idx], radius)

            tl_x_offset = ftlx - tl_x_idx
            tl_y_offset = ftly - tl_y_idx
            br_x_offset = fbrx - br_x_idx
            br_y_offset = fbry - br_y_idx

            gt_tl_offsets[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_offset
            gt_tl_offsets[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_offset
            gt_br_offsets[b_id, 0, br_y_idx, br_x_idx] = br_x_offset
            gt_br_offsets[b_id, 1, br_y_idx, br_x_idx] = br_y_offset

            if with_embedding:
                corner_match.append([[tl_y_idx, tl_x_idx],
                                     [br_y_idx, br_x_idx]])

            if with_guiding_shift:
                gt_tl_guiding_shift[b_id, 0, tl_y_idx,
                                    tl_x_idx] = fctx - tl_x_idx
                gt_tl_guiding_shift[b_id, 1, tl_y_idx,
                                    tl_x_idx] = fcty - tl_y_idx
                gt_br_guiding_shift[b_id, 0, br_y_idx,
                                    br_x_idx] = br_x_idx - fctx
                gt_br_guiding_shift[b_id, 1, br_y_idx,
                                    br_x_idx] = br_y_idx - fctx

            if with_centripetal_shift:
                gt_tl_centripetal_shift[b_id, 0, tl_y_idx,
                                        tl_x_idx] = np.log(fctx - ftlx)
                gt_tl_centripetal_shift[b_id, 1, tl_y_idx,
                                        tl_x_idx] = np.log(fcty - ftly)
                gt_br_centripetal_shift[b_id, 0, br_y_idx,
                                        br_x_idx] = np.log(fbrx - fctx)
                gt_br_centripetal_shift[b_id, 1, br_y_idx,
                                        br_x_idx] = np.log(fbry - fcty)

        if with_embedding:
            match.append(corner_match)

    gt_tl_heatmap = torch.from_numpy(gt_tl_heatmap).type_as(feats)
    gt_br_heatmap = torch.from_numpy(gt_br_heatmap).type_as(feats)
    gt_tl_offsets = torch.from_numpy(gt_tl_offsets).type_as(feats)
    gt_br_offsets = torch.from_numpy(gt_br_offsets).type_as(feats)

    if with_guiding_shift:
        gt_tl_guiding_shift = torch.from_numpy(gt_tl_guiding_shift).type_as(feats)
        gt_br_guiding_shift = torch.from_numpy(gt_br_guiding_shift).type_as(feats)

    if with_guiding_shift:
        gt_tl_centripetal_shift = torch.from_numpy(gt_tl_centripetal_shift).type_as(feats)
        gt_br_centripetal_shift = torch.from_numpy(gt_br_centripetal_shift).type_as(feats)

    result_list = [gt_tl_heatmap, gt_br_heatmap, gt_tl_offsets, gt_br_offsets]

    if with_embedding:
        result_list.append(match)

    if with_guiding_shift:
        result_list.append(gt_tl_guiding_shift)
        result_list.append(gt_br_guiding_shift)

    if with_centripetal_shift:
        result_list.append(gt_tl_centripetal_shift)
        result_list.append(gt_br_centripetal_shift)

    return result_list
