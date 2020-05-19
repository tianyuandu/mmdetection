import torch
import torch.nn as nn
import numpy as np

# for debug
from matplotlib.pyplot import get_cmap
import cv2


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


# decode the output feature map to detection boxes
def decode_heatmap(tl_heat,
                   br_heat,
                   tl_tag,
                   br_tag,
                   tl_regr,
                   br_regr,
                   img_meta,
                   K=100,
                   kernel=3,
                   ae_threshold=0.5,
                   num_dets=1000):
    batch, _, height, width = tl_heat.size()
    inp_h, inp_w, _ = img_meta['img_shape']

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    tl_xs *= (inp_w / width)
    tl_ys *= (inp_h / height)
    br_xs *= (inp_w / width)
    br_ys *= (inp_h / height)

    x_off = img_meta['border'][2]
    y_off = img_meta['border'][0]

    tl_xs -= torch.Tensor([x_off]).type_as(tl_xs)
    tl_ys -= torch.Tensor([y_off]).type_as(tl_ys)
    br_xs -= torch.Tensor([x_off]).type_as(br_xs)
    br_ys -= torch.Tensor([y_off]).type_as(br_ys)

    tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
    tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
    br_xs *= br_xs.gt(0.0).type_as(br_xs)
    br_ys *= br_ys.gt(0.0).type_as(br_ys)

    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)

    scores = (tl_scores + br_scores) / 2  # scores for all possible boxes

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)  # tl and br should have the same class

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    # tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    # tl_scores = _gather_feat(tl_scores, inds).float()
    # br_scores = br_scores.contiguous().view(batch, -1, 1)
    # br_scores = _gather_feat(br_scores, inds).float()

    # det_bboxes = torch.cat([bboxes, scores], dim=2)

    # detections = torch.cat(
    #     [bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return bboxes, scores, clses


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x*x + y*y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gs = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    # process the border
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
    masked_gaussian = gs[radius-top:radius+bottom, radius-left:radius+right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def vis_density(heatmap):
    cmap = get_cmap('jet')
    color_heatmap = cmap(heatmap, bytes=True)[:, :, :3]
    color_heatmap = color_heatmap[:, :, ::-1]
    return color_heatmap


def vis_density_with_img(heatmap, img):
    color_heatmap = vis_density(heatmap)
    alpha = 0.5
    color_heatmap = cv2.resize(color_heatmap, (img.shape[1], img.shape[0]))
    vis_img = cv2.addWeighted(color_heatmap, alpha, img, 1 - alpha, 0)
    return vis_img
