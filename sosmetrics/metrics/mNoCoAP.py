import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from skimage import measure as skm
from torchvision import transforms

from .base import BaseMetric, time_cost_deco

# cSpell:ignore noco, nocoap, nproc, ridx, cind, rmin, cmin, rmax, cmax, rmin_ind
# cSpell:ignore cmin_ind, rmax_ind, cmax_ind, unb_bg_rmin, unb_bg_rmax, ndim, dets
# cSpell:ignore mpre, mrec, precs, astype, inds, hstack, prec, tpfp, ccol, cenx, ceny
# cSpell:ignore mnocoap, finfo, arange, preds, thrs, nocoaps


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


class NoCoTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """

    def __init__(self, mu=0, sigma=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def get_bboxes(self, gt_semantic_seg):
        bboxes = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def generate_gt_noco_map(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        if len(img.shape) == 3:
            img = img.mean(-1)
        gt_noco_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_noco_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * 3)
            max_bg_wid = int(tgt_w * 3)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x * mesh_x + mesh_y * mesh_y)
            gaussian_like = np.exp(-((dist - self.mu)**2 /
                                     (2.0 * self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # generate contrast weights
            tgt_cell = img[tgt_rmin:tgt_rmax, tgt_cmin:tgt_cmax]
            max_tgt = tgt_cell.max().astype(float)
            contrast_rgn = img[bnd_bg_rmin:bnd_bg_rmax,
                               bnd_bg_cmin:bnd_bg_cmax]
            min_bg = contrast_rgn.min().astype(float)
            contrast_rgn = (contrast_rgn - min_bg) / (max_tgt - min_bg + 0.01)

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian * contrast_rgn
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            gt_noco_map[bnd_bg_rmin:bnd_bg_rmax,
                        bnd_bg_cmin:bnd_bg_cmax] = noco_rgn

        return gt_noco_map

    def get_gt_noco_map(self, img, gt_semantic_seg):

        gt_bboxes = self.get_bboxes(gt_semantic_seg)
        gt_noco_map = self.generate_gt_noco_map(img, gt_bboxes)
        # results['gt_noco_map'] = gt_noco_map
        # if "mask_fields" in results:
        #     results["mask_fields"].append("gt_noco_map")

        return gt_noco_map


def seg2centroid(pred, score_thr=0.5):
    """Convert pred to centroid detection results
    Args:
        pred (np.ndarray): shape (1, H, W)

    Returns:
        det_centroids (np.ndarray): shape (num_dets, 3)
    """

    if pred.ndim == 3:
        pred = pred.squeeze(0)
    # seg_mask = (pred > score_thr).astype(int)
    seg_mask = pred.copy()
    seg_mask[seg_mask > 0] = 1
    gt_labels = skm.label(seg_mask, background=0)
    gt_regions = skm.regionprops(gt_labels)
    centroids = []
    for props in gt_regions:
        ymin, xmin, ymax, xmax = props.bbox
        tgt_pred = pred[ymin:ymax, xmin:xmax]
        ridx, cind = np.unravel_index(np.argmax(tgt_pred, axis=None),
                                      tgt_pred.shape)
        tgt_score = tgt_pred[ridx, cind]
        centroids.append((xmin + cind, ymin + ridx, tgt_score))
        # centroids.append((xmin + cind, ymin + ridx))
    if len(centroids) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    else:
        return np.array(centroids, dtype=np.float32)


def contain_point(cx, cy, bbox):
    """Check if a point is inside a bbox.

    Args:
        cx (float): x coordinate of the point.
        cy (float): y coordinate of the point.
        bbox (ndarray): bounding box of shape (4,).

    Returns:
        bool: Whether the point is inside the bbox.
    """
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def get_gt_bbox(gt_map):
    """Get gt bboxes for evaluation."""
    label_img = skm.label(gt_map, background=0)
    regions = skm.regionprops(label_img)
    bboxes = []
    for region in regions:
        ymin, xmin, ymax, xmax = region.bbox
        bboxes.append([xmin, ymin, xmax, ymax])
    if len(bboxes) == 0:
        return np.zeros((0, 4))
    else:
        return np.array(bboxes)


def get_matched_gt_bbox_idx(cx, cy, gt_bboxes):
    """Get matched gt bbox index.

    Returns:
        int: Matched gt bbox index.
    """
    matched_gt_bbox_idx = -1
    for i, bbox in enumerate(gt_bboxes):
        if contain_point(cx, cy, bbox):
            matched_gt_bbox_idx = i
            break
    return matched_gt_bbox_idx


def tpfp_noco(det_centroids, gt_noco_map, gt_bboxes, noco_thr):
    """Check if detected centroids are true positive or false positive.

    Args:
        det_centroids (ndarray): Detected centroids of this image,
            of shape (m, 3).
        gt_noco_map (ndarray): GT noco maps of this image, of shape (H, W).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        noco_thr (float): NoCo threshold to be considered as matched.
            Default: 0.5.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_dets,).
    """
    num_dets = det_centroids.shape[0]
    num_gts = gt_bboxes.shape[0]

    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)

    # if there is no gt bboxes in this image, then all det centroids
    # are false positives
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    img_hei, img_wid = gt_noco_map.shape
    gt_covered = np.zeros(num_gts, dtype=bool)
    sort_inds = np.argsort(-det_centroids[:, -1])
    for i in sort_inds:
        cx, cy = det_centroids[i, :2]
        crow, ccol = int(np.round(cy)), int(np.round(cx))
        if crow == img_hei:
            crow = img_hei - 1
        if img_wid == ccol:
            ccol = img_wid - 1
        noco_val = gt_noco_map[crow, ccol]
        if noco_val > noco_thr:
            # get matched gt bbox
            matched_gt = get_matched_gt_bbox_idx(cx, cy, gt_bboxes)
            if matched_gt >= 0 and not gt_covered[matched_gt]:
                tp[i] = 1
                gt_covered[matched_gt] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def eval_mnocoap(det_centroids,
                 gt_noco_maps,
                 gt_bboxes,
                 noco_thr=0.5,
                 logger=None,
                 nproc=4):
    """Evaluate mNoCoAP of a dataset.

    Args:

        det_centroids (list[np.ndarray]): each row (cenx, ceny, score).
        gt_noco_maps (list[np.ndarray]): Ground truth normalized contrast map.
            The shape of each np.ndarray is (H, W).
        gt_bboxes (list[np.ndarray]): Ground truth bounding boxes.
        noco_thr (float): NoCo threshold to be considered as matched.
            Default: 0.5.
        score_thr (float): Score threshold to be considered as foregrounds.
            Default: 0.5.
        logger (logging.Logger | str | None): The way to print the mNoCoAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_noco` is used as default.
            Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple: (mNoCoAP, [dict, dict, ...])
    """
    assert len(det_centroids) == len(gt_noco_maps) == len(gt_bboxes)
    num_imgs = len(det_centroids)

    eval_results = []
    tpfp = []
    noco_thrs = [noco_thr for _ in range(num_imgs)]
    for det_centroid, gt_noco_map, gt_bbox, noco_thr in zip(
            det_centroids, gt_noco_maps, gt_bboxes, noco_thrs):
        result = tpfp_noco(det_centroid, gt_noco_map, gt_bbox, noco_thr)
        tpfp.append(result)

    tp, fp = tuple(zip(*tpfp))
    # calculate gt bbox number in total
    num_gts = 0
    for bbox in gt_bboxes:
        num_gts += bbox.shape[0]
    # sort all det bboxes by score, also sort tp and fp
    det_centroids = np.vstack(det_centroids)
    num_dets = det_centroids.shape[0]
    sort_inds = np.argsort(-det_centroids[:, -1])
    tp = np.hstack(tp)[sort_inds]
    fp = np.hstack(fp)[sort_inds]
    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gts, eps)
    precisions = tp / np.maximum((tp + fp), eps)
    ap = average_precision(recalls, precisions, mode='area')
    eval_results.append({
        'num_gts': num_gts,
        'num_dets': num_dets,
        'recall': recalls,
        'precision': precisions,
        'ap': ap
    })
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    return mean_ap, eval_results


class mNoCoAP(BaseMetric):

    def __init__(self, conf_thr: float=0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.conf_thr = conf_thr
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: torch.Tensor, preds: torch.Tensor,
               batch: torch.Tensor) -> None:

        def evaluate_worker(self, label: torch.Tensor, pred: torch.Tensor,
                            batch: torch.Tensor):
            # pred = pred>self.conf_thr
            # pred = pred.to(torch.int64)
            eval_mnocoap = self._compute_mnocoap_metrics(pred, label, batch)
            with self.lock:
                self.total_mnocoap += eval_mnocoap
                self.num_batch += 1

        if isinstance(labels, (np.ndarray, torch.Tensor)):
            evaluate_worker(self, labels, preds, batch)

        elif isinstance(labels, (list, tuple)):
            threads = [
                threading.Thread(
                    target=evaluate_worker,
                    args=(self, labels[i], preds[i], batch[i]),
                ) for i in range(len(labels))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplementedError

    @time_cost_deco
    def get(self):

        self.mnocoap = self.total_mnocoap / self.num_batch

        if self.print_table:
            table = PrettyTable()
            table.add_column('mNoCoAP', ['{:.4f}'.format(self.mnocoap)])
            print(table)

        return self.mnocoap

    @property
    def table(self):
        df = pd.DataFrame(self.mnocoap.reshape(1, 1))
        df.columns = ['mNoCoAP']
        return df

    def reset(self):
        self.total_mnocoap = 0
        self.num_batch = 0

    def _compute_mnocoap_metrics(self, preds: torch.Tensor,
                                 labels: torch.Tensor,
                                 batch: torch.Tensor) -> np.ndarray:

        best_mnocoap = -np.inf
        gray_transform = transforms.Grayscale(num_output_channels=1)
        noco_targets_instance = NoCoTargets()
        results = []

        for i in range(len(labels)):
            pred_label = preds[i].squeeze().cpu().numpy().astype(np.int64)
            label = labels[i].squeeze().cpu().numpy()
            input = batch[i]
            input = gray_transform(input).squeeze().cpu().numpy()
            pred_label[pred_label <= 0] = 0
            det_centroids = seg2centroid(pred_label, self.conf_thr)
            # 获取每个样本的非重叠区域地面真实分割图和边界框
            gt_noco_map = noco_targets_instance.get_gt_noco_map(input, label)
            gt_bbox = get_gt_bbox(label)

            results.append({
                'det_centroids': det_centroids,
                'gt_noco_map': gt_noco_map,
                'gt_bbox': gt_bbox
            })
        noco_thrs = np.linspace(.1,
                                0.9,
                                int(np.round((0.9 - .1) / .1)) + 1,
                                endpoint=True)
        noco_thrs = [noco_thrs] if isinstance(noco_thrs, float) else noco_thrs

        det_centroids = []
        gt_noco_maps = []
        gt_bboxes = []

        # 迭代 self.results 并提取每个结果中的数据
        for result in results:
            det_centroids.append(result['det_centroids'])
            gt_noco_maps.append(result['gt_noco_map'])
            gt_bboxes.append(result['gt_bbox'])

        eval_results = OrderedDict()
        mean_nocoaps = []
        for noco_thr in noco_thrs:
            mean_nocoap, _ = eval_mnocoap(det_centroids,
                                          gt_noco_maps,
                                          gt_bboxes,
                                          noco_thr=noco_thr,
                                          logger=None)
            mean_nocoaps.append(mean_nocoap)
            eval_results[f'NoCoAP{int(noco_thr * 100):02d}'] = round(
                mean_nocoap, 3)
        eval_results['mNoCoAP'] = sum(mean_nocoaps) / len(mean_nocoaps)
        print("eval_results['mNoCoAP']:", eval_results['mNoCoAP'])
        if best_mnocoap < eval_results['mNoCoAP']:
            best_mnocoap = eval_results['mNoCoAP']
        print("best eval_results['mNoCoAP']:", best_mnocoap)
        e_mnocoap = eval_results['mNoCoAP']

        return e_mnocoap

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}')
