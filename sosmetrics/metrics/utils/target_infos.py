from typing import List, Tuple

import numpy as np
import torch
from skimage.measure._regionprops import RegionProperties

from .bbox_overlaps import bbox_overlaps
from .mask_overlaps import target_mask_iou


def calculate_target_infos(coord_label: List[RegionProperties],
                           coord_pred: List[RegionProperties], img_h: int,
                           img_w) -> Tuple:
    """calculate distances between label and pred. single image.

        Args:
            coord_label (List[RegionProperties]): measure.regionprops(label)
            coord_pred (List[RegionProperties]): measure.regionprops(pred)
            img (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            np.ndarray: distances between label and pred. (num_lbl * num_pred)
        """

    num_lbl = len(coord_label)  # number of label
    num_pred = len(coord_pred)  # number of pred

    if num_lbl * num_pred == 0:
        empty = np.empty((num_lbl, num_pred))
        return empty, empty, empty  # gt=0 or pred=0, (num_lbl, 0) or (0, num_pred)

    # eul distance
    centroid_lbl = np.array([prop.centroid for prop in coord_label
                             ]).astype(np.float32)  # N*2
    centroid_pred = np.array([prop.centroid for prop in coord_pred
                              ]).astype(np.float32)  # M*2
    eul_distance = np.linalg.norm(centroid_lbl[:, None, :] -
                                  centroid_pred[None, :, :],
                                  axis=-1)  # num_lbl * num_pred

    # bbox iou
    bbox_lbl = torch.tensor([prop.bbox for prop in coord_label],
                            dtype=torch.float32)  # N*4
    bbox_pred = torch.tensor([prop.bbox for prop in coord_pred],
                             dtype=torch.float32)  # M*4
    bbox_iou = bbox_overlaps(bbox_lbl, bbox_pred, mode='iou',
                             is_aligned=False).numpy()  # num_lbl * num_pred

    # mask iou
    pixel_coords_lbl = [prop.coords for prop in coord_label]
    pixel_coords_pred = [prop.coords for prop in coord_pred]
    gt_mask = np.zeros((num_lbl, img_h, img_w))  # [num_lbl, H, W]
    pred_mask = np.zeros((num_pred, img_h, img_w))  # [num_pred, H, W]
    for i in range(num_lbl):
        gt_mask[i, pixel_coords_lbl[i][:, 0], pixel_coords_lbl[i][:, 1]] = 1
    for i in range(num_pred):
        pred_mask[i, pixel_coords_pred[i][:, 0], pixel_coords_pred[i][:,
                                                                      1]] = 1
    mask_iou = target_mask_iou(gt_mask, pred_mask)  # num_lbl * num_pred

    return eul_distance, mask_iou, bbox_iou
