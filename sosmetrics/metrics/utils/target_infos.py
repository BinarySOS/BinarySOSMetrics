from typing import List, Tuple

import cv2
import numpy as np
import torch
from skimage import measure
from skimage.measure._regionprops import RegionProperties

from .bbox_overlaps import bbox_overlaps
from .mask_overlaps import target_mask_iou
from .misc import convert2gray


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
            np.ndarray: distances, mask iou, bbox iou between label and pred.
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


def _get_dilated(image: np.ndarray,
                 dilate_kernel_size: List[int] = [0, 0]) -> np.ndarray:
    """_summary_

    Args:
        image (np.ndarray): hwc of np.ndarray.
    """

    kernel = np.ones(dilate_kernel_size, np.uint8)
    if kernel.sum() == 0:
        return image
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    return dilated_image


def get_label_coord_and_gray(
        label: np.ndarray) -> Tuple[RegionProperties, np.ndarray]:
    cur_label = label > 0  # sometimes is 0-1, force to 255.
    cur_label = cur_label.astype('uint8')
    cur_label = convert2gray(cur_label).astype('int64')
    cur_label = measure.label(cur_label, connectivity=2)
    coord_label = measure.regionprops(cur_label)
    return coord_label, cur_label


def get_pred_coord_and_gray(
        pred: np.ndarray,
        conf_thr: float,
        dilate_kernel: List[int] = [0,
                                    0]) -> Tuple[RegionProperties, np.ndarray]:
    cur_pred = pred >= conf_thr
    cur_pred = cur_pred.astype('uint8')
    # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
    # 'int64' is for measure.label().
    cur_pred = convert2gray(cur_pred)
    if dilate_kernel != [0, 0]:
        cur_pred = _get_dilated(cur_pred, dilate_kernel).astype('int64')
    else:
        cur_pred = cur_pred.astype('int64')
    pred_img = measure.label(cur_pred, connectivity=2)
    coord_pred = measure.regionprops(pred_img)
    return coord_pred, cur_pred


def second_match_method(eul_dis: np.ndarray, mask_iou: np.ndarray,
                        bbox_iou: np.ndarray, match_mode: str) -> np.ndarray:
    """Second match func. Supports 4 secondary matching schemes.
    1. mask: mask_iou is used to mark non-overlapping data pairs.
    2. bbox: bbox_iou is used to mark non-overlapping data pairs.
    3. mask_plus: will return the sum of eul_dis and (1-mask_iou).
    4. bbox_plus: will return the sum of eul_dis and (1-bbox_iou).

        The reason for adding (1-iou) is to maintain the same trend as distance, \
            i.e., smaller means closer to gt.

    Args:
        eul_dis (np.ndarray): N*M matrix, N is number of gt, M is number of pred.
        mask_iou (np.ndarray): Like eul_dis, but the value is mask_iou.
        bbox_iou (np.ndarray): Like eul_dis, but the value is bbox_iou.
        match_mode (str): 'mask', 'bbox', 'mask_plus' and 'bbox_plus'.

    Returns:
        np.ndarray: secondary matching results, N*M matrix, N is number of gt, M is number of pred.
    """
    eul_dis = eul_dis.copy()
    mask_iou = mask_iou.copy()
    bbox_iou = bbox_iou.copy()

    if match_mode == 'mask':
        mask_iou[mask_iou == 0.] = np.inf
        mask_iou[mask_iou != np.inf] = 0.
        distances = eul_dis + mask_iou

    elif match_mode == 'bbox':
        bbox_iou[bbox_iou == 0.] = np.inf
        bbox_iou[bbox_iou != np.inf] = 0.
        distances = eul_dis + bbox_iou

    elif match_mode == 'mask_plus':
        distances = eul_dis + (1 - mask_iou)

    elif match_mode == 'bbox_plus':
        distances = eul_dis + (1 - bbox_iou)

    else:
        raise ValueError(
            "Invalid second match mode, second match mode should be one of ['mask', 'bbox', 'mask_plus', 'bbox_plus']"
        )

    return distances
