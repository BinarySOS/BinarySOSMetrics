import threading
import time
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.measure._regionprops import RegionProperties

from .base import BaseMetric
from .utils import (_TYPES, _adjust_conf_thr_arg, _adjust_dis_thr_arg,
                    bbox_overlaps, convert2gray, convert2iterable,
                    target_mask_iou)


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


def _get_label_coord(label: np.ndarray) -> List[RegionProperties]:
    label = label > 0  # sometimes is 0-1, force to 255.
    label = label.astype('uint8')
    label = convert2gray(label).astype('int64')
    label = measure.label(label, connectivity=2)
    coord_label = measure.regionprops(label)
    return coord_label


def _get_pred_coord(
        pred: np.ndarray,
        conf_thr: float,
        dilate_kernel_size: List[int] = [0, 0]) -> List[RegionProperties]:
    cur_pred = pred >= conf_thr
    cur_pred = cur_pred.astype('uint8')
    # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
    # 'int64' is for measure.label().

    cur_pred = convert2gray(cur_pred)
    dilated_pred = _get_dilated(cur_pred, dilate_kernel_size).astype('int64')

    pred_img = measure.label(dilated_pred, connectivity=2)
    coord_pred = measure.regionprops(pred_img)
    return coord_pred


class BinaryCenterMetric(BaseMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: float = 0.5,
                 dilate_kernel_size: List[int] = [7, 7],
                 match_alg: str = 'hungarian',
                 iou_mode: str = 'none',
                 **kwargs: Any):
        """
        TP: True Positive, GT is Positive and Pred is Positive, If Euclidean Distance < threshold, matched.
        FN: False Negative, GT is Positive and Pred is Negative.
        FP: False Positive, GT is Negative and Pred is Positive. If Euclidean Distance > threshold, not matched.
        Recall: TP/(TP+FN).
        Precision: TP/(TP+FP).
        F1: 2*Precision*Recall/(Precision+Recall).
        .get will return Precision, Recall, F1 in array.

        Args:
            dis_thr (Union[List[float], int], optional): dis_thr of Euclidean distance,
                - If set to an `int` , will use this value to distance threshold.
                - If set to an `list` of float or int, will use the indicated thresholds \
                    in the list as bins for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                bins for the calculation.
                if List, closed interval.
                Defaults to [1, 10].
            cong_thr (float, optional): confidence threshold. Defaults to 0.5.
            dilate_kernel_size (List[int], optional): kernel size of cv2.dilated.
                The difference is that when [0, 0], the dilation algorithm will not be used. Defaults to [7, 7].
            match_alg (str, optional): 'hungarian' or 'forloop' to match pred and gt,
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle.Usually, hungarian is fast and accurate. Defaults to 'hungarian'.
            iou_mode (str, optional): 'none', 'mask_iou', 'bbox_iou'. if not 'none', (eul_dis + '**_iou') for match.
        """

        super().__init__(**kwargs)
        self.dis_thr = _adjust_dis_thr_arg(dis_thr)
        self.conf_thr = np.array([conf_thr])
        self.match_alg = match_alg
        self.dilate_kernel_size = dilate_kernel_size
        self.iou_mode = iou_mode
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels: _TYPES, preds: _TYPES) -> None:
        """Support CHW, BCHW, HWC,BHWC, Image Path, or in their list form (except BHWC/BCHW),
            like [CHW, CHW, ...], [HWC, HWC, ...], [Image Path, Image Path, ...].

            Although support Image Path, but not recommend.
            Note : All preds are probabilities image from 0 to 1 in default.
            If images, Preds must be probability image from 0 to 1 in default.
            If path, Preds must be probabilities image from 0-1 in default, if 0-255,
            we are force /255 to 0-1 to probability image.
        Args:
            labels (_TYPES): Ground Truth images or image paths in list or single.
            preds (_TYPES): Preds images or image paths in list or single.
        """

        def evaluate_worker(label, pred):
            coord_label = _get_label_coord(label)
            coord_pred = _get_pred_coord(pred.copy(), self.conf_thr,
                                         self.dilate_kernel_size)
            distances = self._calculate_infos(coord_label, coord_pred, pred)
            for idx, threshold in enumerate(self.dis_thr):
                TP, FN, FP = self._calculate_tp_fn_fp(distances.copy(),
                                                      threshold)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

        start_time = time.time()

        labels, preds = convert2iterable(labels, preds)

        for i in range(len(labels)):
            evaluate_worker(labels[i], preds[i])
        # threads = [threading.Thread(target=evaluate_worker,
        #                             args=(self, labels[i], preds[i]),
        #                             )
        #            for i in range(len(labels))]
        # for thread in threads:
        #     thread.start()
        # for thread in threads:
        #     thread.join()

        print(
            f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
        )

    def get(self):
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self._calculate_precision_recall_f1()
        head = ['Dis-Thr']
        head.extend(self.dis_thr.tolist())
        table = PrettyTable()
        table.add_column('Threshold', self.dis_thr)
        table.add_column('TP', ['{:.0f}'.format(num) for num in self.TP])
        table.add_column('FP', ['{:.0f}'.format(num) for num in self.FP])
        table.add_column('FN', ['{:.0f}'.format(num) for num in self.FN])
        table.add_column('target_Precision',
                         ['{:.5f}'.format(num) for num in self.Precision])
        table.add_column('target_Recall',
                         ['{:.5f}'.format(num) for num in self.Recall])
        table.add_column('target_F1score',
                         ['{:.5f}'.format(num) for num in self.F1])
        print(table)

        return self.Precision, self.Recall, self.F1

    def reset(self):
        self.TP = np.zeros_like(self.dis_thr)
        self.FP = np.zeros_like(self.dis_thr)
        self.FN = np.zeros_like(self.dis_thr)
        self.Precision = np.zeros_like(self.dis_thr)
        self.Recall = np.zeros_like(self.dis_thr)
        self.F1 = np.zeros_like(self.dis_thr)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thr, self.TP, self.FP, self.FN, self.Precision,
            self.Recall, self.F1
        ],
                              axis=1)
        df = pd.DataFrame(all_metric)
        df.columns = ['dis_thr', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        return df

    def _calculate_infos(self, coord_label: List[RegionProperties],
                         coord_pred: List[RegionProperties],
                         img: np.ndarray) -> np.ndarray:
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
            return np.empty(
                (num_lbl,
                 num_pred))  # gt=0 or pred=0, (num_lbl, 0) or (0, num_pred)

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
        bbox_pre = torch.tensor([prop.bbox for prop in coord_pred],
                                dtype=torch.float32)  # M*4
        bbox_iou = bbox_overlaps(
            bbox_lbl, bbox_pre, mode='iou',
            is_aligned=False).numpy()  # num_lbl * num_pred

        # mask iou
        pixel_coords_lbl = [prop.coords for prop in coord_label]
        pixel_coords_pred = [prop.coords for prop in coord_pred]
        gt_mask = np.zeros((num_lbl, *img.shape))  # [num_lbl, H, W]
        pred_mask = np.zeros((num_pred, *img.shape))  # [num_pred, H, W]
        for i in range(num_lbl):
            gt_mask[i, pixel_coords_lbl[i][:, 0], pixel_coords_lbl[i][:,
                                                                      1]] = 1
        for i in range(num_pred):
            pred_mask[i, pixel_coords_pred[i][:, 0],
                      pixel_coords_pred[i][:, 1]] = 1
        mask_iou = target_mask_iou(gt_mask, pred_mask)  # num_lbl * num_pred

        if self.debug:
            print(f'centroid_lbl={centroid_lbl}')
            print(f'centroid_pred={centroid_pred}')
            print(f'bbox_iou={bbox_iou}')
            print(f'mask_iou={mask_iou}')
            print(f'eul_distance={eul_distance}')
            print('____' * 20)

        if self.iou_mode == 'mask_iou':
            reciprocal = np.reciprocal(mask_iou, where=mask_iou != 0)
            return eul_distance + reciprocal
        elif self.iou_mode == 'bbox_iou':
            reciprocal = np.reciprocal(bbox_iou, where=bbox_iou != 0)
            return eul_distance + reciprocal
        elif self.iou_mode == 'none':
            return eul_distance

        else:
            raise NotImplementedError(
                f"{self.iou_mode} is not implemented, please use 'none', 'bbox_iou', 'mask_iou' for distances."
            )

    def _calculate_tp_fn_fp(self, distances: np.ndarray,
                            threshold: int) -> Tuple[int]:
        """_summary_

        Args:
            distances (np.ndarray): distances in shape (num_lbl * num_pred)
            threshold (int): threshold of distances.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: TP, FN, FP
        """
        num_lbl, num_pred = distances.shape
        if num_lbl * num_pred == 0:
            # no lbl or no pred
            TP = 0
        else:
            if self.match_alg == 'hungarian':
                row_indexes, col_indexes = linear_sum_assignment(distances)
                selec_distance = distances[row_indexes, col_indexes]
                TP = np.sum(selec_distance < threshold)

            elif self.match_alg == 'forloop':
                for i in range(num_lbl):
                    for j in range(num_pred):
                        if distances[i, j] < threshold:
                            distances[:, j] = np.inf  # Set inf to mark matched
                            break
                TP = distances[distances == np.inf].size // num_lbl

            else:
                raise NotImplementedError(
                    f"{self.match_alg} is not implemented, please use 'hungarian' or 'forloop' for match_alg."
                )

        FP = num_pred - TP
        FN = num_lbl - TP

        return TP, FN, FP

    def _calculate_precision_recall_f1(self):
        self.Precision = self.TP / np.maximum(self.TP + self.FP,
                                              np.finfo(np.float64).eps)
        self.Recall = self.TP / (self.TP + self.FN)
        # micro F1 socre.
        self.F1 = 2 * self.Precision * self.Recall / np.maximum(
            self.Precision + self.Recall,
            np.finfo(np.float64).eps)


class BinaryCenterAveragePrecisionMetric(BinaryCenterMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: Union[int, List[float], np.ndarray] = 10,
                 dilate_kernel_size: List[int] = [7, 7],
                 match_alg: str = 'hungarian',
                 iou_mode: str = 'none',
                 **kwargs: Any):
        """
        NOTE:
            - For conf thresholds, we refer to torchmetrics using the `>=` for conf thresholds.
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/precision_recall_curve.py#L22

            - For the calculation of AP, we refer to torchmetrics.
                Step.1. Precision, Recall, we calculate the pr-curve from a multi threshold confusion matrix:
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/precision_recall_curve.py#L265

                Step.2. AP:
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/average_precision.py#L70

         Args:
            cong_thr (float, optional):
                - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
                0 to 1 as bins for the calculation.
                - If set to an `list` of floats, will use the indicated thresholds \
                    in the list as bins for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                bins for the calculation.

            Other parameters are the same as BinaryCenterMetric.
        """
        super().__init__(dis_thr=dis_thr,
                         conf_thr=0.5,
                         dilate_kernel_size=dilate_kernel_size,
                         match_alg=match_alg,
                         iou_mode=iou_mode,
                         **kwargs)

        self.conf_thr = _adjust_conf_thr_arg(conf_thr)
        self.reset()

    def update(self, labels: _TYPES, preds: _TYPES) -> None:
        """Support CHW, BCHW, HWC,BHWC, Image Path, or in their list form (except BHWC/BCHW),
            like [CHW, CHW, ...], [HWC, HWC, ...], [Image Path, Image Path, ...].

            Although support Image Path, but not recommend.
            Note : All preds are probabilities image from 0 to 1 in default.
            If images, Preds must be probability image from 0 to 1 in default.
            If path, Preds must be probabilities image from 0-1 in default, if 0-255,
            we are force /255 to 0-1 to probability image.
        Args:
            labels (_TYPES): Ground Truth images or image paths in list or single.
            preds (_TYPES): Preds images or image paths in list or single.
        """

        def evaluate_worker(label, pred):
            coord_label = _get_label_coord(label)

            for idx, conf_thr in enumerate(self.conf_thr):
                coord_pred = _get_pred_coord(pred.copy(), conf_thr,
                                             self.dilate_kernel_size)
                distances = self._calculate_infos(coord_label, coord_pred,
                                                  pred)
                for jdx, threshold in enumerate(self.dis_thr):
                    TP, FN, FP = self._calculate_tp_fn_fp(
                        distances.copy(), threshold)
                    with self.lock:
                        self.TP[jdx, idx] += TP
                        self.FP[jdx, idx] += FP
                        self.FN[jdx, idx] += FN
            return

        start_time = time.time()

        labels, preds = convert2iterable(labels, preds)

        for i in range(len(labels)):
            evaluate_worker(labels[i], preds[i])
        # threads = [threading.Thread(target=evaluate_worker,
        #                             args=(self, labels[i], preds[i]),
        #                             )
        #            for i in range(len(labels))]
        # for thread in threads:
        #     thread.start()
        # for thread in threads:
        #     thread.join()

        print(
            f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
        )

    def get(self) -> Tuple[np.ndarray]:
        """Compute metric

        Returns:
            _type_: .self.Precision, self.Recall, self.F1 in [dis_thr, conf_thr], self.AP in [1, dis_thr]
        """

        self._calculate_precision_recall_f1()
        # print('Average Update Spend {:0.2f}s.'.format(sum(self.time_cost)/len(self.time_cost)))

        precision = np.concatenate(
            [self.Precision, np.ones((len(self.Precision), 1))], axis=1)
        recall = np.concatenate(
            [self.Recall, np.zeros((len(self.Recall), 1))], axis=1)
        self.AP = -np.sum(
            (recall[:, 1:] - recall[:, :-1]) * precision[:, :-1], axis=1)

        table = PrettyTable()
        head = ['Dis—Thr']
        head.extend(self.dis_thr.tolist())
        table.field_names = head
        ap_row = ['AP']
        ap_row.extend(['{:.5f}'.format(num) for num in self.AP])
        table.add_row(ap_row)
        print(table)

        return self.Precision, self.Recall, self.F1, self.AP

    def reset(self):
        self.TP = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.FP = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.FN = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.Precision = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.Recall = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.F1 = np.zeros((len(self.dis_thr), len(self.conf_thr)))
        self.time_cost = []

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thr, self.AP])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'AP']
        return df
