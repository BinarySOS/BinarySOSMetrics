import threading
from typing import Any, List, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment

from sosmetrics.metrics import time_cost_deco
from sosmetrics.metrics.utils import (_TYPES, _adjust_dis_thr_arg,
                                      _safe_divide, calculate_target_infos,
                                      convert2format, get_label_coord_and_gray,
                                      get_pred_coord_and_gray,
                                      second_match_method)

from .pixel_normalized_iou import PixelNormalizedIoU


class HybridNormalizedIoU(PixelNormalizedIoU):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 dilate_kernel: Union[List[int], int] = [0, 0],
                 **kwargs: Any):
        """We did the optimisation.
            The task in the original code is to have only one target per image.
            In this implementation, we can have multiple targets per image.
            N is the number of ground truth targets, TP is the number of true positive targets, \
                T is the number of ground truth targets, \
                P is the number of predicted targets.

        About Second Match:
            Supports 4 secondary matching schemes.
            1. mask: mask_iou is used to mark non-overlapping data pairs.
            2. bbox: bbox_iou is used to mark non-overlapping data pairs.
            3. plus_mask: Enhanced version of 'mask', will return a result that adding new conditions to 'mask' mode, \
                the sum of eul_dis and (1-mask_iou).
            4. plus_bbox: Enhanced version of 'bbox', will return a result that adding new conditions to 'bbox' mode, \
                the sum of eul_dis and (1-bbox_iou).

            The reason for adding (1-iou) is to maintain the same trend as distance, \
                i.e., smaller means closer to gt.


        Args:
            conf_thr (float, optional): _description_. Defaults to 0.5.
            dis_thrs (Union[List[int], int], optional): _description_. Defaults to [1, 10].
            match_alg (str, optional):Match algorithm, support 'hungarian' and 'forloop' to match pred and gt.\
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle. Defaults to 'forloop'.
            second_match (str, optional): Second match algorithm, support 'none', 'mask', 'bbox', \
                'plus_mask' and 'plus_bbox', 'none' means no secondary matching. Defaults to 'none'.
            dilate_kernel (Union[List[int], int], optional): Dilated kernel size, support Rect and Circle, \
                [0, 0] or 0 means no dilate; \
                list of int means Rect dilated kernel, like [3, 3] or [3,4]; \
                int means diameter of Circle dilated kernel. Defaults to [0, 0].

            . Defaults to [0, 0].
        """
        self.dis_thrs = _adjust_dis_thr_arg(dis_thrs)
        self.match_alg = match_alg
        self.second_match = second_match
        self.dilate_kernel = dilate_kernel
        super().__init__(conf_thr=conf_thr, **kwargs)

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            coord_label, gray_label = get_label_coord_and_gray(label)
            coord_pred, gray_pred = get_pred_coord_and_gray(
                pred.copy(), self.conf_thr, self.dilate_kernel)
            distances, mask_iou, bbox_iou = calculate_target_infos(
                coord_label, coord_pred, gray_pred.shape[0],
                gray_pred.shape[1])

            if self.debug:
                print(f'bbox_iou={bbox_iou}')
                print(f'mask_iou={mask_iou}')
                print(f'eul_distance={distances}')
                print('____' * 20)

            if self.second_match != 'none':
                distances = second_match_method(distances, mask_iou, bbox_iou,
                                                self.second_match)
                if self.debug:
                    print(f'After second matche eul distance={distances }')
                    print('____' * 20)

            for idx, threshold in enumerate(self.dis_thrs):
                iou = self._get_matched_iou(distances.copy(), mask_iou.copy(),
                                            threshold)  # (num_lbl or num_pred)

                with self.lock:
                    self.total_iou[idx] = np.append(self.total_iou[idx], iou)
                    pass
            with self.lock:
                self.total_gt += distances.shape[0]

        labels, preds = convert2format(labels, preds)

        # for i in range(len(labels)):
        #     evaluate_worker(labels[i], preds[i])
        threads = [
            threading.Thread(
                target=evaluate_worker,
                args=(self, labels[i], preds[i]),
            ) for i in range(len(labels))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    @time_cost_deco
    def get(self):
        target_niou = np.array([iou.sum() for iou in self.total_iou])
        self.target_niou = _safe_divide(target_niou, self.total_gt)

        if self.print_table:
            table = PrettyTable()
            head = ['Dis—Thr']
            head.extend(self.dis_thrs.tolist())
            table.field_names = head
            niou_row = [f'nIoU-{self.conf_thr}']
            niou_row.extend(['{:.4f}'.format(num) for num in self.target_niou])
            table.add_row(niou_row)
            print(table)
        return self.target_niou

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thrs, self.target_niou])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'nIoU']
        return df

    def reset(self) -> None:
        self.total_iou = [np.array([]) for _ in range(len(self.dis_thrs))]
        self.total_gt = np.zeros(1)
        self.target_niou = np.zeros(len(self.dis_thrs))

    def _get_matched_iou(self, distances: np.ndarray, mask_iou: np.ndarray,
                         threshold: int) -> np.ndarray:

        num_lbl, num_pred = distances.shape

        iou = np.array([])
        if num_lbl * num_pred == 0:
            # no lbl or no pred
            return iou

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:,
                                  j] = np.nan  # Set inf to mark matched preds.
                        iou = np.append(iou, mask_iou[i, j])
                        break

        elif self.match_alg == 'hungarian':
            row_indexes, col_indexes = linear_sum_assignment(distances)
            selec_distance = distances[row_indexes, col_indexes]
            matched = selec_distance < threshold
            for i, j in zip(row_indexes[matched], col_indexes[matched]
                            ):  # col_indexes present matched pred index.
                iou = np.append(iou, mask_iou[i, j])
        else:
            raise ValueError(f'Unknown match_alg: {self.match_alg}')
        return iou

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}, '
                f'match_alg={self.match_alg}, '
                f'second_match={self.second_match})')
