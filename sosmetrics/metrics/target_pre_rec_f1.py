import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment

from .base import BaseMetric, time_cost_deco
from .utils import (_TYPES, _adjust_dis_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray,
                    second_match_method)


class TargetPrecisionRecallF1(BaseMetric):

    def __init__(self,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 conf_thr: float = 0.5,
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 dilate_kernel: Union[List[int], int] = [0, 0],
                 **kwargs: Any):
        """
        TP: True Positive, GT is Positive and Pred is Positive, If Euclidean Distance < threshold, matched.
        FN: False Negative, GT is Positive and Pred is Negative.
        FP: False Positive, GT is Negative and Pred is Positive. If Euclidean Distance > threshold, not matched.
        Recall: TP/(TP+FN).
        Precision: TP/(TP+FP).
        F1: 2*Precision*Recall/(Precision+Recall).
        .get will return Precision, Recall, F1 in array.

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
            dis_thrs (Union[List[float], int], optional): dis_thrs of Euclidean distance,
                - If set to an `int` , will use this value to distance threshold.
                - If set to an `list` of float or int, will use the indicated thresholds \
                    in the list as conf_thr for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                conf_thr for the calculation.
                if List, closed interval.
                Defaults to [1, 10].
            cong_thr (float, optional): confidence threshold. Defaults to 0.5.
            match_alg (str, optional):Match algorithm, support 'hungarian' and 'forloop' to match pred and gt.\
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle. Defaults to 'forloop'.
            second_match (str, optional): Second match algorithm, support 'none', 'mask', 'bbox', \
                'plus_mask' and 'plus_bbox', 'none' means no secondary matching. Defaults to 'none'.
            dilate_kernel (Union[List[int], int], optional): Dilated kernel size, support Rect and Circle, \
                [0, 0] or 0 means no dilate; \
                list of int means Rect dilated kernel, like [3, 3] or [3,4]; \
                int means diameter of Circle dilated kernel. Defaults to [0, 0].
        """

        super().__init__(**kwargs)
        self.dis_thrs = _adjust_dis_thr_arg(dis_thrs)
        self.conf_thr = np.array([conf_thr])
        self.match_alg = match_alg
        self.second_match = second_match
        self.dilate_kernel = dilate_kernel
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
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

        def evaluate_worker(self, label, pred):
            # to unit8 for ``convert2gray()``
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
                TP, FN, FP = self._calculate_tp_fn_fp(distances.copy(),
                                                      threshold)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

        labels, preds = convert2format(labels, preds)

        for i in range(len(labels)):
            evaluate_worker(self, labels[i], preds[i])
        # threads = [
        #     threading.Thread(
        #         target=evaluate_worker,
        #         args=(self, labels[i], preds[i]),
        #     ) for i in range(len(labels))
        # ]
        # for thread in threads:
        #     thread.start()
        # for thread in threads:
        #     thread.join()

    @time_cost_deco
    def get(self):
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self._calculate_precision_recall_f1()
        head = ['Dis-Thr']
        head.extend(self.dis_thrs.tolist())
        table = PrettyTable()
        table.add_column('Dis-Thr', self.dis_thrs)
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

        self.mean_Precision = self.Precision.mean()
        self.mean_Recall = self.Recall.mean()
        self.mean_F1 = self.F1.mean()
        return self.Precision, self.Recall, self.F1

    def reset(self):
        self.TP = np.zeros_like(self.dis_thrs)
        self.FP = np.zeros_like(self.dis_thrs)
        self.FN = np.zeros_like(self.dis_thrs)
        self.Precision = np.zeros_like(self.dis_thrs)
        self.Recall = np.zeros_like(self.dis_thrs)
        self.F1 = np.zeros_like(self.dis_thrs)
        self.mean_Precision = np.zeros_like(self.dis_thrs)
        self.mean_Recall = np.zeros_like(self.dis_thrs)
        self.mean_F1 = np.zeros_like(self.dis_thrs)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thrs, self.TP, self.FP, self.FN, self.Precision,
            self.Recall, self.F1,
            self.mean_Precision.reshape(1, ).repeat(len(self.dis_thrs)),
            self.mean_Recall.reshape(1, ).repeat(len(self.dis_thrs)),
            self.mean_F1.reshape(1, ).repeat(len(self.dis_thrs))
        ],
                              axis=1)
        df = pd.DataFrame(all_metric)
        df.columns = [
            'dis_thrs',
            'TP',
            'FP',
            'FN',
            'Precision',
            'Recall',
            'F1',
            'mean_Precision',
            'mean_Recall',
            'mean_F1',
        ]
        return df

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

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:, j] = np.nan  # Set inf to mark matched
                        break
            TP = np.sum(np.isnan(distances)) // num_lbl
        elif self.match_alg == 'hungarian':
            # fix feasible in hungarian
            distances[np.isinf(distances)] = 1e10
            row_indexes, col_indexes = linear_sum_assignment(distances)
            selec_distance = distances[row_indexes, col_indexes]
            matched = selec_distance < threshold
            TP = np.sum(matched)

        else:
            raise ValueError(
                f"{self.match_alg} is not implemented, please use 'forloop' for match_alg."
            )

        FP = num_pred - TP
        FN = num_lbl - TP

        return TP, FN, FP

    def _calculate_precision_recall_f1(self):

        self.Precision = _safe_divide(self.TP, self.TP + self.FP)
        self.Recall = _safe_divide(self.TP, self.TP + self.FN)
        # micro F1 socre.
        self.F1 = _safe_divide(2 * self.Precision * self.Recall,
                               self.Precision + self.Recall)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}, '
                f'match_alg={self.match_alg}, '
                f'second_match={self.second_match}, '
                f'dilate_kernel={self.dilate_kernel})')
