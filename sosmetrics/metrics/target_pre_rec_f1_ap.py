import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment

from .base import BaseMetric, time_cost_deco
from .utils import (_TYPES, _adjust_conf_thr_arg, _adjust_dis_thr_arg,
                    _safe_divide, calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray)


class TargetPrecisionRecallF1(BaseMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: float = 0.5,
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
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
            match_alg (str, optional): 'hungarian' or 'forloop' to match pred and gt,
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle.. Defaults to 'forloop'.
            second_match (str, optional): 'none' or 'mask_iou' to match pred and gt after distance matching. \
                Defaults to 'none'.
        """

        super().__init__(**kwargs)
        self.dis_thr = _adjust_dis_thr_arg(dis_thr)
        self.conf_thr = np.array([conf_thr])
        self.match_alg = match_alg
        self.second_match = second_match
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
                pred.copy(), self.conf_thr)
            distances, mask_iou, bbox_iou = calculate_target_infos(
                coord_label, coord_pred, gray_pred.shape[0],
                gray_pred.shape[1])
            if self.debug:
                print(f'bbox_iou={bbox_iou}')
                print(f'mask_iou={mask_iou}')
                print(f'eul_distance={distances}')
                print('____' * 20)

            if self.second_match == 'mask_iou':
                mask_iou[mask_iou == 0.] = np.inf
                mask_iou[mask_iou != np.inf] = 0.
                distances = distances + mask_iou

                if self.debug:
                    print(f'after second match mask iou={mask_iou}')
                    print(
                        f'after second matche eul distance={distances + mask_iou}'
                    )
                    print('____' * 20)

            for idx, threshold in enumerate(self.dis_thr):
                TP, FN, FP = self._calculate_tp_fn_fp(distances.copy(),
                                                      threshold)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

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
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self._calculate_precision_recall_f1()
        head = ['Dis-Thr']
        head.extend(self.dis_thr.tolist())
        table = PrettyTable()
        table.add_column('Dis-Thr', self.dis_thr)
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

        self.Precision = _safe_divide(self.TP, self.TP + self.FP)
        self.Recall = _safe_divide(self.TP, self.TP + self.FN)
        # micro F1 socre.
        self.F1 = _safe_divide(2 * self.Precision * self.Recall,
                               self.Precision + self.Recall)

    def __repr__(self) -> str:
        message = (f'{self.__class__.__name__}'
                   f'(match_alg={self.match_alg} '
                   f'conf_thr={self.conf_thr})')
        return message


class TargetAveragePrecision(TargetPrecisionRecallF1):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: Union[int, List[float], np.ndarray] = 10,
                 match_alg: str = 'forloop',
                 **kwargs: Any):
        """
        Compute AP for each dis_thr, and Precision, Recall, F1 for each dis_thr and conf_thr.
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
                         match_alg=match_alg,
                         **kwargs)

        self.conf_thr = _adjust_conf_thr_arg(conf_thr)
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
            coord_label, gray_label = get_label_coord_and_gray(label)

            for idx, conf_thr in enumerate(self.conf_thr):
                coord_pred, gray_pred = get_pred_coord_and_gray(
                    pred.copy(), conf_thr)

                distances, _, _ = calculate_target_infos(
                    coord_label, coord_pred, gray_pred.shape[0],
                    gray_pred.shape[1])
                for jdx, threshold in enumerate(self.dis_thr):
                    TP, FN, FP = self._calculate_tp_fn_fp(
                        distances.copy(), threshold)
                    with self.lock:
                        self.TP[jdx, idx] += TP
                        self.FP[jdx, idx] += FP
                        self.FN[jdx, idx] += FN
            return

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
    def get(self) -> Tuple[np.ndarray]:
        """Compute metric

        Returns:
            _type_: .self.Precision, self.Recall, self.F1 in [dis_thr, conf_thr], self.AP in [1, dis_thr]
        """

        self._calculate_precision_recall_f1()

        precision = np.concatenate(
            [self.Precision, np.ones((len(self.Precision), 1))], axis=1)
        recall = np.concatenate(
            [self.Recall, np.zeros((len(self.Recall), 1))], axis=1)
        self.AP = -np.sum(
            (recall[:, 1:] - recall[:, :-1]) * precision[:, :-1], axis=1)
        if self.print_table:
            table = PrettyTable()
            head = ['Dis—Thr']
            head.extend(self.dis_thr.tolist())
            table.field_names = head
            ap_row = ['AP']
            ap_row.extend(['{:.4f}'.format(num) for num in self.AP])
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
        self.AP = np.zeros(len(self.dis_thr))

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thr, self.AP])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'AP']
        return df

    def __repr__(self) -> str:
        message = (f'{self.__class__.__name__}'
                   f'(match_alg={self.match_alg})')
        return message
