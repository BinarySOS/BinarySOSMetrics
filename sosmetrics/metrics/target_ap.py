import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from .base import time_cost_deco
from .target_pre_rec_f1 import TargetPrecisionRecallF1
from .utils import (_TYPES, _adjust_conf_thr_arg, calculate_target_infos,
                    convert2format, get_label_coord_and_gray,
                    get_pred_coord_and_gray, second_match_method)


class TargetAveragePrecision(TargetPrecisionRecallF1):

    def __init__(self,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 conf_thrs: Union[int, List[float], np.ndarray] = 10,
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 **kwargs: Any):
        """
        Compute AP for each dis_thrs, and Precision, Recall, F1 for each dis_thrs and conf_thrs.
        NOTE:
            - For conf thresholds, we refer to torchmetrics using the `>=` for conf thresholds.
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/precision_recall_curve.py#L22

            - For the calculation of AP, we refer to torchmetrics.
                Step.1. Precision, Recall, we calculate the pr-curve from a multi threshold confusion matrix:
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/precision_recall_curve.py#L265

                Step.2. AP:
                https://github.com/Lightning-AI/torchmetrics/blob/3f112395b1ca0141ad2d8622628110fa363f9953/src/torchmetrics/functional/classification/average_precision.py#L70

        About Second Match:
            Supports 4 secondary matching schemes.
            1. mask: mask_iou is used to mark non-overlapping data pairs.
            2. bbox: bbox_iou is used to mark non-overlapping data pairs.
            3. mask_plus: will return the sum of eul_dis and (1-mask_iou).
            4. bbox_plus: will return the sum of eul_dis and (1-bbox_iou).

                The reason for adding (1-iou) is to maintain the same trend as distance, \
                    i.e., smaller means closer to gt.

         Args:
            cong_thr (float, optional):
                - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
                0 to 1 as conf_thrs for the calculation.
                - If set to an `list` of floats, will use the indicated thresholds \
                    in the list as conf_thrs for the calculation
                - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                conf_thrs for the calculation.
            second_match (str, optional): Second match algorithm, support 'none', 'mask', 'bbox', \
                'mask_plus' and 'bbox_plus', 'none' means no secondary matching. Defaults to 'none'.
            Other parameters are the same as TargetPrecisionRecallF1.
        """

        self.conf_thrs = _adjust_conf_thr_arg(conf_thrs)
        super().__init__(dis_thrs=dis_thrs,
                         conf_thr=0.5,
                         match_alg=match_alg,
                         second_match=second_match,
                         **kwargs)
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

            for idx, conf_thr in enumerate(self.conf_thrs):
                coord_pred, gray_pred = get_pred_coord_and_gray(
                    pred.copy(), conf_thr)

                distances, mask_iou, bbox_iou = calculate_target_infos(
                    coord_label, coord_pred, gray_pred.shape[0],
                    gray_pred.shape[1])

                if self.debug:
                    print(f'bbox_iou={bbox_iou}')
                    print(f'mask_iou={mask_iou}')
                    print(f'eul_distance={distances}')
                    print('____' * 20)

                if self.second_match != 'none':
                    distances = second_match_method(distances, mask_iou,
                                                    bbox_iou,
                                                    self.second_match)
                if self.debug:
                    print(f'After second matche eul distance={distances}')
                    print('____' * 20)

                for jdx, threshold in enumerate(self.dis_thrs):
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
            _type_: .self.Precision, self.Recall, self.F1 in [dis_thrs, conf_thrs], self.AP in [1, dis_thrs]
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
            head.extend(self.dis_thrs.tolist())
            table.field_names = head
            ap_row = ['AP']
            ap_row.extend(['{:.4f}'.format(num) for num in self.AP])
            table.add_row(ap_row)
            print(table)

        return self.Precision, self.Recall, self.F1, self.AP

    def reset(self):
        self.TP = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.FP = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.FN = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.Precision = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.Recall = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.F1 = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.AP = np.zeros(len(self.dis_thrs))

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thrs, self.AP])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'AP']
        return df

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(match_alg={self.match_alg}, '
                f'second_match={self.second_match})')
