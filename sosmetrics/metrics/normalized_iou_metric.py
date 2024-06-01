import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage import measure

from sosmetrics.metrics import BaseMetric, time_cost_deco
from sosmetrics.metrics.utils import (_TYPES, _adjust_dis_thr_arg,
                                      _safe_divide, calculate_target_infos,
                                      convert2format, convert2gray)


class NormalizedIoU(BaseMetric):

    def __init__(self, conf_thr: float = 0.5, **kwargs: Any):
        """ normalized Intersection over Union(nIoU).
        Original Code: https://github.com/YimianDai/open-acm

        Paper:
        @inproceedings{dai21acm,
        title   =  {Asymmetric Contextual Modulation for Infrared Small Target Detection},
        author  =  {Yimian Dai and Yiquan Wu and Fei Zhou and Kobus Barnard},
        booktitle =  {{IEEE} Winter Conference on Applications of Computer Vision, {WACV} 2021}
        year    =  {2021}
        }

        Args:
            conf_thr (float): _description_, Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.conf_thr = conf_thr
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            label = convert2gray(label).astype('int64')
            pred = convert2gray(pred > self.conf_thr).astype('int64')
            inter_arr, union_arr = self.batch_intersection_union(label, pred)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

        labels, preds = convert2format(labels, preds)
        if isinstance(labels, np.ndarray):
            evaluate_worker(self, labels, preds)

        elif isinstance(labels, (list, tuple)):
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
        else:
            raise NotImplementedError

    def batch_intersection_union(self, labels: np.ndarray,
                                 preds: np.ndarray) -> Tuple:
        labels_area = np.count_nonzero(labels == 1, axis=(-1, -2))
        preds_area = np.count_nonzero(preds == 1, axis=(-1, -2))
        intersection = np.count_nonzero(np.logical_and(labels == 1,
                                                       preds == 1),
                                        axis=(-1, -2))
        union = (labels_area + preds_area - intersection)
        return intersection, union

    @time_cost_deco
    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """

        IoU = _safe_divide(1.0 * self.total_inter, self.total_union)
        self.nIoU = IoU.mean()
        if self.print_table:
            table = PrettyTable()
            table.add_column(f'nIoU-{self.conf_thr}',
                             ['{:.4f}'.format(self.nIoU)])
            print(table)
        return self.nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])

    @property
    def table(self):
        df = pd.DataFrame(self.nIoU.reshape(1, 1))
        df.columns = ['nIoU']
        return df

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr})')


class TargetNormalizedIoU(NormalizedIoU):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thr: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 **kwargs: Any):
        self.dis_thr = _adjust_dis_thr_arg(dis_thr)
        self.match_alg = match_alg
        super().__init__(conf_thr=conf_thr, **kwargs)

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            pred = pred > self.conf_thr
            pred = pred.astype('uint8')
            gray_pred = convert2gray(pred).astype('int64')
            gray_label = convert2gray(label).astype('int64')

            image = measure.label(gray_pred, connectivity=2)
            coord_pred = measure.regionprops(image)

            image = measure.label(gray_label, connectivity=2)
            coord_label = measure.regionprops(image)
            distances, mask_iou, bbox_iou = calculate_target_infos(
                coord_label, coord_pred, gray_pred.shape[0],
                gray_pred.shape[1])

            if self.debug:
                print(f'bbox_iou={bbox_iou}')
                print(f'mask_iou={mask_iou}')
                print(f'eul_distance={distances}')
                print('____' * 20)

            for idx, threshold in enumerate(self.dis_thr):
                iou = self._get_matched_iou(distances.copy(), mask_iou.copy(),
                                            threshold)  # (num_lbl or num_pred)

                with self.lock:
                    self.total_iou[idx] = np.append(self.total_iou[idx], iou)
                    pass

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
        self.target_niou = np.array([iou.mean() for iou in self.total_iou])

        if self.print_table:
            table = PrettyTable()
            head = ['Dis—Thr']
            head.extend(self.dis_thr.tolist())
            table.field_names = head
            niou_row = [f'nIoU-{self.conf_thr}']
            niou_row.extend(['{:.4f}'.format(num) for num in self.target_niou])
            table.add_row(niou_row)
            print(table)
        return self.target_niou

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thr, self.target_niou])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'nIoU']
        return df

    def reset(self) -> None:
        self.total_iou = [np.array([]) for _ in range(len(self.dis_thr))]

    def _get_matched_iou(self, distances: np.ndarray, mask_iou: np.ndarray,
                         threshold: int) -> np.ndarray:

        num_lbl, num_pred = distances.shape

        iou = np.array([])
        if num_lbl * num_pred == 0:
            # no lbl or no pred
            return iou

        elif self.match_alg == 'hungarian':
            row_indexes, col_indexes = linear_sum_assignment(distances)
            selec_distance = distances[row_indexes, col_indexes]
            matched = selec_distance < threshold
            for i, j in zip(row_indexes[matched], col_indexes[matched]
                            ):  # col_indexes present matched pred index.
                iou = np.append(iou, mask_iou[i, j])

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:,
                                  j] = np.inf  # Set inf to mark matched preds.
                        iou = np.append(iou, mask_iou[i, j])
                        break
        return iou
