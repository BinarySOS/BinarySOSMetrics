import threading
from typing import Any, Tuple

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from sosmetrics.metrics import BaseMetric, time_cost_deco
from sosmetrics.metrics.utils import (_TYPES, _safe_divide, convert2format,
                                      convert2gray)


class PixelNormalizedIoU(BaseMetric):

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

        math: $\begin{aligned}\text{nIoU}&=\frac{1}{N}\\sum_i^N\frac{\text{TP}[i]} \
                {\text{T}[i]+\text{P}[i]-\text{TP}[i]}\\end{aligned}$
                N is number of images, TP is the number of true positive pixels, \
                    T is the number of ground truth positive pixels, \
                    P is the number of predicted positive pixels.
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
        self.nIoU = np.zeros((1, 1))

    @property
    def table(self):
        df = pd.DataFrame(self.nIoU.reshape(1, 1))
        df.columns = ['nIoU']
        return df

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr})')
