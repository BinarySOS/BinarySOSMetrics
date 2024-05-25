import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix

from .base import BaseMetric
from .utils import _TYPES, convert2iterable


class Precision_Recall_F1_IoUMetric(BaseMetric):

    def __init__(self, conf_thr: float, debug: bool = False, **kwargs: Any):
        """length of true positive rates(tpr), false positive(fpr) are bins+1.

        Args:
            bins (int, optional): score thresholds. Defaults to 30.
        """
        super().__init__(**kwargs)
        self.conf_thr = conf_thr
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            tn, fp, fn, tp = self._confusion_mat(label, pred, self.conf_thr)
            with self.lock:
                self.tp[0] += tp
                self.fp[0] += fp
                self.fn[0] += fn
                self.tn[0] += tn

        if self.debug:
            start_time = time.time()

        labels, preds = convert2iterable(labels, preds)
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

        print(
            f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
        )

    def get(self):
        self.recall = self.tp / (self.tp + self.fn + 1e-10)
        self.precision = self.tp / (self.tp + self.fp + 1e-10)
        self.iou = self.tp / (self.tp + self.fp + self.fn + 1e-10)
        self.f1_score = 2 * self.precision * self.recall / (
            self.precision + self.recall + 1e-10)
        if self.print_table:
            head = [
                f'Precision-{self.conf_thr}', f'Recall-{self.conf_thr}',
                f'F1-{self.conf_thr}', f'IOU-{self.conf_thr}'
            ]
            table = PrettyTable(head)
            table.add_row([
                '{:.5f}'.format(self.precision[0]),
                '{:.5f}'.format(self.recall[0]),
                '{:.5f}'.format(self.f1_score[0]), '{:.5f}'.format(self.iou[0])
            ])
            print(table)

        return self.precision, self.recall, self.f1_score, self.iou

    def reset(self):
        self.tp = np.zeros((1))
        self.fp = np.zeros((1))
        self.fn = np.zeros((1))
        self.tn = np.zeros((1))

    def _confusion_mat(self, label: np.array, pred: np.array,
                       score_thresh: float):
        predict = (pred > score_thresh).astype('float32').flatten()
        target = label.astype('int64').flatten()  # T
        tn, fp, fn, tp = confusion_matrix(target, predict).ravel()
        return tn, fp, fn, tp

    @property
    def table(self):
        all_metric = np.stack(
            [self.precision, self.recall, self.f1_score,
             self.iou])[:, np.newaxis].T
        df = pd.DataFrame(all_metric)
        df.columns = [
            f'Precision-{self.conf_thr}', f'Recall-{self.conf_thr}',
            f'F1-{self.conf_thr}', f'IOU-{self.conf_thr}'
        ]
        return df
