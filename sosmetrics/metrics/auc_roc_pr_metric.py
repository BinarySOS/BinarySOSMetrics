import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn.metrics import auc
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryROC

from .base import BaseMetric
from .utils import _TYPES, convert2iterable


# codespell:ignore fpr
class AUC_ROC_PRMetric(BaseMetric):

    def __init__(self, bins: int = 30, **kwargs: Any):
        """length of tpr, fpr are bins+1.

        .get() will return auc_roc, auc_pr, fpr, tpr, precision,
            recall in array.
        Args:
            bins (int, optional): score thresholds. Defaults to 30.
        """
        super().__init__(**kwargs)
        self.bins = bins
        self.lock = threading.Lock()
        self.roc_curve_fn = BinaryROC(thresholds=self.bins)
        self.pr_curve_fn = BinaryPrecisionRecallCurve(thresholds=self.bins)
        self.reset()

    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array):
            ten_pred = torch.from_numpy(pred)
            ten_gt = torch.from_numpy(label).to(torch.int64)
            self.roc_curve_fn.update(ten_pred, ten_gt)
            self.pr_curve_fn.update(ten_pred, ten_gt)

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

        if self.debug:
            print(
                f'{self.__class__.__name__} spend time for update:{time.time() - start_time}'
            )

    def get(self):

        self.fpr, self.tpr, _ = self.roc_curve_fn.compute()
        self.precision, self.recall, _ = self.pr_curve_fn.compute()
        self.auc_roc = auc(self.fpr, self.tpr)
        self.auc_pr = auc(self.recall, self.precision)
        if self.print_table:
            head = ['AUC_ROC', 'AUC_PR']
            table = PrettyTable(head)
            table.add_row([self.auc_roc.numpy(), self.auc_pr.numpy()])
            print(table)

        return self.auc_roc, self.auc_pr, self.fpr.numpy(), self.tpr.numpy(
        ), self.precision.numpy(), self.recall.numpy()

    def reset(self):
        self.roc_curve_fn.reset()
        self.pr_curve_fn.reset()

    @property
    def table(self):
        all_metric = np.stack([self.auc_roc.numpy(),
                               self.auc_pr.numpy()])[:, np.newaxis].T
        df = pd.DataFrame(all_metric)
        df.columns = ['AUC_ROC', 'AUC_PR']
        return df
