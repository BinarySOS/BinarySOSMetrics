import threading
import time
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from skimage import measure
from skimage.measure._regionprops import RegionProperties

from .base import BaseMetric
from .utils import _TYPES, convert2gray, convert2iterable


class PD_FAMetric(BaseMetric):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thr: Union[List[int], int] = [1, 10],
                 **kwargs: Any):
        """Modified from https://github.com/XinyiYing/BasicIRSTD/blob/main/metrics.py
        We added multi-threading as well as batch processing.

        1. get connectivity region of gt and pred
        2. ergodic connectivity region of gt, compute the distance between every connectivity region of pred,
            if distance < threshold, then match.
            and delete the connectivity region of pred,
            and set to 1 for compute number of pixel.

        TD: Number of correctly predicted targets, GT is positive and Pred is positive, like TP.
        AT: All Targets, Number of target in GT, like TP + FN.
        PD: Probability of Detection, PD =TD/AT, like Recall = TP/(TP+FN).
        NP: All image Pixels, NP = H*W*num_gt_img.
        FD: The numbers of falsely predicted pixels, dismatch pixel, FD = NP - pixel_of_each_TD = FP * pixel_of_each_FP.
        FA: False-Alarm Rate, FA = FD/NP.

        Args:
            conf_thr (float, Optional): Confidence threshold. Defaults to 0.5.
            dis_thr (Union[List[int], int], optional): dis_thr of Euclidean distance,
                if List, closed interval. . Defaults to [1,10].
        """
        super().__init__(**kwargs)
        if isinstance(dis_thr, int):
            self.dis_thr = np.array([dis_thr])
        else:
            self.dis_thr = np.arange(dis_thr[0], dis_thr[1] + 1)
        self.conf_thr = conf_thr

        self.lock = threading.Lock()
        self.reset()

    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
            # to unit8 for ``convert2gray()``
            label = label > 0  # sometimes is 0-1, force to 255.
            label = label.astype('uint8')
            pred = pred > self.conf_thr
            pred = pred.astype('uint8')

            # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
            # 'int64' is for measure.label().
            pred = convert2gray(pred).astype('int64')
            label = convert2gray(label).astype('int64')

            for idx, threshold in enumerate(self.dis_thr):
                image = measure.label(pred, connectivity=2)
                coord_pred = measure.regionprops(image)

                label = measure.label(label, connectivity=2)
                coord_label = measure.regionprops(label)

                AT, TD, FD, NP = self._calculate_tp_fn_fp(
                    coord_label, coord_pred, threshold, pred)
                with self.lock:
                    self.AT[idx] += AT
                    self.FD[idx] += FD
                    self.NP[idx] += NP
                    self.TD[idx] += TD

        if self.debug:
            start_time = time.time()

        # Packaged in the format we need, bhwc of np.array or hwc of list.
        labels, preds = convert2iterable(labels, preds)

        # for i in range(len(labels)):
        #     evaluate_worker(labels[i].squeeze(0), preds[i].squeeze(0))
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

        if self.debug:
            print(
                f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
            )

    def get(self):
        self.FA = self.FD / self.NP
        self.PD = self.TD / self.AT

        if self.print_table:
            head = ['Threshold']
            head.extend(self.dis_thr.tolist())
            table = PrettyTable()
            table.add_column('Threshold', self.dis_thr)
            table.add_column('TD', ['{:.0f}'.format(num) for num in self.TD])
            table.add_column('AT', ['{:.0f}'.format(num) for num in self.AT])
            table.add_column('FD', ['{:.0f}'.format(num) for num in self.FD])
            table.add_column('NP', ['{:.0f}'.format(num) for num in self.NP])
            table.add_column('target_Pd',
                             ['{:.5f}'.format(num) for num in self.PD])
            table.add_column('pixel_Fa',
                             ['{:.5e}'.format(num) for num in self.FA])
            print(table)

        return self.PD, self.FA

    def reset(self) -> None:
        self.FA = np.zeros_like(self.dis_thr)
        self.TD = np.zeros_like(self.dis_thr)
        self.FD = np.zeros_like(self.dis_thr)
        self.NP = np.zeros_like(self.dis_thr)
        self.AT = np.zeros_like(self.dis_thr)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thr, self.TD, self.AT, self.FD, self.NP, self.PD, self.FA
        ],
                              axis=1)
        df_pd_fa = pd.DataFrame(all_metric)
        df_pd_fa.columns = [
            'dis_thr', 'TD', 'AT', 'FD', 'NP', 'target_Pd', 'pixel_Fa'
        ]
        return df_pd_fa

    def _calculate_tp_fn_fp(self, coord_label: List[RegionProperties],
                            coord_pred: List[RegionProperties], threshold: int,
                            pred_img: np.array) -> Tuple[int, int, int, int]:
        """_summary_

        Args:
            coord_label (List[RegionProperties]): measure.regionprops(label)
            coord_pred (List[RegionProperties]): measure.regionprops(pred)
            threshold (int): _description_
            pred_img (np.array): _description_

        Returns:
            tuple[int, int, int, int]: AT, TD, FD, NP
        """
        centroid_lbl = np.array([prop.centroid for prop in coord_label
                                 ]).astype(np.float32)  # N*2
        centroid_pre = np.array([prop.centroid for prop in coord_pred
                                 ]).astype(np.float32)  # M*2

        num_lbl = centroid_lbl.shape[0]  # number of label
        num_pred = centroid_pre.shape[0]  # number of pred
        true_img = np.zeros(pred_img.shape)
        if num_lbl * num_pred == 0:
            # no lbl or no pred
            TD = 0
        else:
            eul_distance = np.linalg.norm(centroid_lbl[:, None, :] -
                                          centroid_pre[None, :, :],
                                          axis=-1)  # num_lbl * num_pred
            for i in range(num_lbl):
                for j in range(num_pred):
                    if eul_distance[i, j] < threshold:
                        eul_distance[:, j] = np.inf  # Set inf to mark matched
                        true_img[coord_pred[j].coords[:, 0],
                                 coord_pred[j].coords[:, 1]] = 1
                        break
            TD = eul_distance[eul_distance == np.inf].size // num_lbl

        FD = (pred_img - true_img).sum()
        NP = pred_img.shape[0] * pred_img.shape[1]
        AT = num_lbl
        return AT, TD, FD, NP
