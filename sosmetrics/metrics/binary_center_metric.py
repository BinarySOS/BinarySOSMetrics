import threading
import time
from typing import Any, List, Union

import cv2
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage import measure

from .base import BaseMetric
from .utils import _TYPES, convert2gray, convert2iterable


class BinaryCenterMetric(BaseMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: float = 0.5,
                 dilate_kernel_size: List[int] = [7, 7],
                 match_alg: str = 'hungarian',
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
            dis_thr (Union[List[int], int], optional): dis_thr of Euclidean distance,
                if List, closed interval. Defaults to [1, 10].
            cong_thr (float, optional): confidence threshold. Defaults to 0.5.
            dilate_kernel_size (List[int], optional): kernel size of cv2.dilated.
                The difference is that when [0, 0], the dilation algorithm will not be used. Defaults to [7, 7].
            match_alg (str, optional): 'hungarian' or 'ergodic' to match pred and gt,
                'ergodic'is the original implementation of PD_FA,
                based on the first-match principle.Usually, hungarian is fast and accurate. Defaults to 'hungarian'.
        """

        super().__init__(**kwargs)
        if isinstance(dis_thr, int):
            self.dis_thr = np.array([dis_thr])
        else:
            self.dis_thr = np.arange(dis_thr[0], dis_thr[1] + 1)
        self.conf_thr = conf_thr
        self.match_alg = match_alg
        self.dilate_kernel_size = dilate_kernel_size
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
            # to unit8 for ``convert2gray()``
            label = label > 0  # sometimes is 0-1, force to 255.
            label = label.astype('uint8')
            pred = pred > self.conf_thr
            pred = pred.astype('uint8')

            # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
            # 'int64' is for measure.label().

            label = convert2gray(label).astype('int64')
            pred = convert2gray(pred)
            dilated_pred = self._get_dilated(pred).astype('int64')

            for idx, threshold in enumerate(self.dis_thr):
                pred_img = measure.label(dilated_pred, connectivity=2)
                coord_pred = measure.regionprops(pred_img)
                label = measure.label(label, connectivity=2)
                coord_label = measure.regionprops(label)

                TP, FN, FP = self._calculate_tp_fn_fp(coord_label, coord_pred,
                                                      threshold)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

        if self.debug:
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

        if self.debug:
            print(
                f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
            )

    def get(self):
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self.Precision = self.TP / (self.TP + self.FP)
        self.Recall = self.TP / (self.TP + self.FN)
        # micro F1 socre.
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision +
                                                      self.Recall)

        head = ['Threshold']
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

    def _get_dilated(self, image: np.array):
        """_summary_

        Args:
            image (np.array): hwc of np.array.
        """

        kernel = np.ones(self.dilate_kernel_size, np.uint8)
        if kernel.sum() == 0:
            return image

        dilated_image = cv2.dilate(image, kernel, iterations=1)

        return dilated_image

    def _calculate_tp_fn_fp(self, coord_label: np.array, coord_pred: np.array,
                            threshold: int):
        """_summary_

        Args:
            gt_centroids (np.array): shape = (N,2)
            pred_centroids (np.array): shape = (M,2)
            threshold (int): threshold, Euclidean Distance.

        Returns:
            _type_: _description_
        """
        TP = 0
        FN = 0
        FP = 0

        if self.match_alg == 'hungarian':
            centroid_lbl = np.array([prop.centroid for prop in coord_label])
            centroid_pre = np.array([prop.centroid for prop in coord_pred])
            if centroid_lbl.shape[0] == 0:
                centroid_lbl = np.empty((0, 2))
            if centroid_pre.shape[0] == 0:
                centroid_pre = np.empty((0, 2))
            eul_distance = np.linalg.norm(
                centroid_lbl[:, None, :] - centroid_pre[None, :, :],
                axis=-1)  # N*M, N==coord_label.shape, M = coord_pred.shape
            row_indexes, col_indexes = linear_sum_assignment(eul_distance)
            selec_distance = eul_distance[row_indexes, col_indexes]
            TP = np.sum(selec_distance < threshold)
            FP = len(coord_pred) - TP
            FN = len(coord_label) - TP

        elif self.match_alg == 'ergodic':
            for i in range(len(coord_label)):
                # get one gt centroid
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_pred)):
                    # get one pred centroid
                    centroid_image = np.array(list(coord_pred[m].centroid))
                    # calculate
                    distance = np.linalg.norm(centroid_image - centroid_label)

                    if distance < threshold:
                        TP += 1
                        del coord_pred[m]  # remove matched pred object
                        break
            FP = len(coord_pred)
            FN = len(coord_label) - TP
        else:
            raise NotImplementedError(
                f"{self.match_alg} is not implemented, please use 'hungarian' or 'ergodic' for match_alg."
            )

        return TP, FN, FP


class BinaryCenterAveragePrecisionMetric(BaseMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = 10,
                 bins: int = 50,
                 dilate_kernel_size: List[int] = [7, 7],
                 match_alg: str = 'hungarian',
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
            dis_thr (Union[List[int], int], optional): dis_thr of Euclidean distance,
                if List, closed interval. Defaults to [1, 10].
            cong_thr (float, optional): confidence threshold. Defaults to 0.5.
            dilate_kernel_size (List[int], optional): kernel size of cv2.dilated.
                The difference is that when [0, 0], the dilation algorithm will not be used. Defaults to [7, 7].
            match_alg (str, optional): 'hungarian' or 'ergodic' to match pred and gt,
                'ergodic'is the original implementation of PD_FA,
                based on the first-match principle.Usually, hungarian is fast and accurate. Defaults to 'hungarian'.
        """

        super().__init__(**kwargs)
        self.bins = bins
        self.dis_thr = dis_thr
        self.match_alg = match_alg
        self.dilate_kernel_size = dilate_kernel_size
        self.lock = threading.Lock()
        self.reset()

    # TODO: support AP.
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
            # to unit8 for ``convert2gray()``
            label = label > 0  # sometimes is 0-1, force to 255.
            label = label.astype('uint8')
            pred = pred > self.ap_fn
            pred = pred.astype('uint8')

            # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
            # 'int64' is for measure.label().

            label = convert2gray(label).astype('int64')
            pred = convert2gray(pred)
            dilated_pred = self._get_dilated(pred).astype('int64')

            for idx, threshold in enumerate(self.dis_thr):

                pred_img = measure.label(dilated_pred, connectivity=2)
                coord_pred = measure.regionprops(pred_img)
                label = measure.label(label, connectivity=2)
                coord_label = measure.regionprops(label)

                TP, FN, FP = self._calculate_tp_fn_fp(coord_label, coord_pred,
                                                      self.dis_thr)
                with self.lock:
                    self.TP[idx] += TP
                    self.FP[idx] += FP
                    self.FN[idx] += FN
            return

        if self.debug:
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

        if self.debug:
            print(
                f'{self.__class__.__name__} spend time for update: {time.time() - start_time}'
            )

    def get(self):
        """Compute metric

        Returns:
            _type_: Precision, Recall, micro-F1, shape == [1, num_threshold].
        """
        self.Precision = self.TP / (self.TP + self.FP)
        self.Recall = self.TP / (self.TP + self.FN)
        # micro F1 socre.
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision +
                                                      self.Recall)

        head = ['Threshold']
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

    def _get_dilated(self, image: np.array):
        """_summary_

        Args:
            image (np.array): hwc of np.array.
        """

        kernel = np.ones(self.dilate_kernel_size, np.uint8)
        if kernel.sum() == 0:
            return image

        dilated_image = cv2.dilate(image, kernel, iterations=1)

        return dilated_image

    def _calculate_tp_fn_fp(self, coord_label: np.array, coord_pred: np.array,
                            threshold: int):
        """_summary_

        Args:
            gt_centroids (np.array): shape = (N,2)
            pred_centroids (np.array): shape = (M,2)
            threshold (int): threshold, Euclidean Distance.

        Returns:
            _type_: _description_
        """
        TP = 0
        FN = 0
        FP = 0

        if self.match_alg == 'hungarian':
            centroid_lbl = np.array([prop.centroid for prop in coord_label])
            centroid_pre = np.array([prop.centroid for prop in coord_pred])
            if centroid_lbl.shape[0] == 0:
                centroid_lbl = np.empty((0, 2))
            if centroid_pre.shape[0] == 0:
                centroid_pre = np.empty((0, 2))
            eul_distance = np.linalg.norm(
                centroid_lbl[:, None, :] - centroid_pre[None, :, :],
                axis=-1)  # N*M, N==coord_label.shape, M = coord_pred.shape
            row_indexes, col_indexes = linear_sum_assignment(eul_distance)
            selec_distance = eul_distance[row_indexes, col_indexes]
            TP = np.sum(selec_distance < threshold)
            FP = len(coord_pred) - TP
            FN = len(coord_label) - TP

        elif self.match_alg == 'ergodic':
            for i in range(len(coord_label)):
                # get one gt centroid
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_pred)):
                    # get one pred centroid
                    centroid_image = np.array(list(coord_pred[m].centroid))
                    # calculate
                    distance = np.linalg.norm(centroid_image - centroid_label)

                    if distance < threshold:
                        TP += 1
                        del coord_pred[m]  # remove matched pred object
                        break
            FP = len(coord_pred)
            FN = len(coord_label) - TP
        else:
            raise NotImplementedError(
                f"{self.match_alg} is not implemented, please use 'hungarian' or 'ergodic' for match_alg."
            )

        return TP, FN, FP
