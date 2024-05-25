import threading
import time
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.measure._regionprops import RegionProperties

from .base import BaseMetric
from .utils import (_TYPES, bbox_overlaps, convert2gray, convert2iterable,
                    target_mask_iou)


class BinaryCenterMetric(BaseMetric):

    def __init__(self,
                 dis_thr: Union[List[int], int] = [1, 10],
                 conf_thr: float = 0.5,
                 dilate_kernel_size: List[int] = [7, 7],
                 match_alg: str = 'hungarian',
                 iou_mode: str = 'none',
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
            match_alg (str, optional): 'hungarian' or 'forloop' to match pred and gt,
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle.Usually, hungarian is fast and accurate. Defaults to 'hungarian'.
            iou_mode (str, optional): 'none', 'mask_iou', 'bbox_iou'. if not 'none', (eul_dis + '**_iou') for match.
        """

        super().__init__(**kwargs)
        if isinstance(dis_thr, int):
            self.dis_thr = np.array([dis_thr])
        else:
            self.dis_thr = np.arange(dis_thr[0], dis_thr[1] + 1)
        self.conf_thr = conf_thr
        self.match_alg = match_alg
        self.dilate_kernel_size = dilate_kernel_size
        self.iou_mode = iou_mode
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

            pred_img = measure.label(dilated_pred, connectivity=2)
            coord_pred = measure.regionprops(pred_img)
            label = measure.label(label, connectivity=2)
            coord_label = measure.regionprops(label)

            distances = self._calculate_infos(coord_label, coord_pred,
                                              pred_img)
            for idx, threshold in enumerate(self.dis_thr):
                TP, FN, FP = self._calculate_tp_fn_fp(distances.copy(),
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

    def _calculate_infos(self, coord_label: List[RegionProperties],
                         coord_pred: List[RegionProperties],
                         img: np.array) -> np.array:
        """calculate distances between label and pred. single image.

        Args:
            coord_label (List[RegionProperties]): measure.regionprops(label)
            coord_pred (List[RegionProperties]): measure.regionprops(pred)
            img (np.array): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            np.array: distances between label and pred. (num_lbl * num_pred)
        """

        num_lbl = len(coord_label)  # number of label
        num_pred = len(coord_pred)  # number of pred

        if num_lbl * num_pred == 0:
            return np.empty(
                (num_lbl,
                 num_pred))  # gt=0 or pred=0, (num_lbl, 0) or (0, num_pred)

        # eul distance
        centroid_lbl = np.array([prop.centroid for prop in coord_label
                                 ]).astype(np.float32)  # N*2
        centroid_pred = np.array([prop.centroid for prop in coord_pred
                                  ]).astype(np.float32)  # M*2
        eul_distance = np.linalg.norm(centroid_lbl[:, None, :] -
                                      centroid_pred[None, :, :],
                                      axis=-1)  # num_lbl * num_pred

        # bbox iou
        bbox_lbl = torch.tensor([prop.bbox for prop in coord_label],
                                dtype=torch.float32)  # N*4
        bbox_pre = torch.tensor([prop.bbox for prop in coord_pred],
                                dtype=torch.float32)  # M*4
        bbox_iou = bbox_overlaps(
            bbox_lbl, bbox_pre, mode='iou',
            is_aligned=False).numpy()  # num_lbl * num_pred

        # mask iou
        pixel_coords_lbl = [prop.coords for prop in coord_label]
        pixel_coords_pred = [prop.coords for prop in coord_pred]
        gt_mask = np.zeros((num_lbl, *img.shape))  # [num_lbl, H, W]
        pred_mask = np.zeros((num_pred, *img.shape))  # [num_pred, H, W]
        for i in range(num_lbl):
            gt_mask[i, pixel_coords_lbl[i][:, 0], pixel_coords_lbl[i][:,
                                                                      1]] = 1
        for i in range(num_pred):
            pred_mask[i, pixel_coords_pred[i][:, 0],
                      pixel_coords_pred[i][:, 1]] = 1
        mask_iou = target_mask_iou(gt_mask, pred_mask)  # num_lbl * num_pred

        if self.debug:
            print(f'centroid_lbl={centroid_lbl}')
            print(f'centroid_pred={centroid_pred}')
            print(f'bbox_iou={bbox_iou}')
            print(f'mask_iou={mask_iou}')
            print(f'eul_distance={eul_distance}')
            print('____' * 20)

        if self.iou_mode == 'mask_iou':
            reciprocal = np.reciprocal(mask_iou, where=mask_iou != 0)
            return eul_distance + reciprocal
        elif self.iou_mode == 'bbox_iou':
            reciprocal = np.reciprocal(bbox_iou, where=bbox_iou != 0)
            return eul_distance + reciprocal
        elif self.iou_mode == 'none':
            return eul_distance

        else:
            raise NotImplementedError(
                f"{self.iou_mode} is not implemented, please use 'none', 'bbox_iou', 'mask_iou' for distances."
            )

    def _calculate_tp_fn_fp(self, distances: np.array,
                            threshold: int) -> Tuple[int]:
        """_summary_

        Args:
            distances (np.array): distances in shape (num_lbl * num_pred)
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
