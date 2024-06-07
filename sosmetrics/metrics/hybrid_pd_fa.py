import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage.measure._regionprops import RegionProperties

from .base import BaseMetric, time_cost_deco
from .utils import (_TYPES, _adjust_dis_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray,
                    second_match_method)


class TargetPdPixelFa(BaseMetric):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thrs: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 dilate_kernel: Union[List[int], int] = [0, 0],
                 **kwargs: Any):
        """
        Target Level Pd and Pixel Level Fa.
        Original Code: https://github.com/XinyiYing/BasicIRSTD/blob/main/metrics.py

        Paper:
            @ARTICLE{9864119,
            author={Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, \
                Zaiping and Li, Miao and An, Wei and Guo, Yulan},
            journal={IEEE Transactions on Image Processing},
            title={Dense Nested Attention Network for Infrared Small Target Detection},
            year={2023},
            volume={32},
            number={},
            pages={1745-1758},
            keywords={Feature extraction;Object detection;Shape;Clutter;Decoding;Annotations;\
                Training;Infrared small target detection;deep learning;dense nested interactive module;\
                    channel and spatial attention;dataset},
            doi={10.1109/TIP.2022.3199107}}

        We have made the following improvements
            1. Supports multi-threading as well as batch processing.
            2. Supports secondary matching using mask iou.

        Original setting: conf_thr=0.5, dis_thrs=3, match_alg='forloop', second_match='none'

        Pipeline:
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
            conf_thr (float, Optional): Confidence threshold. Defaults to 0.5.
            dis_thrs (Union[List[int], int], optional): dis_thrs of Euclidean distance,
                if List, closed interval. . Defaults to [1,10].
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

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
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
                AT, TD, FD, NP = self._calculate_at_td_fd_np(
                    distances.copy(), coord_pred, threshold, gray_pred)
                with self.lock:
                    self.AT[idx] += AT
                    self.FD[idx] += FD
                    self.NP[idx] += NP
                    self.TD[idx] += TD

        # Packaged in the format we need, bhwc of np.array or hwc of list.
        labels, preds = convert2format(labels, preds)

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

    @time_cost_deco
    def get(self):

        self.FA = _safe_divide(self.FD, self.NP)
        self.PD = _safe_divide(self.TD, self.AT)

        if self.print_table:
            head = ['Threshold']
            head.extend(self.dis_thrs.tolist())
            table = PrettyTable()
            table.add_column('Threshold', self.dis_thrs)
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
        self.FA = np.zeros_like(self.dis_thrs)
        self.TD = np.zeros_like(self.dis_thrs)
        self.FD = np.zeros_like(self.dis_thrs)
        self.NP = np.zeros_like(self.dis_thrs)
        self.AT = np.zeros_like(self.dis_thrs)
        self.PD = np.zeros_like(self.dis_thrs)

    @property
    def table(self):
        all_metric = np.stack([
            self.dis_thrs, self.TD, self.AT, self.FD, self.NP, self.PD, self.FA
        ],
                              axis=1)
        df_pd_fa = pd.DataFrame(all_metric)
        df_pd_fa.columns = [
            'dis_thrs', 'TD', 'AT', 'FD', 'NP', 'target_Pd', 'pixel_Fa'
        ]
        return df_pd_fa

    def _calculate_at_td_fd_np(self, distances: np.ndarray,
                               coord_pred: List[RegionProperties],
                               threshold: int, pred_img: np.ndarray) -> Tuple:
        """_summary_

        Args:
            distances (np.array): distances in shape (num_lbl * num_pred)
            coord_pred (List[RegionProperties]): measure.regionprops(pred)
            threshold (int): _description_
            pred_img (np.array): _description_

        Returns:
            tuple[int, int, int, int]: AT, TD, FD, NP
        """
        num_lbl, num_pred = distances.shape
        true_img = np.zeros(pred_img.shape)

        if num_lbl * num_pred == 0:
            # no lbl or no pred
            TD = 0

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:,
                                  j] = np.nan  # Set inf to mark matched preds.
                        true_img[coord_pred[j].coords[:, 0],
                                 coord_pred[j].coords[:, 1]] = 1
                        break
            # get number of nan columns, is equal to TD
            TD = np.sum(np.isnan(distances)) // num_lbl

        elif self.match_alg == 'hungarian':
            # fix feasible in hungarian
            distances[np.isinf(distances)] = 1e10
            row_indexes, col_indexes = linear_sum_assignment(distances)
            selec_distance = distances[row_indexes, col_indexes]
            matched = selec_distance < threshold
            for j in col_indexes[
                    matched]:  # col_indexes present matched pred index.
                true_img[coord_pred[j].coords[:, 0],
                         coord_pred[j].coords[:, 1]] = 1
            TD = np.sum(matched)
        else:
            raise ValueError(f'Unknown match_alg: {self.match_alg}')

        FD = (pred_img - true_img).sum()
        NP = pred_img.shape[0] * pred_img.shape[1]
        AT = num_lbl
        return AT, TD, FD, NP

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}, '
                f'match_alg={self.match_alg}, '
                f'second_match={self.second_match}, '
                f'dilate_kernel={self.dilate_kernel})')
