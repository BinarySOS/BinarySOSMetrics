import threading
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.measure._regionprops import RegionProperties

from .base import BaseMetric, time_cost_deco
from .utils import (_TYPES, _adjust_dis_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format, convert2gray)


class PD_FAMetric(BaseMetric):

    def __init__(self,
                 conf_thr: float = 0.5,
                 dis_thr: Union[List[int], int] = [1, 10],
                 match_alg: str = 'forloop',
                 second_match: str = 'none',
                 **kwargs: Any):
        """
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
            2. Supports the Hungarian algorithm to match gt and pred, which is faster and more accurate.
            3. Supports secondary matching using mask iou.

        Original setting: conf_thr=0.5, dis_thr=3, match_alg='forloop', second_match='none'

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

        Args:
            conf_thr (float, Optional): Confidence threshold. Defaults to 0.5.
            dis_thr (Union[List[int], int], optional): dis_thr of Euclidean distance,
                if List, closed interval. . Defaults to [1,10].
            match_alg (str, optional): 'hungarian' or 'forloop' to match pred and gt,
                'forloop'is the original implementation of PD_FA,
                based on the first-match principle. Defaults to 'forloop'
                But, usually, hungarian is fast and accurate.
            second_math (str, optional): 'none' or 'mask_iou' to match pred and gt after distance matching. \
                Defaults to 'none'.
        """
        super().__init__(**kwargs)
        self.dis_thr = _adjust_dis_thr_arg(dis_thr)
        self.conf_thr = conf_thr
        self.match_alg = match_alg
        self.second_match = second_match
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
            # to unit8 for ``convert2gray()``
            pred = pred > self.conf_thr
            pred = pred.astype('uint8')

            # sometimes mask and label are read from cv2.imread() in default params, have 3 channels.
            # 'int64' is for measure.label().
            gray_pred = convert2gray(pred).astype('int64')
            gray_label = convert2gray(label).astype('int64')

            image = measure.label(gray_pred, connectivity=2)
            coord_pred = measure.regionprops(image)

            image = measure.label(gray_label, connectivity=2)
            coord_label = measure.regionprops(image)
            distances, mask_iou, _ = calculate_target_infos(
                coord_label, coord_pred, gray_pred.shape[0],
                gray_pred.shape[1])

            if self.debug:
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

        elif self.match_alg == 'hungarian':
            row_indexes, col_indexes = linear_sum_assignment(distances)
            selec_distance = distances[row_indexes, col_indexes]
            matched = selec_distance < threshold
            TD = np.sum(matched)
            for j in col_indexes[
                    matched]:  # col_indexes present matched pred index.
                true_img[coord_pred[j].coords[:, 0],
                         coord_pred[j].coords[:, 1]] = 1

        elif self.match_alg == 'forloop':
            for i in range(num_lbl):
                for j in range(num_pred):
                    if distances[i, j] < threshold:
                        distances[:,
                                  j] = np.inf  # Set inf to mark matched preds.
                        true_img[coord_pred[j].coords[:, 0],
                                 coord_pred[j].coords[:, 1]] = 1
                        break
            # get number of inf columns, is equal to TD
            TD = distances[distances == np.inf].size // num_lbl

        else:
            raise NotImplementedError(
                f'match_alg={self.match_alg} is not implemented.')

        FD = (pred_img - true_img).sum()
        NP = pred_img.shape[0] * pred_img.shape[1]
        AT = num_lbl
        return AT, TD, FD, NP

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(conf_thr={self.conf_thr}, '
                f'match_alg={self.match_alg})')
