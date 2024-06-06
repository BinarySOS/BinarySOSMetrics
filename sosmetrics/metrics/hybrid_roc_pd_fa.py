import threading
from typing import Any, List, Union

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import auc

from .base import time_cost_deco
from .hybrid_pd_fa import TargetPdPixelFa
from .utils import (_TYPES, _adjust_conf_thr_arg, _safe_divide,
                    calculate_target_infos, convert2format,
                    get_label_coord_and_gray, get_pred_coord_and_gray,
                    second_match_method)


class TargetPdPixelFaROC(TargetPdPixelFa):

    def __init__(self,
                 conf_thrs: Union[int, List[float], np.ndarray] = 10,
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
            conf_thrs (float, Optional): Confidence threshold. Defaults to 0.5.
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
        self.conf_thrs = _adjust_conf_thr_arg(conf_thrs)
        super().__init__(dis_thrs=dis_thrs,
                         conf_thr=0.5,
                         match_alg=match_alg,
                         second_match=second_match,
                         dilate_kernel=dilate_kernel,
                         **kwargs)
        self.lock = threading.Lock()
        self.reset()

    @time_cost_deco
    def update(self, labels: _TYPES, preds: _TYPES) -> None:

        def evaluate_worker(self, label: np.array, pred: np.array) -> None:
            # to unit8 for ``convert2gray()``
            coord_label, gray_label = get_label_coord_and_gray(label)

            for idx, conf_thr in enumerate(self.conf_thrs):
                coord_pred, gray_pred = get_pred_coord_and_gray(
                    pred.copy(), conf_thr, self.dilate_kernel)
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
                        print(f'After second matche eul distance={distances }')
                        print('____' * 20)

                for jdx, threshold in enumerate(self.dis_thrs):
                    AT, TD, FD, NP = self._calculate_at_td_fd_np(
                        distances.copy(), coord_pred, threshold, gray_pred)
                    with self.lock:
                        self.AT[jdx, idx] += AT
                        self.FD[jdx, idx] += FD
                        self.NP[jdx, idx] += NP
                        self.TD[jdx, idx] += TD

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

        index = np.argsort(self.FA, axis=-1)
        fa = np.take_along_axis(self.FA, index, axis=1)
        pd = np.take_along_axis(self.PD, index, axis=1)
        fa = np.concatenate([np.zeros((fa.shape[0], 1)), fa], axis=-1)
        pd = np.concatenate([np.ones((pd.shape[0], 1)), pd], axis=1)

        self.auc = [auc(fa[i], pd[i]) for i in range(len(self.dis_thrs))]

        if self.print_table:
            head = ['Dis—Thr']
            head.extend(self.dis_thrs.tolist())
            table = PrettyTable()
            table.field_names = head
            auc_row = ['AUC-ROC']
            auc_row.extend(['{:.4f}'.format(num) for num in self.auc])
            table.add_row(auc_row)

            print(table)

        return self.PD, self.FA, self.auc

    def reset(self) -> None:
        self.FA = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.TD = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.FD = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.NP = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.AT = np.zeros((len(self.dis_thrs), len(self.conf_thrs)))
        self.PD = np.zeros(len(self.dis_thrs))
        self.auc = np.zeros(len(self.dis_thrs))

    @property
    def table(self):
        all_metric = np.vstack([self.dis_thrs, self.auc])
        df = pd.DataFrame(all_metric)
        df.index = ['Dis-Thr', 'AUC-ROC']
        return df

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(match_alg={self.match_alg}, '
                f'second_match={self.second_match})')
