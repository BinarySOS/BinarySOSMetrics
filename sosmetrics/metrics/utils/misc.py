from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
import torch

_TYPES = Union[np.array, torch.tensor, str, List[str], List[np.array],
               List[torch.tensor]]


def channels2last(labels: np.array,
                  preds: np.array) -> Tuple[np.array, np.array]:
    """ chw -> hwc,  bchw -> bhwc

    Args:
        labels (np.array): _description_
        preds (np.array): _description_

    Returns:
        Tuple[np.array, np.array]: _description_
    """
    assert labels.shape == preds.shape, f'labels and preds should have the same shape, \
        but got labels.shape={labels.shape}, preds.shape={preds.shape}'

    if labels.shape[-1] not in [1, 3]:
        if labels.ndim == 3:
            # chw -> hwc
            labels = labels.transpose((1, 2, 0))
            preds = preds.transpose((1, 2, 0))
        elif labels.ndim == 4:
            # bchw/ -> bhwc
            labels = labels.transpose((0, 2, 3, 1))
            preds = preds.transpose((0, 2, 3, 1))
        else:
            raise ValueError(
                f'labels.ndim or preds.ndim should be 3 or 4, but got {labels.ndim} and {preds.ndim}'
            )

    assert (labels.shape[-1] in [1, 3]) and (
        preds.shape[-1] in [1, 3]
    ), f'labels and preds should have 3 or 1 channels, but got labels.shape={labels.shape}, preds.shape={preds.shape}'

    return labels, preds


def convert2batch(labels: np.array,
                  preds: np.array) -> Tuple[np.array, np.array]:
    """ convert labels and preds to batch format

    Args:
        labels (np.array): _description_
        preds (np.array): _description_

    Returns:
        Tuple[np.array, np.array]: _description_
    """
    assert labels.shape == preds.shape, f'labels and preds should have the same shape, \
        but got labels.shape={labels.shape}, preds.shape={preds.shape}'

    if labels.ndim == 3:
        labels = labels[np.newaxis, ...]
        preds = preds[np.newaxis, ...]
    return labels, preds


def is_batchable(data: List[np.ndarray]) -> bool:
    try:
        data = np.stack(data, axis=0)
        return True
    except TypeError:
        return False


def convert2format(labels: _TYPES, preds: _TYPES) -> Tuple[Iterable, Iterable]:
    """Convert labels and preds to Iterable , bhwc or [hwc, ...] and scale preds, label to 0-1.
        preds will be convert to probability image in 0-1.
        If path, we will grayscale it and /255 to 0-1.

    Args:
        labels (_TYPES): [str, ], [hwc, ...], and others.
        preds (_TYPES): [str, ], [hwc, ...], and others.
    Raises:
        ValueError: _description_

    Returns:
        Tuple[Iterable, Iterable]:  labels, preds in Iterable format.
    """
    if isinstance(labels, (np.ndarray, torch.Tensor)):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

        #  chw/bchw -> hwc/bhwc
        labels, preds = channels2last(labels, preds)
        #  hwc -> bhwc
        labels, preds = convert2batch(labels, preds)

        if np.any((preds <= 255) & (preds > 1)):
            # convert to 0-1, if preds is not probability image.
            preds = preds / 255.0

        labels = (labels > 0).astype('uint8')

    elif isinstance(labels, str):
        # hwc in np.uint8, -> [hwc]
        labels = cv2.imread(labels, cv2.IMREAD_GRAYSCALE)[..., None] / 255

        # bwc in np.uint8, -> probability image from 0-1.
        preds = cv2.imread(preds, cv2.IMREAD_GRAYSCALE)[..., None] / 255.
        labels = [labels]
        preds = [preds]

    elif isinstance(labels, list) and isinstance(labels[0], str):
        labels = [
            cv2.imread(label, cv2.IMREAD_GRAYSCALE)[..., None] / 255
            for label in labels
        ]
        preds = [
            cv2.imread(pred, cv2.IMREAD_GRAYSCALE)[..., None] / 255.
            for pred in preds
        ]

    elif isinstance(labels, list) and isinstance(labels[0],
                                                 (np.ndarray, torch.Tensor)):
        # labels = [
        #     label.detach().cpu().numpy()
        #     if isinstance(label, torch.Tensor) else label for label in labels
        # ]
        new_labels = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy()
            label = (label > 0).astype('uint8')
            new_labels.append(label)
        labels = new_labels

        new_preds = []
        for pred in preds:
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            if np.any((pred < 255) & (pred > 1)):
                # convert to 0-1, if preds is not probability image.
                pred = pred / 255.
            new_preds.append(pred)
        preds = new_preds
        tmp = [
            channels2last(label, pred) for label, pred in zip(labels, preds)
        ]
        labels = [_tmp[0] for _tmp in tmp]
        preds = [_tmp[1] for _tmp in tmp]

    else:
        raise ValueError(
            f'labels should be np.array, torch.tensor, str or list of str, but got {type(labels)}'
        )
    if isinstance(labels, list) and is_batchable(labels):
        labels = np.stack(labels, axis=0)
        preds = np.stack(preds, axis=0)
    return labels, preds


def convert2gray(image: np.ndarray) -> np.ndarray:
    """ Image bhwc/hwc to bhw/hw.
        Adaptation of images read by cv2.imread by default (usually 3 channels).

    Args:
        image (np.array): _description_

    Returns:
        np.array: _description_
    """
    channels = image.shape[-1]
    if channels == 3:
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image.astype('uint8'),
                                      cv2.COLOR_BGR2GRAY)
        else:
            gray_image = np.stack([
                cv2.cvtColor(image[i].astype('uint8'), cv2.COLOR_BGR2GRAY)
                for i in range(image.shape[0])
            ],
                                  axis=0)
    elif channels == 1:
        gray_image = np.squeeze(image, axis=-1)
    return gray_image


def _adjust_conf_thr_arg(
        thresholds: Union[int, List[float], np.ndarray]) -> np.ndarray:
    """Convert threshold arg for list and int to np.array format.

    Args:
        thresholds (Union[int, List[float], np.ndarray]):
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced \
                from 0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins \
                for the calculation
            - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
                bins for the calculation.

    Returns:
        np.ndarray: _description_
    """
    if isinstance(thresholds, int):
        return np.linspace(0, 1, thresholds, endpoint=True)
    if isinstance(thresholds, list):
        return np.array(thresholds)
    return thresholds


def _adjust_dis_thr_arg(
        thresholds: Union[int, float, List[float], np.ndarray]) -> np.ndarray:
    """Convert threshold arg for list and int to np.array format.

    Args:
        thresholds (Union[int, float, List[float], np.ndarray]):
            - If set to an `int` , will use this value to distance threshold.
            - If set to an `list` of float or int, will use the indicated thresholds \
                in the list as bins for the calculation
            - If set to an 1d `array` of floats, will use the indicated thresholds in the array as
            bins for the calculation.
            if List, closed interval.

    Returns:
        np.ndarray: _description_
    """
    if isinstance(thresholds, (int, float)):
        return np.array([thresholds])
    if isinstance(thresholds, list):
        return np.linspace(*thresholds,
                           int(np.round(
                               (thresholds[1] - thresholds[0]) / 1)) + 1,
                           endpoint=True)
    return thresholds


def _safe_divide(
        num: Union[torch.Tensor, np.ndarray],
        denom: Union[torch.Tensor, np.ndarray],
        zero_division: float = 0.0) -> Union[torch.Tensor, np.ndarray]:
    """Safe division, by preventing division by zero.

    Args:
        num (Union[Tensor, np.ndarray]): _description_
        denom (Union[Tensor, np.ndarray]): _description_
        zero_division (float, optional): _description_. Defaults to 0.0.

    Returns:
        Union[Tensor, np.ndarray]: _description_
    """
    if isinstance(num, np.ndarray):
        num = num if np.issubdtype(num.dtype,
                                   np.floating) else num.astype(float)
        denom = denom if np.issubdtype(denom.dtype,
                                       np.floating) else denom.astype(float)
        results = np.divide(
            num,
            denom,
            where=(denom != np.array([zero_division]).astype(float)))
    else:
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        zero_division = torch.tensor(zero_division).float()
        results = torch.where(denom != 0, num / denom, zero_division)
    return results
