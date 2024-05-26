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


def convert2iterable(labels: _TYPES,
                     preds: _TYPES) -> Tuple[Iterable, Iterable]:
    """Convert labels and preds to Iterable, bhwc or [hwc, ...] and scale preds to 0-1.
        Preds must be probability image in 0-1.
        If path, we will grayscale it and /255 to 0-1.

    Args:
        labels (_TYPES): [str, ], [hwc, ...], and others.
        preds (_TYPES): [str, ], [hwc, ...], and others.
    Raises:
        ValueError: _description_

    Returns:
        Tuple[Iterable, Iterable]: _description_
    """
    if isinstance(labels, (np.ndarray, torch.Tensor)):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

        #  chw/bchw -> hwc/bhwc
        labels, preds = channels2last(labels, preds)
        #  hwc -> bhwc
        labels, preds = convert2batch(labels, preds)

        if np.any((preds < 255) & (preds > 1)):
            # convert to 0-1, if preds is not probability image.
            preds = preds / 255.0
        return labels, preds

    if isinstance(labels, str):
        # hwc in np.uint8, -> [hwc]
        labels = cv2.imread(labels, cv2.IMREAD_GRAYSCALE)[..., None] / 255

        # bwc in np.uint8, -> probability image from 0-1.
        preds = cv2.imread(preds, cv2.IMREAD_GRAYSCALE)[..., None] / 255
        return [labels], [preds]

    if isinstance(labels, list) and isinstance(labels[0], str):
        labels = [
            cv2.imread(label, cv2.IMREAD_GRAYSCALE)[..., None]
            for label in labels
        ]
        preds = [
            cv2.imread(pred, cv2.IMREAD_GRAYSCALE)[..., None] / 255.
            for pred in preds
        ]
        return labels, preds

    if isinstance(labels, list) and isinstance(labels[0],
                                               (np.ndarray, torch.Tensor)):
        labels = [
            label.detach().cpu().numpy()
            if isinstance(label, torch.Tensor) else label for label in labels
        ]
        # preds = [
        #     pred.detach().cpu().numpy()
        #     if isinstance(pred, torch.Tensor) else pred for pred in preds
        # ]

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
        return labels, preds

    raise ValueError(
        f'labels should be np.array, torch.tensor, str or list of str, but got {type(labels)}'
    )


def convert2gray(image: np.array) -> np.array:
    """ Image hwc to  hw

    Args:
        image (np.array): _description_

    Returns:
        np.array: _description_
    """
    channels = image.shape[-1]
    if channels == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 1:
        gray_image = np.squeeze(image, axis=-1)
    return gray_image
