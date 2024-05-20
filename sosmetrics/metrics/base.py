from typing import Any

from .utils import _TYPES


class BaseMetric:

    def __init__(self, **kwargs: Any):
        """
        Args:
            debug (bool): Debug mode. if True, print process time. Default is False.
            print_table (bool): Print table, if True, print table. Default is True.
        """
        self.debug = kwargs.get('debug', False)
        self.print_table = kwargs.get('print_table', True)

    def update(self, preds: _TYPES, labels: _TYPES):
        """Support CHW, BCHW, HWC,BHWC, Image Path, or in their list form (except BHWC/BCHW),
            like [CHW, CHW, ...], [HWC, HWC, ...], [Image Path, Image Path, ...].

            Although support Image Path, but not recommend.
            Note :
                All preds are probabilities image from 0 to 1 in default.
                If images, Preds must be probability image from 0 to 1 in default.
                If path, Preds must be probabilities image from 0-1 in default, if 0-255,
                    we are force /255 to 0-1 to probability image.
        Args:
            labels (_TYPES): Ground Truth images or image paths in list or single.
            preds (_TYPES): Preds images or image paths in list or single.
        """
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def table(self):
        raise NotImplementedError
