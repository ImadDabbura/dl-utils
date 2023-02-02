import math
from functools import partial, wraps

import torch

from .utils import listify


def annealer(func):
    wraps(func)

    def annealer_wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return annealer_wrapper
