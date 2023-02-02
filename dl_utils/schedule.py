import math
from functools import partial, wraps

import torch

from .utils import listify


def annealer(func):
    wraps(func)

    def annealer_wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return annealer_wrapper


@annealer
def no_sched(start, end, pos):
    """Constant schedular."""
    return start


@annealer
def lin_sched(start, end, pos):
    """Linear scheduler."""
    return start + (end - start) * pos


@annealer
def cos_sched(start, end, pos):
    """Cosine scheduler."""
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def exp_sched(start, end, pos):
    """Exponential scheduler."""
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    """
    combine two cosine schedulers where first scheduler goes from `start` to
    `high` and second scheduler goes from `high` to `end`.
    """
    return [cos_sched(start, high), cos_sched(high, end)]
