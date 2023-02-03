import os
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from .utils import listify, setify


def compose(x, funcs, *args, order="_order", **kwargs):
    """
    Applies functions/transformations in `funcs` to the input `x` in the order
    of `order`.
    """
    for func in sorted(listify(funcs), key=lambda x: getattr(x, order, 0)):
        x = func(x, *args, **kwargs)
    return x


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """
    Returns two dataloaders: 1 for training and 1 for 1 for validation. The
    validation dataloader has twice the batch size and doesn't shuffle data.
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, **kwargs),
    )


class DataBunch:
    """
    Utility class that stores train and valid dataloaders as well as number of
    input features and number of output (1 for regression, number of classes
    for classfication).
    """

    def __init__(
        self, train_dl: DataLoader, valid_dl: DataLoader, c_in: int, c_out: int
    ):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c_in = c_in
        self.c_out = c_out

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid.dataset
