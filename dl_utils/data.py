import os
from pathlib import Path

import numpy as np
import torch
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
        return self.valid_dl.dataset


class L:
    """Extensible list container that adds some functionality to a list"""

    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx) == len(self)  # bool mask
            return [o for m, o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__}: ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + ", ...]"
        return res


class ItemList(L):
    """Base class for all type of datasets such as image, text, etc."""

    def __init__(self, items, path=".", tfms=None):
        super().__init__(items)
        self.path = path
        self.tfms = tfms

    def __repr__(self):
        return super().__repr__() + f"\nPath: {self.path}"

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, item):
        """Every class that inherits from ItemList has to override this method."""
        return item

    def _get(self, item):
        """Returns items after applying all transforms `tfms`."""
        return compose(self.get(item), self.tfms)

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        if isinstance(idx, list):
            return [self._get(item) for item in items]
        return self._get(items)


def _get_files(path, fs, extensions=None):
    """Get filenames in `path` that have extension `extensions`."""
    path = Path(path)
    res = [
        path / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, include=None, recurse=False):
    """
    Get filenames in `path` that have extension `extensions` starting
    with `path` and optionally recurse to subdirectories.
    """
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, fs) in enumerate(os.walk(path)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, fs, extensions)
        return res
    else:
        fs = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, fs, extensions)


def random_splitter(f_name: str, p_valid: float = 0.2):
    return np.random.random() < p_valid


def grandparent_splitter(
    f_name: Path, valid_name: str = "valid", train_name: str = "train"
):
    """
    Split items based on whether they fall under validation or training
    direcotories. This assumes that the directory structure is
    train/label/items or valid/label/items.
    """
    gp = f_name.parent.parent.name
    if gp == valid_name:
        return True
    elif gp == train_name:
        return False
    return


def split_by_func(items, func):
    mask = [func(o) for o in items]
    # `None` values will be filtered out
    val = [o for o, m in zip(items, mask) if m]
    train = [o for o, m in zip(items, mask) if m is False]
    return train, val


class SplitData:
    """Split Item list into train and validation data lists."""

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData objects and be able to load
    # it back without recursion errors
    def __setstate__(self, data):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, item_list, split_func):
        """Split item list by splitter function and returns a SplitData object."""
        train_files, val_files = split_by_func(item_list.items, split_func)
        train_list, val_list = map(item_list.new, (train_files, val_files))
        return cls(train_list, val_list)

    def to_databunch(self, bs, c_in, c_out, **kwargs):
        """Returns a DataBunch object using train and valid datasets."""
        dls = get_dls(self.train, self.valid, bs, **kwargs)
        return DataBunch(*dls, c_in=c_in, c_out=c_out)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n---------\nTrain - {self.train}\n\n"
            f"Valid - {self.valid}\n"
        )


def parent_labeler(f_name: Path):
    """Label a file based on its parent directory."""
    return f_name.parent.name


def label_by_func(splitted_data, label_func, proc_x=None, proc_y=None):
    """Label splitted data using `label_func`."""
    train = LabeledData.label_by_func(
        splitted_data.train, label_func, proc_x=proc_x, proc_y=proc_y
    )
    valid = LabeledData.label_by_func(
        splitted_data.valid, label_func, proc_x=proc_x, proc_y=proc_y
    )
    return SplitData(train, valid)


class LabeledData:
    """
    Create a labeled data and expose both x & y as item lists after passing
    them through all processors.
    """

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x = self.process(x, proc_x)
        self.y = self.process(y, proc_y)
        self.proc_x = proc_x
        self.proc_y = proc_y

    def process(self, item_list, proc):
        return item_list.new(compose(item_list.items, proc))

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx):
        return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (
            isinstance(idx, torch.LongTensor) and not idx.ndim
        )
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc._deprocess(item) if isint else proc.deprocess(item)
        return item

    @staticmethod
    def _label_by_func(ds, label_func, cls=ItemList):
        return cls([label_func(o) for o in ds.items], path=ds.path)

    @classmethod
    def label_by_func(cls, item_list, label_func, proc_x=None, proc_y=None):
        return cls(
            item_list,
            LabeledData._label_by_func(item_list, label_func),
            proc_x=proc_x,
            proc_y=proc_y,
        )
