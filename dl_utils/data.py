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
