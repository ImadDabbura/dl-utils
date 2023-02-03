from pathlib import Path

from ..data import ItemList, get_files


def read_file(fn):
    with open(fn, encoding="utf8") as f:
        return f.read()


class TextList(ItemList):
    @classmethod
    def from_files(
        cls, path, extensions=".txt", include=None, recurse=True, **kwargs
    ):
        """
        Build an text list from list of files in the `path` end with
        extensions, optionally recursively.
        """
        return cls(
            get_files(path, extensions, include, recurse), path, **kwargs
        )

    def get(self, i):
        """Returns text in the file as string if `i` is path to a file."""
        if isinstance(i, Path):
            return read_file(i)
        return i
