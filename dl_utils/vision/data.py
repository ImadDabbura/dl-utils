import mimetypes

import PIL

from ..data import ItemList, get_files

IMAGE_EXTENSIONS = [
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
]


class ImageList(ItemList):
    @classmethod
    def from_files(
        cls,
        path,
        extensions=IMAGE_EXTENSIONS,
        include=None,
        recurse=True,
        **kwargs
    ):
        """
        Build an image list from list of files in the `path` end with
        extensions, optionally recursively.
        """
        return cls(get_files(path, extensions, include, recurse), **kwargs)

    def get(self, item):
        """Open an image using PIL."""
        return PIL.Image.open(item)
