from collections.abc import Iterable


def listify(obj):
    "Change type of any object into a list."
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, Iterable):
        return list(obj)
    return [obj]


def tuplify(obj):
    "Change type of any object into a tuple."
    if isinstance(obj, tuple):
        return obj
    return tuple(listify(obj))


def setify(obj):
    "Change type of any object into a set."
    if isinstance(obj, set):
        return obj
    return set(listify(obj))


def uniqueify(x, sort=False):
    res = listify(setify(x))
    if sort:
        res.sort()
    return res
