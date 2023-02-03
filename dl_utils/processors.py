from .utils import uniqueify


class Processor:
    def process(self, item):
        item


class CategoryProcessor(Processor):
    def __init__(self):
        self.vocab = None

    def __call__(self, items):
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {k: v for v, k in enumerate(self.vocab)}
        return [self.process(item) for item in items]

    def process(self, item):
        return self.otoi[item]

    def _deprocess(self, idx):
        self.vocab[idx]

    def deprocess(self, idxs):
        return [self.vocab[idx] for idx in idxs]
