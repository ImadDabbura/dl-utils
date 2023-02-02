import re
from functools import partial

import matplotlib as plt
from fastprogress.fastprogress import format_time, master_bar, progress_bar


class Callback:
    """Base class for all callbacks."""

    _order = 0

    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        """
        Returns the name of the callback after removing the word `callback`
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f is not None:
            f()

    @staticmethod
    def camel2snake(name):
        """
        Convert name of callback by inserting underscores between small and capital
        letters. For example, `TestCallback` becomes `test_callback`.
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()


class TrainEvalCallback(Callback):
    """
    Tracks the number of iterations and epoch done and set training and eval
    modes.
    """

    _order = -10

    def before_fit(self):
        self.learner.n_iters = 0
        self.learner.pct_train = 0

    def after_batch(self):
        if self.learner.training:
            self.learner.n_iters += 1
            # Assuming here that all batches are of the same size
            # Otherwise, self.iters * self.n_epochs would be smaller
            # for the last batch
            self.learner.pct_train += 1 / (self.iters * self.n_epochs)

    def before_train(self):
        self.model.train()
        self.learner.training = True
        self.learner.pct_train = self.epoch / self.n_epochs

    def before_validate(self):
        self.model.eval()
        self.learner.training = False


class ProgressCallback(Callback):
    """Add progress bar as logger for tracking metrics."""

    _order = -20

    def before_fit(self):
        self.mbar = master_bar(range(self.learner.n_epochs))
        self.mbar.on_iter_begin()
        # Overwrite default learner logger
        self.learner.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.learner.iter)

    def before_train(self):
        self.set_pb()

    def before_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.learner.dl, parent=self.mbar)
        self.mbar.update(self.epoch)


class Recorder(Callback):
    _order = 50

    def before_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if self.training:
            for pg, lr in zip(self.opt.param_groups, self.lrs):
                lr.append(pg["lr"])
            self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        """
        Plot learning rates in the parameter group id `pgid`, default to the last parameter group.
        """
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        """
        Plot losses, optionally skip last `skip_last` losses.
        """
        n = len(self.losses) - skip_last
        plt.plot(self.losses[:n])

    def plot(self, skip_last=0, pgid=-1):
        """
        Plot both losses and learning rates.
        """
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])
