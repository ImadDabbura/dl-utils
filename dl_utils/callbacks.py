import re
import time
from functools import partial

import matplotlib as plt
import torch
from fastprogress.fastprogress import format_time, master_bar, progress_bar

from .utils import listify


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


class AvgStats:
    def __init__(self, metrics, training=True):
        self.metrics = listify(metrics)
        self.training = training

    def reset(self):
        self.tot_loss = 0
        self.count = 0
        self.tot_metrics = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        """Returns a list of both loss and metrics."""
        return [self.tot_loss.item()] + self.tot_metrics

    @property
    def avg_stats(self):
        """Returns the average of loss/metrics."""
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.training else 'valid'}: {self.avg_stats}"

    def accumulate(self, learner):
        """Evaluate metrics and accumulate them to at the epoch level."""
        bs = learner.xb.shape[0]
        self.count += bs
        self.tot_loss += learner.loss * bs
        for i, metric in enumerate(self.metrics):
            self.tot_metrics[i] += metric(learner.pred, learner.yb) * bs


class AvgStatsCallback(Callback):
    _order = -10

    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def before_fit(self):
        metrics_names = ["loss"] + [
            metric.__name__ for metric in self.train_stats.metrics
        ]
        names = (
            ["epoch"]
            + [f"train_{name}" for name in metrics_names]
            + [f"valid_{name}" for name in metrics_names]
            + ["time"]
        )
        self.logger(names)

    def before_epoch(self):
        """Reset metrics/loss."""
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        """Evaluate metrics and accumulate them."""
        stats = self.train_stats if self.training else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.learner)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)
