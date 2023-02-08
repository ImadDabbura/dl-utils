import torch

from .callbacks import (
    CancelBatchException,
    CancelEpochException,
    CancelFitException,
    CancelTrainException,
    CancelValidException,
    ProgressCallback,
    Recorder,
    TrainEvalCallback,
)
from .utils import listify, setify


def params_getter(model):
    return model.parameters()


class Learner:
    ALL_CBS: set[str] = {
        "before_fit",
        "before_epoch",
        "before_train",
        "before_validate",
        "before_batch",
        "after_pred",
        "after_loss",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_train",
        "after_train",
        "after_cancel_validate",
        "after_validate",
        "after_cancel_epoch",
        "after_epoch",
        "after_cancel_fit",
        "after_fit",
    }

    def __init__(
        self,
        model,
        data,
        loss_func,
        opt_func,
        lr=1e-2,
        splitter=params_getter,
        cbs=None,
        cb_funcs=None,
    ):
        """
        Learner is a basic class that handles training loop of pytorch model
        and utilize a systems of callbacks that makes training loop very
        customizable and extensible. You just need to provide a list of
        callbacks and callback functions.

        Parameters
        ----------
        model: pytorch module/model.
        data: DataBunch object that contains both train and validation data loaders.
        loss_func: Loss function.
        opt_func: Optimizer function/class.
        lr: Learning rate.
        splitter: function to split model's parameters into groups, default all
            parameters belong to 1 group.
        cbs: list of callbacks of type `Callback`.
        cb_funcs: list of callback functions to call.
        """
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.lr = lr
        self.splitter = splitter
        self.opt, self.training = None, False
        # We can customize it & use progress bar or log to a file
        self.logger = print

        # Callbacks
        self.cbs = []
        # Add default useful callbacks
        self.add_cbs([TrainEvalCallback(), ProgressCallback(), Recorder()])
        self.add_cbs(cbs)
        self.add_cbs(cb_func() for cb_func in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in setify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_learner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.cbs.remove(cb)

    def _one_batch(self, i, xb, yb):
        self.iter = i
        self.xb, self.yb = xb, yb
        try:
            self("before_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if self.training:
                self.loss.backward()
                self("after_backward")
                self.opt.step()
                self("after_step")
                self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def _all_batches(self, dl):
        self.iters = len(dl)
        for i, (xb, yb) in enumerate(dl):
            self._one_batch(i, xb, yb)

    def fit(self, epochs, cbs=None, reset_opt=False):
        cbs = listify(cbs)
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        self.n_epochs = epochs
        self.loss = torch.tensor(0.0)
        try:
            self("before_fit")
            for epoch in range(self.n_epochs):
                try:
                    self.epoch = epoch
                    self.dl = self.data.train_dl
                    self("before_epoch")

                    try:
                        self("before_train")
                        self._all_batches(self.data.train_dl)
                    except CancelTrainException:
                        self("after_cancel_train")
                    finally:
                        self("after_train")

                    try:
                        self.dl = self.data.valid_dl
                        self("before_validate")
                        with torch.no_grad():
                            self._all_batches(self.data.valid_dl)
                    except CancelValidException:
                        self("after_cancel_validate")
                    finally:
                        self("after_validate")
                except CancelEpochException:
                    self("after_cancel_epoch")
                finally:
                    self("after_epoch")
        except CancelFitException:
            self("after_cancel_fit")
        finally:
            self("after_fit")
            # self.remove_cbs(cbs)

    def __call__(self, cb_name):
        assert (
            cb_name in Learner.ALL_CBS
        ), f"{cb_name} isn't a valid callback name"
        for cb in sorted(self.cbs, key=lambda x: x._order):
            cb(cb_name)
