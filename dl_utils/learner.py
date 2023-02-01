class CancelFitException(Exception):
    """Stop training and exit"""


class CancelEpochException(Exception):
    """Stop current epoch and move to next epoch."""


class CancelTrainException(Exception):
    """Stop training current batch and move to validation."""


class CancelValidException(Exception):
    """Stop validation phase and move to next epoch"""


class CancelBatchException(Exception):
    """Stop current batch and move to next batch."""
