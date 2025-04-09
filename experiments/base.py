"""
Base utilities for managing and running experiments.

Classes
-------
    - Experiment
"""
import abc
from typing import Mapping


class Experiment(abc.ABC):
    """
    Represents a type of experiment.

    Used for automatic running in run_experiment. All experiment classes should
    be subclasses of Experiment.
    """

    @abc.abstractmethod
    def run(self, config: Mapping) -> None:
        raise NotImplemented
