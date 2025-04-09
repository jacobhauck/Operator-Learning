"""Simple demo experiment for testing run_experiment script."""
from experiments import Experiment

from ..demo import _print


class DemoSide(Experiment):
    """Demonstration experiment that simply prints the provided config."""

    def run(self, config):
        print('Running demo SUB-SIDE Experiment.')
        _print(config)
