"""Simple demo experiment for testing run_experiment script."""
from experiments import WandBExperiment

from ..demo import _print


class DemoSide(WandBExperiment):
    """Demonstration experiment that simply prints the provided config."""

    def run_wandb(self, config, run):
        print('Running demo SUB-SIDE Experiment.')
        print(run.name)
        _print(config)
