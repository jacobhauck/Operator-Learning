"""Simple demo experiment for testing run_experiment script."""
from experiments import Experiment


def _print(d, depth=0):
    """Recursively print a dictionary"""
    for k, v in d.items():
        if isinstance(v, dict):
            print(' ' * (depth * 2) + f'{k}: {{')
            _print(v, depth + 1)
            print(' ' * (depth * 2) + '}')
        else:
            print(' ' * (depth * 2) + f'{k}: {v}')


class Demo(Experiment):
    """Demonstration experiment that simply prints the provided config."""

    def run(self, config):
        print('Running demo Experiment.')
        _print(config)
