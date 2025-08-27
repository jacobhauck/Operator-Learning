import experiments
import models

import torch


class FNODemoExperiment(experiments.Experiment):
    def run(self, config, name, group=None):
        model = models.create_model(config['model'])

