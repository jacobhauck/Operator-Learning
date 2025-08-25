import importlib
from typing import Mapping


def create_model(config: Mapping):
    """
    Create a model from its configuration.

    :param config: Dictionary of model configuration options. Must contain a
        'name' field to specify which model to load, which refers to the model's
        name within the `models` module.
    :return: An instance of the model specified by the given configuration
        settings.
    """
    config = dict(config)
    name = ['models'] + config.pop('name').split('.')
    py_module = importlib.import_module('.'.join(name[:-1]))

    return getattr(py_module, name[-1])(**config)
