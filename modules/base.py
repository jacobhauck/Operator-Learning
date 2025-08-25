import importlib
from typing import Mapping


def create_module(config: Mapping):
    """
    Create a Module from its configuration.

    :param config: Dictionary of Module configuration options. Must contain a
        'name' field to specify which Module to load, which refers to the Module's
        name within the `modules` module.
    :return: An instance of the Module specified by the given configuration
        settings.
    """
    config = dict(config)
    name = ['modules'] + config.pop('name').split('.')
    py_module = importlib.import_module('.'.join(name[:-1]))

    return getattr(py_module, name[-1])(**config)
