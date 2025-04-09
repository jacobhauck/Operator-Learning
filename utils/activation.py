from typing import Mapping

"""
Dictionary of custom, named activation functions. If an activation function
is requested in create_activation that is not present in the torch.nn namespace,
then create_activation will look for the module in this dictionary.
"""
custom_activations = {  # dict[str, Module]

}


def create_activation(config: Mapping):
    """
    Create an activation function from its configuration.

    :param config: Dictionary of model configuration options. Must contain a
        'name' field to specify which functions to load, which refers to the
        activation function's name within the torch.nn module, or else the name
        used as a key in the utils.activation.custom_activations dictionary of
        custom activation functions.
    :return: An instance of the activation function specified by the given
        configuration settings.
    """
    import torch

    config = dict(config)
    name = config.pop('name')

    module = getattr(torch.nn, name, None)
    if module is None:
        module = custom_activations[name]

    return module(**config)
