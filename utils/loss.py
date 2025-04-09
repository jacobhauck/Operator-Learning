from typing import Mapping

"""
Dictionary of custom, named loss functions. If a loss function is requested in
create_loss that is not present in the torch.nn namespace, then create_loss will
look for the module in this dictionary.
"""
custom_losses = {  # dict[str, Module]

}


def create_loss(config: Mapping):
    """
    Create a loss function from its configuration.

    :param config: Dictionary of model configuration options. Must contain a
        'name' field to specify which function to load, which refers to the
        activation function's name within the torch.nn module, or else the name
        used as a key in the utils.loss.custom_losses dictionary of
        custom loss functions.
    :return: An instance of the loss function specified by the given
        configuration settings.
    """
    import torch

    config = dict(config)
    name = config.pop('name')

    module = getattr(torch.nn, name, None)
    if module is None:
        module = custom_losses[name]

    return module(**config)
