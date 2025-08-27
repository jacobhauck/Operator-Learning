"""
Utility functions for handling configuration objects and files.

A configuration is a dictionary with string keys, with string/int values,
lists of string/int values, or (sub-)configuration values.

Functions
---------
    - config_update_recursive(base_config, override_config, default_option)

"""
import copy
import json
import os
from typing import MutableMapping

import yaml


def config_update_recursive(
        base_config: MutableMapping,
        override_config: MutableMapping,
        default_option: str = None
) -> None:
    """
    Updates a configuration with override values from the given override
    configuration. Recursively handles sub-configurations; that is, if a
    sub-configuration is present in the base configuration, it will not be
    overridden wholesale by the corresponding override sub-configuration;
    instead, only sub-keys present in the override sub-configuration will
    override the base sub-configuration. This rule is also applied recursively.

    :param base_config: Config dict from which to initialize keys
    :param override_config: Config dict from which to add values that override
        the values from base_config
    :param default_option: Default method of handling keys in override_config
        that are not present in base_config. Must be one of None, 'add',
        'ignore', or 'cancel'. Default: None
    """
    if default_option not in ('add', 'ignore', 'cancel'):
        default_option = None

    for key, value in override_config.items():
        if key not in base_config:
            if default_option is None:
                print(f'Override config key {key} not in base config. What would you like to do?')
                print('Enter "add" to add the key')
                print('Enter "ignore" to continue without adding the key')
                print('Enter "cancel" to quit')
                option = None
                while option not in ('add', 'ignore', 'cancel'):
                    if option is not None:
                        print('Invalid option. Try again.')
                    option = input().lower()
            else:
                option = default_option

            if option == 'add':
                if default_option is None:
                    print(f'Adding override key {key}')
                base_config[key] = copy.deepcopy(override_config[key])
            elif option == 'ignore':
                if default_option is None:
                    print(f'Ignoring override key {key}')
                continue
            elif option == 'cancel':
                print('Exiting')
                exit()
        else:
            if isinstance(base_config[key], dict):
                if not isinstance(value, dict):
                    raise ValueError('Override config does not properly override base config.')
                config_update_recursive(base_config[key], value)
            else:
                base_config[key] = copy.deepcopy(value)


class ConfigNotFoundError(BaseException):
    pass


def load_config(path, raise_exc=False) -> dict:
    """
    Load a config file into a dict.

    :param path: Path to config file. Must be either JSON or YAML file. File
        type will be inferred if no extension is present.
    :param raise_exc: Whether to raise an exception if the requested file does
        not exist.
    :return: dict object containing configuration options from the file. Empty
        dict if the file does not exist, or an exception is thrown if raise_exc.
    """
    if os.path.splitext(path)[1].lower() not in ('.yaml', '.yml', '.json'):
        if os.path.exists(path + '.yaml'):
            path = path + '.yaml'
        elif os.path.exists(path + '.yml'):
            path = path + '.yml'
        elif os.path.exists(path + '.json'):
            path = path + '.json'
        else:
            if raise_exc:
                raise ConfigNotFoundError(f'Config {path} not found.')
            return {}

    ext = os.path.splitext(path)[1]
    if ext.lower() in ('.yaml', '.yml'):
        with open(path, 'r') as f:
            result = yaml.safe_load(f)
            return result if result is not None else {}
    elif ext.lower() == '.json':
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.decoder.JSONDecodeError as e:
                saved_exception = e

        with open(path, 'r') as f:
            if f.read() == '':
                return {}

        raise saved_exception
    else:
        if raise_exc:
            raise ConfigNotFoundError(f'Config {path} not found.')
        return {}


def load_base_config() -> dict:
    """
    Load the base config into a dict object.

    The base configuration file is either config.yaml or config.json.

    :return: dict object containing the base configuration options.
    """
    return load_config('config')
