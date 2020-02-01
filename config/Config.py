#!usr/bin/env python

import os

from ruamel.yaml import YAML

"""
The configuration for Multi Agent Exploration SDK.

Implements the configuration mechanism for use in SDK. Contains default
values, Configuration Iniialization, and static configuration class which 
contains the configuration settings as an attribute.
"""

default_config = {
    'no_of_agents': 3,
    'sensor_range': 10,
    'grid_len': 612,
    'grid_width': 1024,
    'free_space': 100,
    'edge_cost': 40,
    'complexity_level': 'difficult'
}


def init_config(cls):
    """
    The initialization function of static config class.
    
    Initializes by getting parameters for config.yaml and sets the varialbles
    as attributes of static Config class(or default values in case the paramets
    are not avalible)

    Returns
    -------
    Config
    """

    yaml = YAML(typ='safe', pure=True)
    config_dict = None
    config_filepath = os.getenv('HOME') + '/multi-agent-exploration/config/config.yaml'
    
    with open (config_filepath, 'r') as config_file:
        config_dict = yaml.load(config_file)

    for name, value in default_config.items():
        try:
            setattr(cls, name.upper(), config_dict[name])
        except KeyError:
            setattr(cls, name.upper(), value)

    return cls


@init_config
class Config(object):
    """
    The Configuration class whose attributes reflects all the configurations.
    Attributes are set using init_config function above.
    
    Parameters
    ----------
    object : [type]
        [description]
    """
    
    pass
