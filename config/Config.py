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
    'complexity_level': 'difficult',
    'verbose': True,
    'show_results': True,
    'path_to_save_results': None
}


def get_path_to_save_results(obj):

    temp_path = os.path.join(getattr(obj, 'complexity_level'.upper()), str(getattr(obj, 'no_of_agents'.upper())) + "_agents")
    temp_path = os.path.join("images", temp_path)
    
    return os.path.join(os.getcwd(), temp_path)


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

    if getattr(cls, 'path_to_save_results'.upper()) == None:
        setattr(cls, 'path_to_save_results'.upper(), get_path_to_save_results(cls))


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
