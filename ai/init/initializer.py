"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM
    
    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),
    
    with the supervision of
    
    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""
from typing import Union

from torch.nn import Linear, Embedding

"""
initializer module: takes care of initializing the layers of the network 
"""
from torch import nn

def init_layer( layer: Union[Linear, Embedding],
               init_method: str,
                **kwargs: dict):
    """
    Replaces the content of `layer` following the `init_method`
    :param init_method: sets the init method to use. Select from: "none", "xavier_uniform",
    "xavier_normal", "kaiming_uniform_fan_in", "kaiming_uniform_fan_out", "kaiming_normal_fan_in" or "kaiming_normal_fan_in"
    :param layer: layer to be initialized
    :return: layer after initialization
    """


    # Switch the init method
    if init_method == 'xavier_uniform':
        nn.init.xavier_uniform_( layer.weight )

    elif init_method == 'xavier_normal':
        nn.init.xavier_normal_( layer.weight )

    elif init_method == 'kaiming_uniform_fan_in':
        nn.init.kaiming_uniform_( layer.weight, mode='fan_in' )

    elif init_method == 'kaiming_uniform_fan_out':
        nn.init.kaiming_uniform_( layer.weight, mode='fan_out' )

    elif init_method == 'kaiming_normal_fan_in':
        nn.init.kaiming_normal_( layer.weight, mode='fan_in')

    elif init_method == 'kaiming_normal_fan_out':
        nn.init.kaiming_normal_( layer.weight, mode='fan_out')

    return layer