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

from ai.features.globl.entity_types import entity_types
from ai.features.globl.shortest_path import shortest_path

"""
global_features class: wrapper for multiple global features
"""

from torch import Tensor
import torch.nn as nn

# Based on the work by Adel and Strötgen (2021)
class global_features(nn.Module):

    def __init__(self,
                 global_feature: str,
                 **kwargs: dict):
        """
        Inits the enriched attention module
        :param global_feature: specifies which attention global feature to use: `shortest_path` or `entity_types`.
        :param kwargs: parameters to initialize the global feature.
        """

        # init nn.Module
        super(global_features, self).__init__()

        # store arguments
        self.global_feature = global_feature

        # to get attention scores.
        self.feature: Union[entity_types, shortest_path] = self.load_global_feature(**kwargs)

    def load_global_feature(self,
                             **kwargs: dict,
                             ) -> Union[entity_types, shortest_path]:
        """
        Initializes the global feature
        :param kwargs: parameters to initialize the global feature.
        :return: global feature
        """

        # switch attention function to initialize
        if self.global_feature == 'shortest_path':
            return shortest_path(**kwargs)

        elif self.global_feature == 'entity_types':
            return entity_types(**kwargs)


    def forward(self,
                **kwargs: dict,
                ) -> Tensor:
        """
        Computes the representation of a sequence according to the global feature selected during initialization
        :param kwargs: parameters to forward to the global feature
        :return: representation of sequence  [batch_size, global_feature_size]
        """

        return self.feature(**kwargs) # e[batch_size, global_feature_size]

    @property
    def output_size(self) -> int:
        """
        Retrieves the size of the global feature
        :return:
        """
        return self.feature.output_size




