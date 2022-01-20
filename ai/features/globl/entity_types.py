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

import torch
from torch import Tensor
from torch.nn import Dropout, BatchNorm1d

from ai.init.initializer import init_layer

"""
entity_types module: models the representation of the entity types as a global feature by embeddings.
 Check out section 3.2.3, (iii) Entity types in the work by Adel and Strötgen (2021) to learn more.
"""

import torch.nn as nn
from transformers import RobertaModel, PretrainedConfig, BatchEncoding

class entity_types(nn.Module):

    def __init__(self,
                 dropout_probability: float,
                 num_entity_embeddings: int,
                 entity_embedding_size: int,
                 **kwargs: dict):
        """
        Sets up the network's plm and layers
        :param dropout_probability: p value for dropout layers
        :param num_entity_embeddings: size of the entity type lookup tables.
        """

        # Set up the nn module
        super(entity_types, self).__init__()
        self.entity_embedding_size = entity_embedding_size

        # embedding layers
        self.subj_embedding = init_layer( nn.Embedding(num_entity_embeddings, entity_embedding_size), **kwargs)
        self.obj_embedding = init_layer( nn.Embedding(num_entity_embeddings, entity_embedding_size), **kwargs)

        # regularization layer
        self.dropout_subj: Dropout = nn.Dropout(p=dropout_probability)
        self.dropout_obj: Dropout = nn.Dropout(p=dropout_probability)

    def forward(self,
                entity_types: Tensor,
                **kwargs: dict) -> Tensor:
        """
         Retrieves the flobal features
        :param entity_types: entity types [batch_size, 2]
        :return: global features of shape [batch_size, 2*entity_embedding_size]
        """

        # embeddings for entity type 1
        e1 = self.dropout_subj( self.subj_embedding(entity_types[:, 0]) )

        # embeddings for entity type 1
        e2 = self.dropout_obj( self.obj_embedding(entity_types[:, 1]) )

        return torch.cat((e1,e2), dim=1)


    @property
    def output_size(self) -> int:
        """
        Retrieves the size of the global feature
        :return:
        """
        return 2*self.entity_embedding_size
