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
from torch import Tensor
from torch.nn import Embedding

"""
dependency_distance module: models the representation of the distance to the two query entities in the dependency
parse tree. Read section 3.2.2, (i) Dependency distance on the work Adel and Strötgen (2021) to learn more.
"""

import torch.nn as nn
import torch

class dependency_distance(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int):
        """
        Configures the module
        :param num_embeddings: number of distinct distances
        :param embedding_size: size of the internal embedding layer
        """

        # init nn.Module
        super(dependency_distance, self).__init__()

        # embedding for distances to entity 1
        self.de1: Embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=num_embeddings-1)

        # embedding for distances to entity 2
        self.de2: Embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=num_embeddings-1)

        self.embedding_size: int = embedding_size

    @property
    def output_size(self) -> int:
        """
        Retrieves the size of the produced local features
        :return:
        """
        return 2*self.embedding_size + 1


    def forward(self,
                de1: Tensor,
                de2: Tensor,
                f: Tensor) -> Tensor:
        """
        Computes the local features as the concatenation of the distance embeddings and the sdp flag.
        :param de1: distances to entity 1 [batch_size, padded_sentence_length]
        :param de2: distances to entity 2 [batch_size, padded_sentence_length]
        :param f: flag indicating whether tokens are in the SDP [batch, sentence_length]
        :return: output of the network of shape [batch_size, padded_sentence_length -2, 2*embedding_size+1]
        """


        a = self.de1(de1) # a[batch, sentence_length, embedding_size]
        b = self.de2(de2) # b[batch, sentence_length, embedding_size]

        # convert shape (batch, sentence_length) into shape (batch, sentence_length, 1), to allow concatenation
        f_us = f.unsqueeze(2) # f[batch, sentence_length, 1]

        # return local feature: concatenation of distance embeddings and flag
        return torch.cat ( (a, b, f_us), dim=2) # [batch_size, padded_sentence_length -2, 2*embedding_size+1]