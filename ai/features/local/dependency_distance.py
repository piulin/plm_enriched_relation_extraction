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

"""
dependency_distance module: models the representation of the distance to the two query entities in the dependency
parse tree. Read section 3.2.2, (i) Dependency distance on the work Adel and Strötgen (2021) to learn more.
"""

import torch.nn as nn
import torch

class dependency_distance(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_size):
        """
        Configures the module
        :param num_embeddings: number of distinct distances
        :param embedding_size: size of the internal embedding layer
        """

        # init nn.Module
        super(dependency_distance, self).__init__()

        # embedding for distances to entity 1
        self.de1 = nn.Embedding(num_embeddings, embedding_size, padding_idx=num_embeddings-1)

        # embedding for distances to entity 2
        self.de2 = nn.Embedding(num_embeddings, embedding_size, padding_idx=num_embeddings-1)

        self.embedding_size = embedding_size

    @property
    def output_size(self):
        """
        Retrieves the size of the produced local features
        :return:
        """
        return 2*self.embedding_size + 1


    def forward(self,
                de1,
                de2,
                f):
        """
        Computes the local features as the concatenation of the distance embeddings and the sdp flag.
        :param de1: distances to entity 1
        :param de2: distances to entity 2
        :param f: flag indicating wether tokens are in the SDP
        :return: dependency distance layer.
        """

        # shape (batch, sentence_length, dep_embedding)
        a = self.de1(de1)
        b = self.de2(de2)

        # convert shape (batch, sentence_length) into shape (batch, sentence_length, 1), to allow concatenation
        f_us = f.unsqueeze(2)

        # return local feature: concatenation of distance embeddings and flag
        return torch.cat ( (a, b, f_us), dim=2)