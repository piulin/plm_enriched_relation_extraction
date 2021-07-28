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
from typing import List, Iterator

from torch import Tensor
from torch.nn import Linear, LogSoftmax

"""
ESS_plm class: it implements the Entity Start State (ESS) relation of Matching the Blanks: Distributional Similarity for
Relational Learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, PretrainedConfig, BatchEncoding


class ess_plm(nn.Module):


    def __init__(self,
                 number_of_relations: int,
                 vocabulary_length: int,
                 plm_model_path: str = 'roberta-base',
                 **kwargs: dict):
        """
        Sets up the network's plm and layers
        :param number_of_relations: Number of different relations in the labels
        :param vocabulary_length: the length of the vocabulary, i.e. the length of the tokenizer.
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(ess_plm, self).__init__()

        # Load the pretrained language model
        self.plm: RobertaModel = RobertaModel.from_pretrained(plm_model_path)

        # update vocab length in order to accommodate new special tokens ( if added any )
        self.plm.resize_token_embeddings(vocabulary_length)

        self.config: PretrainedConfig = self.plm.config


        # Linear layer on top of the plm (input size: concatenation of h_i and h_{j+2}, i.e. two hidden states)
        self.out: Linear = nn.Linear(self.config.hidden_size * 2, number_of_relations )

        # Softmax classification
        self.softmax: LogSoftmax = nn.LogSoftmax(dim=1)


    def forward(self,
                X: BatchEncoding,
                e1_indices: List[int],
                e2_indices: List[int],
                **kwargs: dict) -> Tensor:
        """
        Performs a forward pass.
        :param X: PLM batch encoding to be passed onto the PLM.
        :param e1_indices: indices to locate E1S (of size `batch_size`)
        :param e2_indices: indices to locate E2S (of size `batch_size`)
        :return: tensor of shape [batch_size, n_classes]
        """

        # Pass the data onto the pretrained language model
        X = self.plm( ** X )

        # list from 0 to no_batches-1
        x_indices: List[int] = list(range(0, X.last_hidden_state.shape[0])) # size `batch_size`

        # extract ESS
        h_1: Tensor = X.last_hidden_state[x_indices,e1_indices,:] # h1[batch_size, hidden_size]
        h_2: Tensor = X.last_hidden_state[x_indices,e2_indices,:] # h2[batch_size, hidden_size]

        # concatenate them
        r_h: Tensor = torch.cat((h_1,h_2),1) # r_h[batch_size, 2*hidden_size]

        # Last linear layer
        X: Tensor = self.out(r_h) # X[batch_size, n_classes]

        # classification
        X = self.softmax(X) # X[batch_size, n_classes]

        return X

    @property
    def plm_parameters(self) -> Iterator:
        """
        Retrieves the PLM
        :return:
        """
        return self.plm.parameters()

    @property
    def post_plm_parameters(self) -> Iterator:
        """
        Retrieves the post transformer layers
        :return: list of layers
        """
        return self.out.parameters()
