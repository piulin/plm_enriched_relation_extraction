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
from ai.init.initializer import init_layer

"""
cls_plm class: it implements the cls relation representation of Matching the Blanks: Distributional Similarity for
Relational Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, PretrainedConfig, BatchEncoding
from collections import Iterable

from torch import Tensor
from torch.nn import Linear, LogSoftmax


class cls_plm(nn.Module):

    def __init__(self,
                 number_of_relations: int,
                 plm_model_path: str = 'roberta-base',
                 **kwargs):
        """
        Sets up the network's plm and layers
        :param number_of_relations: e.g. number of different relations in the labels
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(cls_plm, self).__init__()

        # Load the pretrained language model
        self.plm: RobertaModel = RobertaModel.from_pretrained(plm_model_path)

        self.config: PretrainedConfig = self.plm.config

        # Linear layer on top of the plm
        self.out: Linear =  init_layer( nn.Linear(self.config.hidden_size, number_of_relations), **kwargs)

        # Softmax classification
        self.softmax: LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,
                X: BatchEncoding,
                **kwargs: dict) -> Tensor:
        """
        Performs a forward pass.
        :param X: Batch to be passed.
        :return: output of the network with shape [batch_size, n_classes]
        """

        # Pass the data onto the pretrained language model
        X = self.plm(**X)

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        X: Tensor = X.last_hidden_state[:,0,:] # X[batch_size, hidden_size]

        # Last linear layer
        X: Tensor = self.out(X) # X[batch_size, n_classes]

        # classification
        X: Tensor = self.softmax(X) # X[batch_size, n_classes]

        return X

    @property
    def plm_parameters(self) -> Iterable:
        """
        Retrieves the PLM
        :return:
        """
        return self.plm.parameters()

    @property
    def post_plm_parameters(self) -> Iterable:
        """
        Retrieves the post transformer layers
        :return: list of layers
        """
        return self.out.parameters()

