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

from torch import Tensor
from torch.nn import Dropout, BatchNorm1d

"""
shortest_path module: models the representation of the SDP as globl features. Check out section 3.2.2, (ii) Shortest Path
 in the work by Adel and Strötgen (2021) to learn more.
"""

import torch.nn as nn
from transformers import RobertaModel, PretrainedConfig, BatchEncoding


class shortest_path(nn.Module):

    def __init__(self,
                 dropout_probability: float,
                 plm_model: Union[RobertaModel, None] = None,
                 plm_model_path: Union[str, None] = 'roberta-base',
                 **kwargs: dict):
        """
        Sets up the network's plm and layers
        :param dropout_probability: p value for dropout layers
        :param plm_model: If provided, then use that plm instead of a new instance
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(shortest_path, self).__init__()

        # if provided...
        if plm_model is not None:
            self.plm: RobertaModel = plm_model

        else:
            # Load the pretrained language model
            self.plm: RobertaModel = RobertaModel.from_pretrained(plm_model_path)

        self.norm: BatchNorm1d = nn.BatchNorm1d(self.plm.config.hidden_size)

        # regularization layer
        self.dropout: Dropout = nn.Dropout(p=dropout_probability)

        self.config: PretrainedConfig = self.plm.config

    def forward(self,
                sdp: BatchEncoding,
                **kwargs: dict) -> Tensor:
        """
        Performs a forward pass.
        :param sdp: shortest dependency path
        :return: output of the network of shape [batch_size, hidden_size]
        """

        # Pass the data onto the pretrained language model
        sdp = self.plm(**sdp)

        # Retrieve the representation of sentences ([CLS] tokens)
        X: Tensor = sdp.last_hidden_state[:, 0, :] # X[batch_size, hidden_size]

        # batch normalization
        # X: Tensor = self.norm( X ) # X[batch_size, hidden_size]

        # dropout reg.
        X: Tensor = self.dropout( X ) # X[batch_size, hidden_size]


        return X

    @property
    def output_size(self) -> int:
        """
        Retrieves the size of the global feature
        :return:
        """
        return self.config.hidden_size
