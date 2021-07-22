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
shortest_path module: models the representation of the SDP as globl features. Read section 3.2.2, (ii) Shortest Path
 on the work Adel and Strötgen (2021) to learn more.
"""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel

class shortest_path(nn.Module):

    def __init__(self,
                 plm_model=None,
                 plm_model_path='roberta-base'):
        """
        Sets up the network's plm and layers
        :param plm_model: If provided, then use that plm instead of a new instance
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(shortest_path, self).__init__()

        # if provided...
        if plm_model is not None:
            self.plm = plm_model

        else:
            # Load the pretrained language model
            self.plm = RobertaModel.from_pretrained(plm_model_path)

        self.config = self.plm.config

    def forward(self,
                X):
        """
        Performs a forward pass.
        :param X: Batch to be passed.
        :return:
        """

        # Pass the data onto the pretrained language model
        X = self.plm(**X)

        # Retrieve the representation of sentences ([CLS] tokens)
        X = X.last_hidden_state[:, 0, :]

        return X

    @property
    def output_size(self):
        """
        Retrieves the size of the global feature
        :return:
        """
        return self.config.hidden_size
