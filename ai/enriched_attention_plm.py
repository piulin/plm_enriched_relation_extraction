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
enriched_attention_transformers class: it is responsible of defining the top layers after the PLM, as well as 
inlcuding the enriched attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class enriched_attention_transformers(nn.Module):


    def __init__(self,
                 number_of_relations,
                 plm_model_path='roberta-base'):
        """
        Sets up the network's plm and layers
        :param number_of_relations: Number of different relations in the labels
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(enriched_attention_transformers, self).__init__()

        # Load the pretrained language model
        self.plm = RobertaModel.from_pretrained(plm_model_path)

        self.config = self.plm.config

        # Linear layer on top of the plm
        self.out = nn.Linear(self.config.hidden_size, number_of_relations )

        # Softmax classification
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, X):
        """
        Performs a forward pass.
        :param X: Batch to be passed.
        :return:
        """

        # Pass the data onto the pretrained language model
        X = self.plm( ** X )

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        X = X.last_hidden_state[:,0,:]

        # Last linear layer
        X = self.out(X)

        # classification
        X = self.softmax(X)

        return X

