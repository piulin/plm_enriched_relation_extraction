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
cls_plm class: it implements the cls relation representation of Matching the Blanks: Distributional Similarity for
Relational Learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class cls_plm(nn.Module):

    def __init__(self,
                 number_of_relations,
                 plm_model_path='roberta-base'):
        """
        Sets up the network's plm and layers
        :param number_of_relations: Number of different relations in the labels
        :param vocabulary_length: the length of the vocabulary, i.e. the length of the tokenizer.
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(cls_plm, self).__init__()

        # Load the pretrained language model
        self.plm = RobertaModel.from_pretrained(plm_model_path)

        self.config = self.plm.config

        # Linear layer on top of the plm
        self.out = nn.Linear(self.config.hidden_size, number_of_relations)

        # Softmax classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                X):
        """
        Performs a forward pass.
        :param X: Batch to be passed.
        :param e1_indices: indices to locate E1S
        :param e2_indices: indices to locate E2S
        :return:
        """

        # Pass the data onto the pretrained language model
        X = self.plm(**X)

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        X = X.last_hidden_state[:,0,:]

        # Last linear layer
        X = self.out(X)

        # classification
        X = self.softmax(X)

        return X

