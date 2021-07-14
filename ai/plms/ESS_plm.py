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
ESS_plm class: it implements the Entity Start State (ESS) relation of Matching the Blanks: Distributional Similarity for
Relational Learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class ess_plm(nn.Module):


    def __init__(self,
                 number_of_relations,
                 vocabulary_length,
                 plm_model_path='roberta-base'):
        """
        Sets up the network's plm and layers
        :param number_of_relations: Number of different relations in the labels
        :param vocabulary_length: the length of the vocabulary, i.e. the length of the tokenizer.
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(ess_plm, self).__init__()

        # Load the pretrained language model
        self.plm = RobertaModel.from_pretrained(plm_model_path)

        # update vocab length in order to accommodate new special tokens ( if added any )
        self.plm.resize_token_embeddings(vocabulary_length)

        self.config = self.plm.config


        # Linear layer on top of the plm (input size: concatenation of h_i and h_{j+2}, i.e. two hidden states)
        self.out = nn.Linear(self.config.hidden_size * 2, number_of_relations )

        # Softmax classification
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,
                X,
                e1_indices,
                e2_indices):
        """
        Performs a forward pass.
        :param X: Batch to be passed.
        :param e1_indices: indices to locate E1S
        :param e2_indices: indices to locate E2S
        :return:
        """

        # Pass the data onto the pretrained language model
        X = self.plm( ** X )

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        # X = X.last_hidden_state[:,0,:]

        x_indices = list(range(0, X.last_hidden_state.shape[0]))

        h_1 = X.last_hidden_state[x_indices,e1_indices,:]
        h_2 = X.last_hidden_state[x_indices,e2_indices,:]

        r_h = torch.cat((h_1,h_2),1)

        # Last linear layer
        X = self.out(r_h)

        # classification
        X = self.softmax(X)

        return X

