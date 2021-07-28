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
from collections import Iterable

from torch import Tensor
from torch.nn import Linear, LogSoftmax

"""
enriched_attention_plm class: it implements enriched attention as described in the work by Adel and Strötgen (2021).
Read section 3.2.1 to learn more.
"""
import torch.nn as nn
from transformers import RobertaModel, PretrainedConfig, BatchEncoding
from ai.attention.enriched_attention import enriched_attention
from ai.features.local.dependency_distance import dependency_distance
from ai.features.globl.shortest_path import shortest_path
import torch

class enriched_attention_plm(nn.Module):

    def __init__(self,
                 number_of_relations: int,
                 num_dependency_distance_embeddings: int,
                 dependency_distance_size: int,
                 plm_model_path: str = 'roberta-base',
                 **kwargs: dict):
        """
        Sets up the network's plm and layers
        :param number_of_relations: e.g. number of different relations in the labels
        :param num_dependency_distance_embeddings: number of different dependency distance embeddings
        :param dependency_distance_size: size of the dependency distance embeddings
        :param plm_model_path: path to the pretrained language model
        :param kwargs: parameters to initialize the attention function.
        """

        # Set up the nn module
        super(enriched_attention_plm, self).__init__()

        # Load the pretrained language model
        self.plm: RobertaModel = RobertaModel.from_pretrained(plm_model_path)

        self.config: PretrainedConfig  = self.plm.config

        # define globl and local features
        self.local: dependency_distance = dependency_distance(num_dependency_distance_embeddings,
                                         dependency_distance_size)

        self.globl: shortest_path = shortest_path(plm_model=self.plm)


        # define attention layer
        self.attention: enriched_attention = enriched_attention(hidden_state_size=self.config.hidden_size,
                                            local_size=self.local.output_size,
                                            global_size=self.globl.output_size,
                                            **kwargs)

        # Linear layer on top of the attention layer
        self.out: Linear = nn.Linear(self.config.hidden_size, number_of_relations)

        # Softmax classification
        self.softmax: LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,
                X: BatchEncoding,
                de1: Tensor,
                de2: Tensor,
                f: Tensor,
                sdp: BatchEncoding,
                **kwargs: dict) -> Tensor:
        """
        Performs a forward pass.
        :param X: RoBERTa subtoken split (from Roberta tokenizer)
        :param de1: distance of subtokens with respect to entity 1 in the dependency parse (SDP) [batch_size, padded_sentence_length]
        :param de2: distance of subtokens with respect to entity 2 in the dependency parse (SDP) [batch_size, padded_sentence_length]
        :param f: SDP flag  [batch_size, padded_sentence_length]
        :param sdp: shortest dependency path
        :param kwargs: parameters to forward to the attention function
        :return: output of the network of shape[batch_size, n_classes]
        """

        # Pass the data onto the pretrained language model
        X = self.plm(**X)

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        q: Tensor = X.last_hidden_state[:,0,:] # q[batch_size, hidden_size]

        # Retrieve hidden states (discard CLS and SEP tokens)
        h: Tensor = X.last_hidden_state[:,1:-1,:] # h[batch_size, padded_sentence_length -2 , hidden_size]

        # retrieve local features
        l: Tensor = self.local(de1, de2, f) # l[batch_size, padded_sentence_length -2, 2*dependency_distance_size+1]

        # retrieve global features
        g: Tensor = self.globl(sdp) # g[batch_size, hidden_size]

        # compute attention weights
        alpha: Tensor = self.attention(h=h, q=q, l=l, g=g, **kwargs) # alpha[batch_size,  padded_sentence_length -2]

        # transform alpha shape for element-wise multiplication with hidden states
        alpha: Tensor = alpha.unsqueeze(2).repeat(1, 1, h.shape[2]) # alpha[batch_size,  padded_sentence_length -2, hidden_size]

        # compute contributions of contextual embeddings given attention weights
        o: Tensor = torch.sum(h*alpha, dim=1 ) # o[batch_size, hidden_size]

        # Last linear layer
        o: Tensor = self.out(o) # o[batch_size, n_classes]

        # classification
        o: Tensor = self.softmax(o) # o[batch_size, n_classes]

        return o

    @property
    def plm_parameters(self) -> Iterable:
        """
        Retrieves the PLM
        :return:
        """
        return self.parameters()

    @property
    def post_plm_parameters(self) -> Iterable:
        """
        Retrieves the post transformer layers
        :return: list of layers
        """
        return []

