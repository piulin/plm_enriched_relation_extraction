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
enriched_attention_plm class: it implements enriched attention as described in the work by Adel and Strötgen (2021).
Read section 3.2.1 to learn more.
"""
import torch.nn as nn
from transformers import RobertaModel
from ai.attention.enriched_attention import enriched_attention
from ai.features.local.dependency_distance import dependency_distance
from ai.features.globl.shortest_path import shortest_path
import torch

class enriched_attention_plm(nn.Module):

    def __init__(self,
                 no_output_neurons,
                 num_position_embeddings,
                 position_embedding_size,
                 num_dependency_distance_embeddings,
                 dependency_distance_size,
                 attention_size,
                 plm_model_path='roberta-base'):
        """
        Sets up the network's plm and layers
        :param no_output_neurons: e.g. number of different relations in the labels
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size
        :param num_dependency_distance_embeddings: number of different dependency distance embeddings
        :param dependency_distance_size: size of the dependency distance embeddings
        :param attention_size: dimension of the internal attention space (A)
        :param plm_model_path: path to the pretrained language model
        """

        # Set up the nn module
        super(enriched_attention_plm, self).__init__()

        # Load the pretrained language model
        self.plm = RobertaModel.from_pretrained(plm_model_path)

        self.config = self.plm.config

        # define globl and local features
        self.local = dependency_distance(num_dependency_distance_embeddings,
                                         dependency_distance_size)

        self.globl = shortest_path(plm_model=self.plm)


        # define attention layer
        self.attention = enriched_attention(self.config.hidden_size,
                                            num_position_embeddings,
                                            position_embedding_size,
                                            self.local.output_size,
                                            self.globl.output_size,
                                            attention_size)

        # Linear layer on top of the attention layer
        self.out = nn.Linear(self.config.hidden_size, no_output_neurons)

        # Softmax classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                X,
                ps,
                po,
                de1,
                de2,
                f,
                sdp):
        """
        Performs a forward pass.
        :param X: RoBERTa subtoken split (from Roberta tokenizer)
        :param ps: distance of subtokens with respect to entity 1
        :param po: distance of subtokens with respect to entity 2
        :param de1: distance of subtokens with respect to entity 1 in the dependency parse (SDP)
        :param de2: distance of subtokens with respect to entity 2 in the dependency parse (SDP)
        :param f: SDP flag
        :param sdp: shortest dependency path
        :return:
        """

        # Pass the data onto the pretrained language model
        X = self.plm(**X)

        # Retrieve the representation of sentences ([CLS] tokens)
        # outputs.last_hidden_state shape(batch_size, sequence_length, hidden_size). Sequence length also accounts
        # for the padding introduced in the tokenization process
        q = X.last_hidden_state[:,0,:]

        # Retrieve hidden states (discard CLS and SEP tokens)
        h = X.last_hidden_state[:,1:-1,:]

        # retrieve local features
        l = self.local(de1, de2, f)

        # retrieve global features
        g = self.globl(sdp)


        # compute attention weigths
        alpha = self.attention(h, q, ps, po, l, g)

        # transform alpha shape for element-wise multiplication with hidden states
        alpha = alpha.unsqueeze(2).repeat(1, 1, h.shape[2])

        # compute contributions of contextual embeddings given attention weights
        o = torch.sum(h*alpha, dim=1 )

        # Last linear layer
        o = self.out(o)

        # classification
        o = self.softmax(o)

        return o

    @property
    def plm_parameters(self):
        """
        Retrieves the PLM
        :return:
        """
        return self.parameters()

    @property
    def post_plm_parameters(self):
        """
        Retrieves the post transformer layers
        :return: list of layers
        """
        return []

