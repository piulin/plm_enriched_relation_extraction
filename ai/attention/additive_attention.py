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
from torch import Tensor
from torch.nn import Linear, Embedding

"""
additive_attention class: implementation of the additive attention layer by Adel and Strötgen (2021). See section 3.2.1
to learn more.
"""

import torch
import torch.nn as nn


# Based on the work by Adel and Strötgen (2021)
class additive_attention(nn.Module):

    def __init__(self,
                 hidden_state_size: int,
                 num_position_embeddings: int,
                 position_embedding_size: int,
                 local_size: int,
                 global_size: int,
                 attention_size: int):
        """
        Defines the layers the additive attention module constists of.
        :param hidden_state_size: hidden size of the PLM (H)
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size (P)
        :param local_size: embedding size of the local features (L)
        :param global_size: embedding size of the global features (G)
        :param attention_size: dimension of the internal attention space (A)
        """
        # init nn.Module
        super(additive_attention, self).__init__()

        # declare layers. For more details, please check out the paper by Adel and Strötgen (2021)
        self.v: Linear = nn.Linear(attention_size, 1)
        # TODO: bias?. I think not adding them could leave some expressivenes out of the equation
        self.W_h: Linear = nn.Linear(hidden_state_size, attention_size)
        self.W_q: Linear = nn.Linear(hidden_state_size, attention_size)
        self.W_s: Linear = nn.Linear(position_embedding_size, attention_size)
        self.W_o: Linear = nn.Linear(position_embedding_size, attention_size)
        self.W_l: Linear = nn.Linear(local_size, attention_size)
        self.W_g: Linear = nn.Linear(global_size, attention_size)

        # Position embeddings
        self.Ps: Embedding = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)
        self.Po: Embedding = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)


    def forward(self,
                h: Tensor,
                q: Tensor,
                ps: Tensor,
                po: Tensor,
                l: Tensor,
                g: Tensor
                ) -> Tensor:
        """
        Computes the attention score `e` as follows:
        e = v * tanh ( W_h*h + W_q*q + W_s*ps + W_o*po + W_l*l + W_g*g )
        :param h: hidden state of the PLM [batch_size, padded_sentence_length -2 , hidden_size]
        :param q: CLS token of the PLM (sentence representation) [batch_size, hidden_size]
        :param ps: position representation of the distance to entity 1  [batch_size, padded_sentence_length]
        :param po: position representation of the distance to entity 2  [batch_size, padded_sentence_length]
        :param l: local features [batch_size, padded_sentence_length -2, 2*dependency_distance_size+1]
        :param g: global features  g[batch_size, hidden_size]
        :return: attention scores `e` of shape [batch_size, padded_sentence_length -2, 1]
        """

        # retrieve embeddings representation of positions
        ps: Tensor = self.Ps(ps) # ps[batch_size, padded_sentence_length, position_embedding_size]
        po: Tensor = self.Po(po) # po[batch_size, padded_sentence_length, position_embedding_size]

        # map features into output space
        mh: Tensor = self.W_h(h) # mh[batch_size, padded_sentence_length, attention_size]
        mq: Tensor = self.W_q(q) # mh[batch_size, attention_size]
        ms: Tensor = self.W_s(ps) # ms[batch_size, padded_sentence_length, attention_size]
        mo: Tensor = self.W_o(po) # mo[batch_size, padded_sentence_length, attention_size]
        ml: Tensor = self.W_l(l) # ml[batch_size, padded_sentence_length, attention_size]
        mg: Tensor = self.W_g(g) # mg[batch_size, attention_size]

        # repeat global feature for each subtoken
        mg: Tensor = mg.unsqueeze(1).repeat(1, mh.shape[1], 1) # mg[batch_size, padded_sentence_length, attention_size]
        # same for the sentence representation
        mq: Tensor = mq.unsqueeze(1).repeat(1, mh.shape[1], 1) # mq[batch_size, padded_sentence_length, attention_size]

        # add non-linearity
        nl: Tensor = torch.tanh(mh+mq+ms+mo+ml+mg) # nl[batch_size, padded_sentence_length, attention_size]

        # compute attention score
        return self.v(nl)  #  [batch_size, padded_sentence_length -2, 1]




