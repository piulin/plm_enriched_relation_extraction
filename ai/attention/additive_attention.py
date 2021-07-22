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
additive_attention class: implementation of the additive attention layer by Adel and Strötgen (2021). See section 3.2.1
to learn more.
"""

import torch
import torch.nn as nn


# Based on the work by Adel and Strötgen (2021)
class additive_attention(nn.Module):

    def __init__(self,
                 hidden_state_size,
                 num_position_embeddings,
                 position_embedding_size,
                 local_size,
                 global_size,
                 attention_size):
        """
        Defines the layers the additive attention module constists of.
        :param hidden_state_size: hidden size of the PLM
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size
        :param local_size: embedding size of the local features
        :param global_size: embedding size of the global features
        :param attention_size: dimension of the internal attention space (A)
        """
        # init nn.Module
        super(additive_attention, self).__init__()

        # declare layers. For more details, please check out the paper by Adel and Strötgen (2021)
        self.v = nn.Linear(attention_size, 1)
        # TODO: bias?. I think not adding them could leave some expressivenes out of the equation
        self.W_h = nn.Linear(hidden_state_size, attention_size)
        self.W_q = nn.Linear(hidden_state_size, attention_size)
        self.W_s = nn.Linear(position_embedding_size, attention_size)
        self.W_o = nn.Linear(position_embedding_size, attention_size)
        self.W_l = nn.Linear(local_size, attention_size)
        self.W_g = nn.Linear(global_size, attention_size)

        # Position embeddings
        self.Ps = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)
        self.Po = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)


    def forward(self,
                h,
                q,
                ps,
                po,
                l,
                g):
        """
        Computes the attention score `e` as follows:
        e = v * tanh ( W_h*h + W_q*q + W_s*ps + W_o*po + W_l*l + W_g*g )
        :param h: hidden state of the PLM
        :param q: CLS token of the PLM (sentence representation)
        :param ps: position representation of the distance to entity 1
        :param po: position representation of the distance to entity 2
        :param l: local features
        :param g: global featuers
        :return:
        """

        # retrieve embeddings representation of positions
        ps = self.Ps(ps)
        po = self.Po(po)

        # map features into output space
        mh = self.W_h(h)
        mq = self.W_q(q)
        ms = self.W_s(ps)
        mo = self.W_o(po)
        ml = self.W_l(l)
        mg = self.W_g(g)

        # repeat global feature for each subtoken
        mg = mg.unsqueeze(1).repeat(1, mh.shape[1], 1)
        # same for the sentence representation
        mq = mq.unsqueeze(1).repeat(1, mh.shape[1], 1)

        # add non-linearity
        nl = torch.tanh(mh+mq+ms+mo+ml+mg)

        # compute attention score
        return self.v(nl)




