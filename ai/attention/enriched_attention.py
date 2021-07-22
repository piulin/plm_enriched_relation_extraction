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
enriched_attention class: wrapper for multiple attention schemas
"""

import torch.nn as nn
from ai.attention.additive_attention import additive_attention

# Based on the work by Adel and Strötgen (2021)
class enriched_attention(nn.Module):

    def __init__(self,
                 hidden_state_size,
                 num_position_embeddings,
                 position_embedding_size,
                 local_size,
                 global_size,
                 attention_size):
        """

        :param hidden_state_size: hidden size of the PLM
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size
        :param local_size: embedding size of the local features
        :param global_size: embedding size of the global features
        :param attention_size: dimension of the internal attention space (A)
        """

        # init nn.Module
        super(enriched_attention, self).__init__()

        # to get attention scores.
        # TODO: add positibility of annother attention scores (product-attention)
        self.attention = additive_attention(hidden_state_size,
                                            num_position_embeddings,
                                            position_embedding_size,
                                            local_size,
                                            global_size,
                                            attention_size)

        # to transform attention scores into attention weights
        self.softmax = nn.Softmax(dim=1)


    def forward(self,
                h,
                q,
                ps,
                po,
                l,
                g
                ):
        """
        Computes attention weights from attention scores via softmax.
        :param h: hidden state of the PLM
        :param q: CLS token of the PLM (sentence representation)
        :param ps: position representation of the distance to entity 1
        :param po: position representation of the distance to entity 2
        :param l: local features
        :param g: global featuers
        :return: attention weights
        """


        # get attention scores
        e = self.attention(h, q, ps, po, l, g)

        # remove last dimension
        e = e.squeeze(2)

        # get attention weights
        logits = self.softmax(e)

        return logits



