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

from torch import Tensor
from torch.nn import Softmax
import torch.nn as nn
from ai.attention.additive_attention import additive_attention

# Based on the work by Adel and Strötgen (2021)
class enriched_attention(nn.Module):

    def __init__(self,
                 hidden_state_size: int,
                 num_position_embeddings: int,
                 position_embedding_size: int,
                 local_size: int,
                 global_size: int,
                 attention_size: int):
        """
        Inits the enriched attention module
        :param hidden_state_size: hidden size of the PLM (H)
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size (P)
        :param local_size: embedding size of the local features (L)
        :param global_size: embedding size of the global features (G)
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
        self.softmax: Softmax = nn.Softmax(dim=1)


    def forward(self,
                h: Tensor,
                q: Tensor,
                ps: Tensor,
                po: Tensor,
                l: Tensor,
                g: Tensor
                ) -> Tensor:
        """
        Computes attention weights from attention scores via softmax.
        :param h: hidden state of the PLM [batch_size, padded_sentence_length -2 , hidden_size]
        :param q: CLS token of the PLM (sentence representation) [batch_size, hidden_size]
        :param ps: position representation of the distance to entity 1  [batch_size, padded_sentence_length]
        :param po: position representation of the distance to entity 2  [batch_size, padded_sentence_length]
        :param l: local features [batch_size, padded_sentence_length -2, 2*dependency_distance_size+1]
        :param g: global features  g[batch_size, hidden_size]
        :return: attention weights of shape [batch_size, padded_sentence_length -2]
        """


        # get attention scores
        e: Tensor = self.attention(h, q, ps, po, l, g) # e[batch_size, padded_sentence_length -2, 1]

        # remove last dimension
        e: Tensor = e.squeeze(2) # e[batch_size, padded_sentence_length -2]

        # get attention weights
        logits: Tensor = self.softmax(e)  # logits[batch_size, padded_sentence_length -2]

        return logits



