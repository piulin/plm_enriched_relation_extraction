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
from typing import Union

"""
enriched_attention class: wrapper for multiple attention schemas
"""

from torch import Tensor
from torch.nn import Softmax
import torch.nn as nn
from ai.attention.additive_attention import additive_attention
from ai.attention.dot_product_attention import MultiHeadAttention

# Based on the work by Adel and Strötgen (2021)
class enriched_attention(nn.Module):

    def __init__(self,
                 attention_function: str,
                 **kwargs: dict):
        """
        Inits the enriched attention module
        :param attention_function: specifies which attention function to use: `additive` or `dot-product`.
        :param kwargs: parameters to initialize the attention function.
        """

        # init nn.Module
        super(enriched_attention, self).__init__()

        # store arguments
        self.attention_function = attention_function

        # to get attention scores.
        self.attention: Union[additive_attention, MultiHeadAttention] = self.load_attention_layer(**kwargs)

        # to transform attention scores into attention weights
        self.softmax: Softmax = nn.Softmax(dim=1)

    def load_attention_layer(self,
                             **kwargs: dict,
                             ) -> Union[additive_attention, MultiHeadAttention]:
        """
        Initializes the attention layer
        :param kwargs: parameters to init the attention layer
        :return: attention layer
        """

        # switch attention function to initialize
        if self.attention_function == 'additive':
            return additive_attention(**kwargs)

        elif self.attention_function == 'dot-product':
            return MultiHeadAttention(**kwargs)


    def forward(self,
                **kwargs: dict,
                ) -> Tensor:
        """
        Computes attention weights from attention scores via softmax.
        :param kwargs: parameters to forward to the attention function
        :return: attention weights of shape [batch_size, padded_sentence_length -2]
        """

        # get attention scores
        e: Tensor = self.attention(**kwargs) # e[batch_size, padded_sentence_length -2]

        # get attention weights
        logits: Tensor = self.softmax(e)  # logits[batch_size, padded_sentence_length -2]

        return logits



