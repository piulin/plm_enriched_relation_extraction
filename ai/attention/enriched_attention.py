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
import torch
# Based on the work by Adel and Strötgen (2021)
class enriched_attention(nn.Module):

    def __init__(self,
                 attention_function: str,
                 **kwargs: dict):
        """
        Inits the enriched attention module
        :param attention_function: specifies which attention function to use: `additive` or `dot_product`.
        :param kwargs: parameters to initialize the attention function.
        """

        # init nn.Module
        super(enriched_attention, self).__init__()

        # store arguments
        self.attention_function = attention_function

        # to get attention scores.
        self.attention: Union[additive_attention, MultiHeadAttention] = self.load_attention_layer(**kwargs)

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

        elif self.attention_function == 'dot_product':
            return MultiHeadAttention(**kwargs)


    def forward(self,
                **kwargs: dict,
                ) -> Tensor:
        """
        Computes the final representation of the sequence based on the attention weights of the attention function.
        :param kwargs: parameters to forward to the attention function
        :return: representation of sequence  [batch_size, hidden_size]
        """

        # switch attention function
        if self.attention_function == 'additive':

            return self.attention(**kwargs) # e[batch_size, hidden_size]


        elif self.attention_function == 'dot_product':

            _, o =  self.attention(**kwargs)

            return o  # e[batch_size, hidden_size]





