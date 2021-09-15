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
from torch.nn import Linear, Embedding, Softmax

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
                 attention_size: int,
                 dropout_probability: float,
                 **kwargs: dict):
        """
        Defines the layers the additive attention module consists of.
        :param hidden_state_size: hidden size of the PLM (H)
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size (P)
        :param local_size: embedding size of the local features (L)
        :param global_size: embedding size of the global features (G)
        :param attention_size: dimension of the internal attention space (A)
        :param dropout_probability: p value for dropout layers
        """
        # init nn.Module
        super(additive_attention, self).__init__()

        # declare layers. For more details, please check out the paper by Adel and Strötgen (2021)
        self.v: Linear = nn.Linear(attention_size, 1, bias=False)
        # TODO: bias?. I think not adding them could leave some expressiveness out of the equation
        self.W_h: Linear = nn.Linear(hidden_state_size, attention_size, bias=False)
        self.W_q: Linear = nn.Linear(hidden_state_size, attention_size, bias=False)
        self.W_s: Linear = nn.Linear(position_embedding_size, attention_size, bias=False)
        self.W_o: Linear = nn.Linear(position_embedding_size, attention_size, bias=False)
        self.W_l: Linear = nn.Linear(local_size, attention_size, bias=False)
        self.W_g: Linear = nn.Linear(global_size, attention_size, bias=False)

        # Position embeddings
        self.Ps: Embedding = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)
        self.Po: Embedding = nn.Embedding(num_position_embeddings, position_embedding_size, padding_idx=num_position_embeddings-1)

        # to transform attention scores into attention weights
        self.softmax: Softmax = nn.Softmax(dim=1)

        # regularization
        self.dropout_ps = nn.Dropout( p=dropout_probability )
        self.dropout_po = nn.Dropout( p=dropout_probability )
        self.dropout_nl = nn.Dropout( p=dropout_probability )
        self.dropout_mh = nn.Dropout( p=dropout_probability )
        self.dropout_mq = nn.Dropout( p=dropout_probability )
        self.dropout_ms = nn.Dropout( p=dropout_probability )
        self.dropout_mo = nn.Dropout( p=dropout_probability )
        self.dropout_ml = nn.Dropout( p=dropout_probability )
        self.dropout_mg = nn.Dropout( p=dropout_probability )
        self.dropout_nl = nn.Dropout( p=dropout_probability )

        # self.norm = nn.BatchNorm1d( hidden_state_size )



    def forward(self,
                h: Tensor,
                q: Tensor,
                ps: Tensor,
                po: Tensor,
                l: Tensor,
                g: Tensor,
                mask: Tensor,
                **kwargs: dict
                ) -> Tensor:
        """
        Computes the attention scores `e` as follows:
        e = v * tanh ( W_h*h + W_q*q + W_s*ps + W_o*po + W_l*l + W_g*g ),
        and retrieves the final representation of the sequence as a weighted sum of the sequence states with the
        attention weights
        :param h: hidden state of the PLM [batch_size, padded_sentence_length -2 , hidden_size]
        :param q: CLS token of the PLM (sentence representation) [batch_size, hidden_size]
        :param ps: position representation of the distance to entity 1  [batch_size, padded_sentence_length]
        :param po: position representation of the distance to entity 2  [batch_size, padded_sentence_length]
        :param l: local features [batch_size, padded_sentence_length -2, 2*dependency_distance_size+1]
        :param g: global features  g[batch_size, hidden_size]
        :return: representation of the sentence [batch_size, hidden_size]
        """

        # # TODO: remove debug code.
        # # @@ DEBUG @@
        # alpha: float = 1/h.shape[1]
        # return torch.sum(h * alpha, dim=1)
        # #####

        # retrieve embeddings representation of positions
        pse: Tensor = self.dropout_ps( self.Ps(ps) ) # ps[batch_size, padded_sentence_length, position_embedding_size]
        poe: Tensor = self.dropout_po( self.Po(po) ) # po[batch_size, padded_sentence_length, position_embedding_size]

        # map features into attention space
        mh: Tensor = self.dropout_mh( self.W_h(h) ) # mh[batch_size, padded_sentence_length, attention_size]
        mq: Tensor = self.dropout_mq( self.W_q(q) ) # mh[batch_size, attention_size]
        ms: Tensor = self.dropout_ms( self.W_s(pse) ) # ms[batch_size, padded_sentence_length, attention_size]
        mo: Tensor = self.dropout_mo( self.W_o(poe) ) # mo[batch_size, padded_sentence_length, attention_size]
        ml: Tensor = self.dropout_ml( self.W_l(l) ) # ml[batch_size, padded_sentence_length, attention_size]
        mg: Tensor = self.dropout_mg( self.W_g(g) ) # mg[batch_size, attention_size]

        # repeat global feature for each subtoken.
        mg: Tensor = mg.unsqueeze(1) # mg[batch_size, 1, attention_size]
        # same for the sentence representation
        mq: Tensor = mq.unsqueeze(1) # mq[batch_size, 1, attention_size]

        # add non-linearity
        nl: Tensor = self.dropout_nl( torch.tanh(mh + mq + ms + mo + ml + mg) ) # nl[batch_size, padded_sentence_length, attention_size]
        # nl: Tensor = self.dropout_nl( torch.tanh(ms + mo + mq + ml) ) # nl[batch_size, padded_sentence_length, attention_size]

        # compute attention score
        logits: Tensor = self.v(nl)  #  logits[batch_size, padded_sentence_length -2, 1]

        # remove last dimension
        logits: Tensor = logits.squeeze(2)


        #  use `mask` to mask out padded tokens.
        logits: Tensor = logits.masked_fill(mask, -1e9)

        # get attention weights
        alpha: Tensor = self.softmax(logits)  # alpha[batch_size, padded_sentence_length -2]

        # transform alpha shape for element-wise multiplication with hidden states.
        alpha: Tensor = alpha.unsqueeze(2)  # alpha[batch_size,  padded_sentence_length -2, 1]

        # compute contributions of contextual embeddings given attention weights
        o: Tensor = torch.sum(h * alpha, dim=1)  # o[batch_size, hidden_size]

        return o # self.norm( o )




