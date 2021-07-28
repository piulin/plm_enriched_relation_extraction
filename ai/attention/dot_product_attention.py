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

import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import init
import torch.nn.functional as F

"""
dot_product_attention class: implementation of the dot-product attention layer by Adel and Strötgen (2021). 
Check out section 3.2.1 to learn more.
"""

# code from https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
# Modifications by:
# -- Dr. Heike Adel-Vu
# -- Pedro G. Bascoy
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        #print("local", local_f.size())
        #print("global", global_f.size())
        # query, key: batch_size * self.head_num x seq_len x sub_dim
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk) # [attention_size, seq_len, seq_len]
        # scores: batch_size * self.head_num x seq_len x seq_len
        if mask is not None:
            head_num, batchsize, seq_len = mask.shape
            masked_reshaped = mask.view(batchsize*head_num, seq_len) # [batch_size*head_num, seq_len]
            masked_reshaped = masked_reshaped.unsqueeze(1) # [batch_size*head_num, 1, seq_len]
            scores = scores.masked_fill(masked_reshaped == 1, -1e9)
        attention = F.softmax(scores, dim=-1)
        result = attention.matmul(value) # [attention_size, seq_len, batch_size]
        return result, scores, attention


class ScaledDotProductAttentionScores(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        #print("local", local_f.size())
        #print("global", global_f.size())
        # query, key: batch_size * self.head_num x seq_len x sub_dim
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk) # [attention_size, seq_len, seq_len]
        # scores: batch_size * self.head_num x seq_len x seq_len

        scores = torch.diagonal( query.matmul(key.transpose(1, 2)), dim1=1,dim2=2) / math.sqrt(dk)
        if mask is not None:
            # head_num, batchsize, seq_len = mask.shape
            # masked_reshaped = mask.view(batchsize*head_num, seq_len) # [batch_size*head_num, seq_len]
            # masked_reshaped = masked_reshaped.unsqueeze(1) # [batch_size*head_num, 1, seq_len]
            scores = scores.masked_fill(mask == 1, -1e9)
        # attention = F.softmax(scores, dim=-1)
        # result = attention.matmul(value) # [attention_size, seq_len, batch_size]
        return scores


# code from https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
# Modifications by:
# -- Dr. Heike Adel-Vu
# -- Pedro G. Bascoy
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 hidden_state_size,
                 head_number,
                 local_size,
                 global_size,
                 attention_size,
                 bias=True,
                 activation=F.relu,
                 use_cls=False,
                 **kwargs):
        """Multi-head attention.
        :param hidden_state_size: Size of each input sample.
        :param head_number: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if attention_size % head_number != 0:
            raise ValueError('`attn_size`({}) should be divisible by `head_num`({})'.format(attention_size, head_number))
        self.in_features = hidden_state_size
        self.head_num = head_number
        self.activation = activation
        self.bias = bias
        self.use_cls = use_cls
        self.local_size = local_size
        self.global_size = global_size
        self.linear_q = nn.Linear(hidden_state_size + local_size + global_size, attention_size, bias)
        self.linear_k = nn.Linear(hidden_state_size + local_size + global_size, attention_size, bias)
        self.linear_v = nn.Linear(hidden_state_size, attention_size, bias)
        self.linear_o = nn.Linear(attention_size, hidden_state_size, bias)
        self.layer_norm = nn.BatchNorm1d(attention_size)  # Heike: new

    #def forward(self, q, k, v, mask=None):
    def forward(self,
                h: Tensor,  # shape[batch_size, stc_length, in_features]
                mask: Tensor,  # shape[batch_size, stc_length]
                g: Tensor,  # shape[batch_size, global_size]
                l: Tensor,  # shape[batch_size, stc_length, local_features]
                **kwargs: dict) ->  Tensor: # shape[batch_size, stc_length]
        x_extended_list = [h]
        if g is not None:
            seq_len = h.size()[1] # seq_len == stc_length
            batch_size, global_size = g.size()
            assert global_size == self.global_size
                                                                    #basically here we are puting the global features for each elem. in the seq.
            x_extended_list.append(g.unsqueeze(1).expand(batch_size, seq_len, global_size))
        if l is not None:
            assert l.size()[2] == self.local_size
            x_extended_list.append(l)
        if len(x_extended_list) > 1:
            x_ext = torch.cat(x_extended_list, dim=2)
        else:
            x_ext = x_extended_list[0]
        # x_ext[batch_size, seq_length, in_features+global_size+local_size]

        q = self.linear_q(x_ext)
        k = self.linear_k(x_ext)
        v = self.linear_v(h)
        # q,k,v[batch_size, seq_length, attention_size]

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        residual = q  # Heike: new

        scores = ScaledDotProductAttentionScores()(q,k,v,mask)

        return scores

        q = self._reshape_to_batches(q) # [attention_size, seq_length, batch_size] NO!
        k = self._reshape_to_batches(k) # [attention_size, seq_length, batch_size]
        v = self._reshape_to_batches(v) # [attention_size, seq_length, batch_size]

        # q/k/v.shape: batchsize * num_heads x seq_len x input_dim / num_heads
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1) # [head_num, batch_size, seq_length]
        # x_mask.shape: num_heads x batchsize x seq_len
        y, scores, attention = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)
        # attention_scores = torch.diagonal(scores, dim1=1, dim2=2)

        attention_scores = self._reshape_from_batches(scores)


        # return attention

        y_perm = y.permute(0, 2, 1)  # Heike: new
        y_perm = y_perm.contiguous() + residual.permute(0, 2, 1).contiguous()   # Heike: new
        y_norm = self.layer_norm(y_perm)  # Heike: new
        y = y_norm.permute(0, 2, 1)  # Heike: new

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)

        y_perm = y.permute(0, 2, 1)
        pooled_y = F.max_pool1d(y_perm, kernel_size=y_perm.size()[-1]).squeeze()
        cls_y = y[:,0,:]

        if self.use_cls:
            return cls_y
        else:
            return pooled_y



    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, attn_size = x.size()
        sub_dim = attn_size // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, attn_size = x.size()
        batch_size //= self.head_num
        out_dim = attn_size * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, attn_size)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
