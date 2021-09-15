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
class ScaledDotProductAttentionEnriched(nn.Module):

    def forward(self, query, key, value, local_f=None, global_f=None, mask=None):
        dk = query.size()[-1]
        # query, key: batch_size * self.head_num x seq_len x sub_dim
        projs_q = query
        projs_k = key
        if global_f is not None:
            projs_q = torch.cat([projs_q, global_f], dim=2)
            projs_k = torch.cat([projs_k, global_f], dim=2)
        if local_f is not None:
            projs_q = torch.cat([projs_q, local_f], dim=2)
            projs_k = torch.cat([projs_k, local_f], dim=2)
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        scores = projs_q.matmul(projs_k.transpose(-2, -1)) / math.sqrt(dk)
        # scores: batch_size * self.head_num x seq_len x seq_len
        if mask is not None:
            head_num, batchsize, seq_len = mask.shape
            masked_reshaped = mask.view(batchsize * head_num, seq_len)
            masked_reshaped = masked_reshaped.unsqueeze(1)
            scores = scores.masked_fill(masked_reshaped == 1, -1e9)
        attention = F.softmax(scores, dim=-1)
        result = attention.matmul(value)
        return result


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
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if attention_size % head_number != 0:
            raise ValueError('`attn_size`({}) should be divisible by `head_num`({})'.format(attn_size, head_num))
        self.in_features = hidden_state_size
        self.head_num = head_number
        self.activation = activation
        self.bias = bias
        self.use_cls = use_cls
        self.attn_size = attention_size
        self.local_size = local_size
        self.global_size = global_size
        if local_size > 0:
            self.linear_local = nn.Linear(local_size, attention_size, bias)
        if global_size > 0:
            self.linear_global = nn.Linear(global_size, attention_size, bias)
        self.linear_q = nn.Linear(hidden_state_size, attention_size, bias)
        self.linear_k = nn.Linear(hidden_state_size, attention_size, bias)
        self.linear_v = nn.Linear(hidden_state_size, attention_size, bias)
        self.linear_o = nn.Linear(attention_size, hidden_state_size, bias)
        self.layer_norm = nn.BatchNorm1d(attention_size)  # Heike: new
        # self.attn_layer = ScaledDotProductAttentionEnriched(attn_size // self.head_num)

    def forward(self,
                h,
                mask,
                g,
                l,
                **kwargs):
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        if g is not None:
            global_proj = self.linear_global(g.view(-1, self.global_size)).contiguous().view(
                batch_size, self.attn_size).unsqueeze(1).expand(batch_size, seq_len, self.attn_size)
        else:
            global_proj = None
        if l is not None:
            local_proj = self.linear_local(l.view(-1, self.local_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
        else:
            local_proj = None

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
            if global_proj is not None:
                global_proj = self.activation(global_proj)
            if local_proj is not None:
                local_proj = self.activation(local_proj)

        residual = q  # Heike: new

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if global_proj is not None:
            global_proj = self._reshape_to_batches(global_proj)
        if local_proj is not None:
            local_proj = self._reshape_to_batches(local_proj)

        # q/k/v.shape: batchsize * num_heads x seq_len x attn_dim / num_heads
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        # x_mask.shape: num_heads x batchsize x seq_len
        y = ScaledDotProductAttentionEnriched()(q, k, v, local_proj, global_proj, mask)
        y = self._reshape_from_batches(y)

        # TODO: Remove debug.
        o: Tensor = torch.sum(self.activation( self.linear_o( y ) ), dim=1)
        return o , o

        y_perm = y.permute(0, 2, 1)  # Heike: new
        y_perm = y_perm.contiguous() + residual.permute(0, 2, 1).contiguous()  # Heike: new
        y_norm = self.layer_norm(y_perm)  # Heike: new
        y = y_norm.permute(0, 2, 1)  # Heike: new

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)

        y_perm = y.permute(0, 2, 1)
        pooled_y = F.max_pool1d(y_perm, kernel_size=y_perm.size()[-1]).squeeze()
        cls_y = y[:, 0, :]

        if self.use_cls:
            return y, cls_y
        else:
            return y, pooled_y

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
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, attn_size = x.size()
        batch_size //= self.head_num
        out_dim = attn_size * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, attn_size) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
