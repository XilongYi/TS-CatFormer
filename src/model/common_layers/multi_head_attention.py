import torch.nn as nn
import torch
from einops import rearrange


def init_layer(layer):
    for p in layer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            nn.init.constant_(p.data, 0.0)
    if isinstance(layer, nn.Embedding):
        layer.weight.data[0].zero_()


def init_last_residual_layer(layer):
    layer.weight.data.zero_()
    layer.bias.data.zero_()

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, num_layer):
        super(MultiHeadAttention, self).__init__()

        self.num_layer = num_layer
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        # QKV
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        # edge
        self.linear_e = nn.Linear(hidden_size, num_heads * att_size)
        self.e_output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.init_param()

    def init_param(self):
        init_layer(self.linear_q)
        init_layer(self.linear_k)
        init_layer(self.linear_v)
        init_last_residual_layer(self.output_layer)
        #
        init_layer(self.linear_e)
        init_last_residual_layer(self.e_output_layer)

    def forward(self, q, k, v, e, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        e = rearrange(self.linear_e(e), 'b i j (h k) -> b h i j k', h=self.num_heads, k=d_k)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        # w = torch.einsum('b h i d, b h k d -> b h i k', q, k)
        # w = w + e
        qk_e = torch.einsum('b h i d, b h j d, b h i j d -> b h i j d', q, k, e)  # too damn slow?
        w = qk_e.sum(dim=-1)
        ##
        # Attention-Core
        if attn_bias is not None:
            w_bias = w + attn_bias
        else:
            w_bias = w

        w_softmax = torch.softmax(w_bias, dim=3)
        x = self.att_dropout(w_softmax)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size

        # e_output
        w = self.att_dropout(rearrange(qk_e, 'b h i j d -> b i j (h d)'))
        w = self.e_output_layer(w)
        return x, w
