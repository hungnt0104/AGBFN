import torch
from torch import nn

from labml_helpers.module import Module

class GraphAttentionLayer(Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int, #out_features thi minh tu specify
                 is_concat: bool = False,
                 dropout: float = 0.5,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads #calculate the dimension of output features per head (each head will output an output features h, many heads output many output features)
        else:
            self.n_hidden = out_features
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False) #input size: in_features, output_size: self.n_hidden * n_heads, buoc nay tuong tu voi buoc nhan X voi W
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False) #dau vao la 2 features node sau khi da trai qua linear transformation (nen moi nhan 2), dau ra la e (coefficient)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope) #cho e di qua leaky relu
        self.softmax = nn.Softmax(dim=1) #to calculate alpha
        self.dropout = nn.Dropout(dropout) #set 1 so alpha ngau nhien = 0, tuc la o moi iteration thi neighborhood cua 1 node se khac nhau, boi vi khi set alpha=0 thi coi nhu 2 node k lien quan den nhau
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor): # shape cua h: [n_nodes, in_features], shape cua adj: [n_nodes, n_nodes, n_heads]
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden) #view: dung de reshape. linear layer lay input voi dau vao la [batch size, in_features], dau ra la [n_nodes, self.n_hidden * n_heads]
        g_repeat = g.repeat(n_nodes, 1, 1) #bay gio g_repeat se co shape la n_nodes*n_nodes, n_heads*1, n_hidden*1
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf')) #mask: set attention score ve am vo cuc, dam bao no se k tham gia vao viec dung softmax de normalization
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g) #final output: new features h cua 1 head
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
