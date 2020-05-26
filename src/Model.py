import h5py
import torch
from pygcn import GraphConvolution
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F


class FigmentModel(nn.Module):
    def __init__(self, sub_words_emb_file, sub_words_num_emb, sub_words_emb_dim, clr_num_emb,
                 clr_emb_dim, type_adj, type_embeddings, n_units, n_heads, dropout, attn_dropout,
                 instance_normalization, diag, clr_max_length=30, clr_out_channels=50, clr_kernels=range(7),
                 all_embs_dim=852, hidden_dim=512, type_embedding_dim=512, gcn_hidden_dim=2048, output_dim=102,
                 w=0.08):
        super(FigmentModel, self).__init__()
        sub_words_emb = h5py.File(sub_words_emb_file, 'r')
        sub_words_emb = sub_words_emb['vectors']
        self.dropout = dropout
        self.type_adj = type_adj
        self.type_embeddings = type_embeddings

        self.sub_words_emb = nn.Embedding(num_embeddings=sub_words_num_emb, embedding_dim=sub_words_emb_dim)
        self.sub_words_emb.weight = nn.Parameter(torch.from_numpy(sub_words_emb.value))
        self.sub_words_emb.weight.requires_grad = False
        self.sub_words_attention = TransformerEncoder(TransformerEncoderLayer(d_model=sub_words_emb_dim, nhead=4), 2)
        self.sub_words_linear = nn.Linear(sub_words_emb_dim, hidden_dim)

        self.clr_emb = nn.Embedding(num_embeddings=clr_num_emb, embedding_dim=clr_emb_dim)
        self.clr_emb.weight.data.uniform_(-w, w)
        self.clr_attention = TransformerEncoder(TransformerEncoderLayer(d_model=clr_emb_dim, nhead=4), 2)
        self.clr_linear = nn.Linear(clr_emb_dim, hidden_dim)

        self.ent_linear = nn.Linear(sub_words_emb_dim, hidden_dim)
        self.init_linear(self.ent_linear, w)

        self.type_linear = nn.Linear(type_embedding_dim, hidden_dim)
        self.init_linear(self.type_linear, w)

        self.tc_linear = nn.Linear(output_dim, hidden_dim)
        self.init_linear(self.tc_linear, w)

        self.gc1 = GraphConvolution(type_embedding_dim, type_embedding_dim)
        self.gc2 = GraphConvolution(type_embedding_dim, type_embedding_dim)

        self.all_embs_attention = TransformerEncoder(TransformerEncoderLayer(d_model=hidden_dim, nhead=4), 2)
        self.final_linear = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def init_linear(linear, w):
        linear.weight.data.uniform_(-w / 2, w / 2)
        linear.bias.data.fill_(0)

    def forward(self, ent_emb, letters, sub_words, tc):
        clr_emb = self.clr_emb(letters)
        clr_out = self.clr_attention(clr_emb)
        clr_out = F.normalize(self.clr_linear(clr_out[:, 0, :]), p=2, dim=1)

        sub_words_emb = self.sub_words_emb(sub_words).float()
        subwords_out = self.sub_words_attention(sub_words_emb)
        subwords_out = F.normalize(self.sub_words_linear(subwords_out[:, 0, :]), p=2, dim=1)

        type_embs = self.gc1(self.type_embeddings, self.type_adj)
        type_embs = self.gc2(type_embs, self.type_adj)  # 102 * 4096
        type_out = F.normalize(type_embs, p=2, dim=1)

        ent_out = self.ent_linear(ent_emb)
        ent_out = F.normalize(ent_out, p=2, dim=1)

        tc_out = self.tc_linear(tc)
        tc_out = F.normalize(tc_out, p=2, dim=1)

        all_embs = torch.stack([ent_out, clr_out, subwords_out, tc_out], dim=1)
        att_out = self.all_embs_attention(all_embs)
        att_out = att_out[:, 0, :]
        # out = torch.mm(att_out, type_out.T)
        out = self.final_linear(att_out)
        out = torch.sigmoid(out)
        return out
