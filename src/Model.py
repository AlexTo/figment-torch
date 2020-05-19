import h5py
import torch
from torch import nn


class FigmentModel(nn.Module):
    def __init__(self, sub_words_emb_file, sub_words_num_emb, sub_words_emb_dim, clr_num_emb, clr_emb_dim,
                 clr_max_length=30, clr_out_channels=50, clr_kernels=range(7), all_embs_dim=852, hidden_dim=900,
                 output_dim=102):
        super(FigmentModel, self).__init__()

        sub_words_emb = h5py.File(sub_words_emb_file, 'r')
        sub_words_emb = sub_words_emb['vectors']
        self.clr_kernels = clr_kernels
        self.sub_words_emb = nn.Embedding(num_embeddings=sub_words_num_emb, embedding_dim=sub_words_emb_dim)
        self.sub_words_emb.weight = nn.Parameter(torch.from_numpy(sub_words_emb.value))
        self.sub_words_emb.weight.requires_grad = False
        self.clr_emb = nn.Embedding(num_embeddings=clr_num_emb, embedding_dim=clr_emb_dim)

        self.clr_convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=clr_out_channels,
                                                  kernel_size=(k + 1, clr_emb_dim)) for k in clr_kernels])
        self.clr_max_pools = nn.ModuleList([nn.MaxPool2d(kernel_size=(clr_max_length - k, 1)) for k in clr_kernels])

        self.linear1 = nn.Linear(in_features=all_embs_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, ent_emb, letters, sub_words, tc):
        letters_emb = self.clr_emb(letters)
        letters_emb = torch.unsqueeze(letters_emb, 1)
        sub_words_emb = self.sub_words_emb(sub_words).float()
        sub_words_emb = torch.mean(sub_words_emb, dim=1)
        clr_conv_outs = []
        for k in self.clr_kernels:
            conv = self.clr_convs[k]
            max_pool = self.clr_max_pools[k]
            clr_out = max_pool(conv(letters_emb)).squeeze()
            clr_conv_outs.append(clr_out)
        clr_conv_outs = torch.cat(clr_conv_outs, dim=1)

        all_embs = torch.cat([ent_emb, sub_words_emb, clr_conv_outs, tc], dim=1)
        out = torch.relu(self.linear1(all_embs))
        out = torch.sigmoid(self.linear2(out))
        return out
