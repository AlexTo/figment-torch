import argparse
import pickle
import h5py
import torch
import yaml
import numpy as np
import scipy.sparse as sp
from flair.data import Sentence

from src.InferSent import InferSent
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--seed", type=int, default=23455, required=False)
    parser.add_argument("--emb_type", type=str, default="flair", required=False)
    parser.add_argument("--target_file", type=str, default="data/_targets.h5py", required=False)
    parser.add_argument("--glove_file", type=str, default="data/glove.840B.300d.txt", required=False)
    parser.add_argument("--infer_sent_file", type=str, default="data/infersent1.pkl", required=False)
    parser.add_argument("--gensim_glove_output_file", type=str, default="data/gensim_glove_vectors.txt", required=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    return args, device


def batched_target_to_adj(targets):
    batch_size = targets.size()[0]
    n = targets.size()[1]
    s_target = torch.reshape(targets, [batch_size * n])
    zero = torch.zeros(batch_size * n)
    z_target = torch.where(s_target < 0, s_target, zero)
    indices = torch.nonzero(z_target, as_tuple=True)[0]
    s_mask_adj = torch.ones((batch_size * n, n))
    s_mask_adj[indices] = 0
    mask_adj = torch.reshape(s_mask_adj, [batch_size, n, n])
    mask_adj_r = torch.rot90(mask_adj, 1, [1, 2])
    adj = (mask_adj.bool() & mask_adj_r.bool() & torch.logical_not(
        torch.diag_embed(torch.ones((batch_size, n))).bool())).float()
    return adj


def batched_adj_to_freq(adj):
    return torch.sum(adj, dim=0)


def normalise(A, t=0.4):
    # Thresholding
    A = torch.div(A, torch.add(A.sum(0, keepdim=True), 1e-6))
    A.masked_fill_(A < t, 0.0)
    A.masked_fill_(A >= t, 1.0)
    # Normalisation
    A = torch.div(torch.mul(A, 0.25), torch.add(A.sum(0, keepdim=True), 1e-6))
    # Add identity matrix
    mask = torch.eye(A.shape[0]).bool()
    A.masked_fill_(mask, 1.0)

    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def process_type_embeddings(args, device):
    c = 0.15
    targets = h5py.File(args.target_file, 'r')

    type_to_ix = yaml.load(targets['targets'].attrs['type_to_ix'])
    types = [t.replace('-', ' ').replace('_', ' ').strip() for t in type_to_ix]

    targets_ds = targets['targets']
    targets_ds = torch.tensor(targets_ds[:303798, :], dtype=torch.float32)
    targets_ds[targets_ds == 0] = -1
    type_adj = batched_target_to_adj(targets_ds)
    type_adj = batched_adj_to_freq(type_adj)
    type_adj = normalise(type_adj)
    # type_adj = c * torch.inverse(torch.eye(type_adj.shape[0]) - (1 - c) * type_adj)

    with open('data/type_adj.pickle', 'wb') as ta:
        pickle.dump(type_adj.numpy(), ta)

    if args.emb_type == 'infer_sent':
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        infer_sent = InferSent(params_model).to(device)
        infer_sent.load_state_dict(torch.load(args.infer_sent_file))
        infer_sent.set_w2v_path(args.glove_file)

        infer_sent.build_vocab(types, tokenize=True)
        type_embeddings = infer_sent.encode(types)

    if args.emb_type == 'flair':
        glove_embedding = WordEmbeddings('glove')
        document_embeddings = DocumentRNNEmbeddings([glove_embedding], hidden_size=512)
        type_embeddings = []
        for t in types:
            sentence = Sentence(t)
            document_embeddings.embed(sentence)
            type_embeddings.append(sentence.get_embedding().cpu().detach().numpy())
        type_embeddings = np.array(type_embeddings)
    with open('data/type_embeddings.pickle', 'wb') as te:
        pickle.dump(type_embeddings, te)


def main():
    args, device = initialize()
    process_type_embeddings(args, device)


if __name__ == '__main__':
    main()
