import argparse
import os
import pickle

import neptune
import torch
import numpy as np
import pandas as pd
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import trange

from src.FigmentDataset import FigmentDataset
from src.Model import FigmentModel


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--seed", type=int, default=23455, required=False)
    parser.add_argument("--batch_size", type=int, default=1000, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-06)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--hidden_units", type=str, default="4096,2048,900",
                        help="hidden units in each hidden layer(including in_dim and out_dim), split with comma")
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, split with comma")
    parser.add_argument("--instance_normalization", action="store_true", default=False,
                        help="enable instance normalization")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")

    parser.add_argument("--type_adj_file", type=str, default="data/type_adj.pickle", required=False)
    parser.add_argument("--type_embeddings_file", type=str, default="data/type_embeddings.pickle", required=False)
    parser.add_argument("--entities_train_file", type=str, default="data/train.txt", required=False)
    parser.add_argument("--entities_dev_file", type=str, default="data/dev.txt", required=False)
    parser.add_argument("--entities_test_file", type=str, default="data/test.txt", required=False)
    parser.add_argument("--target_file", type=str, default="data/_targets.h5py", required=False)
    parser.add_argument("--ent_emb_file", type=str, default="data/_entvec.h5py", required=False)
    parser.add_argument("--letters_file", type=str, default="data/_letters.h5py", required=False)
    parser.add_argument("--sub_words_file", type=str, default="data/_subwords.h5py", required=False)
    parser.add_argument("--sub_words_emb_file", type=str, default="data/_subwords_embeddings.h5py", required=False)
    parser.add_argument("--sub_words_num_emb", type=int, default=143123, required=False)
    parser.add_argument("--sub_words_emb_dim", type=int, default=200, required=False)
    parser.add_argument("--tc_file", type=str, default="data/_tc.h5py", required=False)
    parser.add_argument("--clr_num_emb", type=int, default=83, required=False)
    parser.add_argument("--clr_emb_dim", type=int, default=10, required=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    return args, device


def evaluate(model, loader, device, criterion):
    model.eval()
    loss = 0.0
    batches = 0
    with torch.no_grad():
        for ent_emb, letters, sub_words, tc, targets in loader:
            batches += 1
            ent_emb, letters, sub_words, tc, targets = ent_emb.to(device), letters.to(torch.int64).to(
                device), sub_words.to(torch.int64).to(device), tc.to(device), targets.float().to(device)
            outputs = model(ent_emb, letters, sub_words, tc)
            batch_loss = criterion(outputs, targets)
            loss += batch_loss.item()
    loss = loss / batches
    return loss


def train(args, device):
    neptune.init('alexto/figment-multi')

    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay
    }
    neptune.create_experiment(name='Fix shuffle loading when testing',
                              params=params)
    train_ds = FigmentDataset(args.target_file, args.ent_emb_file, args.letters_file, args.sub_words_file,
                              args.tc_file, split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    dev_ds = FigmentDataset(args.target_file, args.ent_emb_file, args.letters_file, args.sub_words_file,
                            args.tc_file, split="dev")
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    with open(args.type_adj_file, 'rb') as f:
        type_adj = torch.tensor(pickle.load(f)).to_sparse().to(device)
    with open(args.type_embeddings_file, 'rb') as f:
        type_embeddings = torch.tensor(pickle.load(f)).to(device)

    model = FigmentModel(args.sub_words_emb_file, args.sub_words_num_emb, args.sub_words_emb_dim, args.clr_num_emb,
                         args.clr_emb_dim, type_adj, type_embeddings, n_units, n_heads, args.dropout, args.attn_dropout,
                         args.instance_normalization, args.diag).to(device)

    if os.path.exists('output/model.pt'):
        model.load_state_dict(torch.load('output/model.pt'))

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    bar = trange(0, args.epochs, desc="Training")
    dev_loss = np.nan
    for _ in bar:
        model.train()
        for ent_emb, letters, sub_words, tc, targets in train_loader:
            ent_emb, letters, sub_words, tc, targets = ent_emb.to(device), letters.to(torch.int64).to(device), \
                                                       sub_words.to(torch.int64).to(device), tc.to(device), \
                                                       targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(ent_emb, letters, sub_words, tc)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            bar.set_postfix({"train_loss": f"{loss:.4f}", "dev_loss": f"{dev_loss:.4f}"})
            neptune.log_metric("train_loss", loss)
        dev_loss = evaluate(model, dev_loader, device, criterion)
        neptune.log_metric("dev_loss", dev_loss)
    torch.save(model.state_dict(), 'output/model.pt')


def write_outputs(args, device, split='dev'):
    ds = FigmentDataset(args.target_file, args.ent_emb_file, args.letters_file, args.sub_words_file,
                        args.tc_file, split=split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    with open(args.type_adj_file, 'rb') as f:
        type_adj = torch.tensor(pickle.load(f)).to_sparse().to(device)

    model = FigmentModel(args.infer_sent_file, args.sub_words_emb_file, args.sub_words_num_emb, args.sub_words_emb_dim,
                         args.clr_num_emb, args.clr_emb_dim, type_adj).to(device)

    model.load_state_dict(torch.load('output/model.pt'))

    model.eval()
    if split == 'dev':
        entities_file = args.entities_dev_file
    else:
        entities_file = args.entities_test_file
    entities = pd.read_csv(entities_file, delimiter='\t', header=None)
    entities = entities[[0]]
    outputs = []
    with torch.no_grad():
        for ent_emb, letters, sub_words, tc, _ in loader:
            ent_emb, letters, sub_words, tc = ent_emb.to(device), letters.to(torch.int64).to(device), \
                                              sub_words.to(torch.int64).to(device), tc.to(device)
            output = model(ent_emb, letters, sub_words, tc)
            outputs.append(output)
    outputs = torch.cat(outputs).cpu().numpy()
    outputs_df = pd.DataFrame(outputs)
    with open(f'output/{split}.probs', 'w') as f:
        for idx in range(entities.shape[0]):
            entity = entities.iloc[idx, 0]
            score = " ".join(outputs_df.iloc[idx].to_numpy().astype(str))
            f.write('\t'.join([entity, score]) + '\n')


def main():
    args, device = initialize()
    if args.train:
        train(args, device)
    if args.test:
        write_outputs(args, device, 'dev')
        write_outputs(args, device, 'test')


if __name__ == '__main__':
    main()
