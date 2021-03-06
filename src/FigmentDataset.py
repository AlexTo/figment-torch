import h5py
from torch.utils.data import Dataset


class FigmentDataset(Dataset):
    def __init__(self, target_file, ent_emb_file, letters_file, sub_words_file, types_cosine_file, split="train",
                 split_indices=[303798, 344018], letters_max_len=30, sub_words_max_len=4):
        self.letters_max_len = letters_max_len
        self.sub_words_max_len = sub_words_max_len
        targets = h5py.File(target_file, 'r')
        ent_emb = h5py.File(ent_emb_file, 'r')
        letters = h5py.File(letters_file, 'r')
        sub_words = h5py.File(sub_words_file, 'r')
        types_cosine = h5py.File(types_cosine_file, 'r')

        self.targets_ds = targets['targets']
        self.ent_emb_ds = ent_emb['entvec']
        self.letters_ds = letters['letters']
        self.sub_words_ds = sub_words['subwords']
        self.types_cosine_ds = types_cosine['tc']

        if split == "train":
            self.targets_ds = self.targets_ds[:split_indices[0], :]
            self.ent_emb_ds = self.ent_emb_ds[:split_indices[0], :]
            self.letters_ds = self.letters_ds[:split_indices[0], :]
            self.sub_words_ds = self.sub_words_ds[:split_indices[0], :]
            self.types_cosine_ds = self.types_cosine_ds[:split_indices[0], :]

        elif split == "dev":
            self.targets_ds = self.targets_ds[split_indices[0]:split_indices[1], :]
            self.ent_emb_ds = self.ent_emb_ds[split_indices[0]:split_indices[1], :]
            self.letters_ds = self.letters_ds[split_indices[0]:split_indices[1], :]
            self.sub_words_ds = self.sub_words_ds[split_indices[0]:split_indices[1], :]
            self.types_cosine_ds = self.types_cosine_ds[split_indices[0]:split_indices[1], :]
        else:
            self.targets_ds = self.targets_ds[split_indices[1]:, :]
            self.ent_emb_ds = self.ent_emb_ds[split_indices[1]:, :]
            self.letters_ds = self.letters_ds[split_indices[1]:, :]
            self.sub_words_ds = self.sub_words_ds[split_indices[1]:, :]
            self.types_cosine_ds = self.types_cosine_ds[split_indices[1]:, :]

    def __len__(self):
        return self.targets_ds.shape[0]

    def __getitem__(self, item):
        ent_emb = self.ent_emb_ds[item]
        letters = self.letters_ds[item, :self.letters_max_len]
        sub_words = self.sub_words_ds[item, :self.sub_words_max_len]
        tc = self.types_cosine_ds[item]
        target = self.targets_ds[item]
        return ent_emb, letters, sub_words, tc, target
