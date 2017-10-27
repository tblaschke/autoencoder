# coding=utf-8
import json
from itertools import islice

import torch
import torch.utils.data as data


def replace_double(smi: str) -> str:
    for s, w in zip(['Br', 'Cl', 'Si', "GO"], ["Ö", "Ä", "Å", "Ü"]):
        smi = smi.replace(s, w)
    return smi


def replace_single(smi: str) -> str:
    for s, w in zip(['Br', 'Cl', 'Si', ""], ["Ö", "Ä", "Å", "Ü"]):
        smi = smi.replace(w, s)
    return smi


class SMILESReader(data.Dataset):

    def __init__(self, filepath, alphabetpath, subset=(0, None), maxlen=120):
        self.onehotencoder = OnehotEncoder(alphabetpath=alphabetpath, maxlen=maxlen)
        with open(filepath, 'r') as fd:
            self.smiles = [line.strip() for line in fd]
        self.smiles = self.smiles[subset[0]:subset[1]]

    def __getitem__(self, index):
        smi = self.smiles[index]
        one_hot = self.onehotencoder(smi)
        cat_hot = torch.LongTensor(1).zero_()
        cat_hot[0] = -1
        return one_hot, cat_hot

    def __len__(self):
        return len(self.smiles)


class DRD2Reader(data.Dataset):

    def __init__(self, filepath, alphabetpath, subset=(0, None), maxlen=120):
        self.onehotencoder = OnehotEncoder(alphabetpath=alphabetpath, maxlen=maxlen)
        with open(filepath, 'r') as fd:
            self.smiles = []
            self.cat = []
            self.id = []
            for line in islice(fd, 1, None):  # ignore the first header line
                line = line.split(" ")
                self.smiles.append(line[0].strip())
                self.cat.append(line[1].strip())
                self.id.append(line[2].strip())
        self.smiles = self.smiles[subset[0]:subset[1]]
        self.cat = self.cat[subset[0]:subset[1]]
        self.id = self.id[subset[0]:subset[1]]

    def __getitem__(self, index):
        smi = self.smiles[index]
        one_hot = self.onehotencoder(smi)
        cat_hot = torch.LongTensor(1).zero_()
        if self.cat[index] == "A":
            cat_hot[0] = 1
        else:
            cat_hot[0] = 0
        return one_hot, cat_hot

    def __len__(self):
        return len(self.smiles)


class MergedDataset(data.Dataset):
    def __init__(self, *args):
        self.datasets = list(args)
        self.len = 0
        for d in self.datasets:
            self.len += len(d)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        for d in self.datasets:
            if index >= len(d):
                index -= len(d)
            else:
                return d[index]


class OnehotEncoder(object):
    def __init__(self, alphabetpath, maxlen=120):
        with open(alphabetpath, 'r') as fd:
            alphabet = ''.join([" ", "GO"] + json.load(fd))
            alphabet = replace_double(alphabet)
            self.alphabet = {k: v for v, k in enumerate(alphabet)}
        self.maxlen = maxlen
        self.alphabetlen = len(alphabet)

    def __call__(self, smi: str):
        indices = torch.LongTensor(self.maxlen, 1)
        indices.zero_()
        smi = replace_double(smi.rstrip())
        for i, char in enumerate(smi):
            indices[i] = self.alphabet[char]
        one_hot = torch.zeros(self.maxlen, self.alphabetlen)
        one_hot.scatter_(1, indices, 1)
        return one_hot


class OnehotDecoder(object):

    def __init__(self, alphabetpath):
        with open(alphabetpath, 'r') as fd:
            self.alphabet = [" ", ""] + json.load(fd)  # Replace GO token with empty string

    def decode(self, onehot: torch.FloatTensor):
        if onehot.dim() == 2:
            onehot = onehot[None, :, :]
        maxs, indices = torch.max(onehot, 2)
        smiles = []
        for i in range(indices.size()[0]):
            chars = [self.alphabet[index] for index in indices[i,].view(-1)]
            smiles.append("".join(chars).strip())
        return smiles

    def decode_int(self, inds: torch.LongTensor):
        if inds.dim() == 2:
            inds = inds[None, :, :]
        smiles = []
        for i in range(inds.size()[0]):
            chars = [self.alphabet[index] for index in inds[i,].view(-1)]
            smiles.append("".join(chars).strip())
            return smiles


if __name__ == "__main__":
    reader = SMILESReader(filepath='data/prior_trainingset_DRD2_actives_removed.smi', alphabetpath='data/alphabet.json',
                          subset=(1, 4), maxlen=120)
    decoder = OnehotDecoder(alphabetpath='data/alphabet.json')
    a = reader[0][0]
    b = reader[1][0]
    c = torch.stack([a, b])
    print(decoder.decode(c))
