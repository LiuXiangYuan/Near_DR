import os
import json
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def load_rel(rel_path):
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].append(pid)
    return dict(reldict)


def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


class TextTokenIdsCache:
    def __init__(self, data_dir, prefix):
        dir_path = os.path.join(data_dir, prefix)
        meta = json.load(open(f"{dir_path}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']
        try:
            self.ids_arr = np.memmap(f"{dir_path}.memmap",
                                     shape=(self.total_number, self.max_seq_len),
                                     dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{dir_path}_length.npy")
        except FileNotFoundError:
            dir_path = os.path.join(data_dir, 'memmap', prefix)
            self.ids_arr = np.memmap(f"{dir_path}.memmap",
                                     shape=(self.total_number, self.max_seq_len),
                                     dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{dir_path}_length.npy")
        assert len(self.lengths_arr) == self.total_number

    def __len__(self):
        return self.total_number

    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]
