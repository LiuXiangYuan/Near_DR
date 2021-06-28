import json
import random

from typing import List
from torch.utils.data import Dataset

from .utils import load_rel


class SequenceDataset(Dataset):
    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length - 1, len(input_ids) - 1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1] * len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }
        return ret_val


class SubsetSeqDataset:
    def __init__(self, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))
        self.alldataset = SequenceDataset(ids_cache, max_seq_length)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]


class TrainInbatchDataset(Dataset):
    def __init__(self, rel_file, queryids_cache, docids_cache,
                 max_query_length, max_doc_length):
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.reldict = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        return query_data, passage_data


class TrainInbatchWithHardDataset(TrainInbatchDataset):
    def __init__(self, rel_file, rank_file, queryids_cache,
                 docids_cache, hard_num,
                 max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self,
                                     rel_file, queryids_cache, docids_cache,
                                     max_query_length, max_doc_length)
        self.rankdict = json.load(open(rank_file))
        assert hard_num > 0
        self.hard_num = hard_num

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        hardpids = random.sample(self.rankdict[str(qid)], self.hard_num)
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data


class TrainInbatchWithRandDataset(TrainInbatchDataset):
    def __init__(self, rel_file, queryids_cache,
                 docids_cache, rand_num,
                 max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self,
            rel_file, queryids_cache, docids_cache,
            max_query_length, max_doc_length)
        assert rand_num > 0
        self.rand_num = rand_num

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]
        return query_data, passage_data, rand_passage_data


class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache,
                 rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_poffsets'] = self.reldict[item]
        return ret_val
