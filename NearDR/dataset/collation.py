import torch

from .utils import pack_tensor_2D


def get_collate_function(max_seq_length):
    cnt = 0

    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function


def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0

    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function


def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        rel_pair_mask = [
            [1 if docid not in rel_dict[qid] else 0 for docid in doc_ids]
            for qid in query_ids
        ]
        input_data = {
            "input_query_ids": query_data['input_ids'],
            "query_attention_mask": query_data['attention_mask'],
            "input_doc_ids": doc_data['input_ids'],
            "doc_attention_mask": doc_data['attention_mask'],
            "rel_pair_mask": torch.FloatTensor(rel_pair_mask),
            }
        return input_data
    return collate_function


def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        hard_doc_data, hard_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        rel_pair_mask = [
            [1 if docid not in rel_dict[qid] else 0 for docid in doc_ids]
            for qid in query_ids
        ]
        hard_pair_mask = [
            [1 if docid not in rel_dict[qid] else 0 for docid in hard_doc_ids]
            for qid in query_ids
        ]
        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0][2])
        input_data = {
            "input_query_ids": query_data['input_ids'],
            "query_attention_mask": query_data['attention_mask'],
            "input_doc_ids": doc_data['input_ids'],
            "doc_attention_mask": doc_data['attention_mask'],
            "other_doc_ids": hard_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_attention_mask": hard_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),
            "rel_pair_mask": torch.FloatTensor(rel_pair_mask),
            "hard_pair_mask": torch.FloatTensor(hard_pair_mask),
            }
        return input_data
    return collate_function
