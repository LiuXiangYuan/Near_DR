import os
import json
import subprocess
import numpy as np
import multiprocessing

from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer


def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def tokenize_to_file(args, in_path, output_dir, line_fn, max_length, begin_idx, end_idx):
    if args.is_roberta:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True, cache_dir=None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True, cache_dir=None)
    os.makedirs(output_dir, exist_ok=True)
    data_cnt = end_idx - begin_idx
    # store qid or pid
    ids_array = np.memmap(
        os.path.join(output_dir, "ids.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    token_ids_array = np.memmap(
        os.path.join(output_dir, "token_ids.memmap"),
        shape=(data_cnt, max_length), mode='w+', dtype=np.int32)
    token_length_array = np.memmap(
        os.path.join(output_dir, "lengths.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    pbar = tqdm(total=end_idx-begin_idx, desc=f"Tokenizing")
    write_idx = 0
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length = line_fn(args, line, tokenizer)
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    pbar.close()
    assert write_idx == data_cnt - 1


def merging_split_dir(splits_dir_lst, output_path, line_number, max_seq_length,
                      merge_query=False, query_positive_id=None):

    token_ids_array = np.memmap(
        output_path + ".memmap",
        shape=(line_number, max_seq_length), mode='w+', dtype=np.int32)

    token_length_array = []
    offset = {}

    idx = 0
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(
            os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = np.memmap(
            os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
        split_token_length_array = np.memmap(
            os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
        for _id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            if merge_query and query_positive_id is not None and _id not in query_positive_id:
                # exclude the query as it is not in label set
                continue
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length)
            offset[_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(_id))

    assert len(token_length_array) == len(token_ids_array) == idx
    np.save(output_path + "_length.npy", np.array(token_length_array))

    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx, 'embedding_size': max_seq_length}
    with open(output_path + "_meta", 'w') as f:
        json.dump(meta, f)

    return offset, idx


def multi_file_process(args, num_process, in_path, out_path, line_fn, max_length):
    output_linecnt = wc_cmd(in_path)
    print("line cnt", output_linecnt)
    all_linecnt = int(output_linecnt.split()[0])
    run_arguments = []
    for i in range(num_process):
        begin_idx = round(all_linecnt * i / num_process)
        end_idx = round(all_linecnt * (i+1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        run_arguments.append((
                args, in_path, output_dir, line_fn,
                max_length, begin_idx, end_idx
            ))
    pool = multiprocessing.Pool(processes=num_process)
    pool.starmap(tokenize_to_file, run_arguments)
    pool.close()
    pool.join()
    splits_dir = [a[2] for a in run_arguments]
    return splits_dir, all_linecnt


def wc_cmd(in_path):
    output_linecnt = subprocess.check_output(["wc", "-l", in_path]).decode("utf-8")
    return output_linecnt
