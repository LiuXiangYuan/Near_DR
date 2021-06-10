import os
import torch
import faiss
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from arguments import InferenceArguments
from transformers import HfArgumentParser
from transformers import AutoConfig
from models.models import get_inference_model
from dataset.utils import TextTokenIdsCache
from dataset.dataset import SequenceDataset, SubsetSeqDataset
from dataset.collation import single_get_collate_function

from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)

logger = logging.Logger(__name__)


def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)


def query_inference(model, args, embedding_size):
    query_collator = single_get_collate_function(args.max_query_length)
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )
    query_memmap = np.memmap(args.query_memmap_path,
                             dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.queryids_memmap_path,
                                dtype=np.int32, mode="w+", shape=(len(query_dataset), ))

    prediction(model, query_collator, args, query_dataset, query_memmap, queryids_memmap, is_query=True)


def doc_inference(model, args, embedding_size):
    if os.path.exists(args.doc_memmap_path):
        print(f"{args.doc_memmap_path} exists, skip inference")
        return
    doc_collator = single_get_collate_function(args.max_seq_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")
    subset = list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_seq_length
    )
    assert not os.path.exists(args.doc_memmap_path)
    doc_memmap = np.memmap(args.doc_memmap_path,
                           dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path,
                             dtype=np.int32, mode="w+", shape=(len(doc_dataset), ))
    prediction(model, doc_collator, args, doc_dataset, doc_memmap, docid_memmap, is_query=False)


def main():
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_path, gradient_checkpointing=False)
    model_class = get_inference_model(args.model_path)
    model = model_class.from_pretrained(args.model_path, config=config)
    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)
    query_inference(model, args, output_embedding_size)
    doc_inference(model, args, output_embedding_size)
    
    del model
    torch.cuda.empty_cache()

    doc_embeddings = np.memmap(args.doc_memmap_path, dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryids_memmap_path, dtype=np.int32, mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)

    with open(args.output_rank_file, 'w') as output_file:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                output_file.write(f"{qid}\t{pid}\t{idx+1}\n")


if __name__ == "__main__":
    main()
