import os
import torch
import logging
import transformers

from utils import set_seed
from transformers import HfArgumentParser
from dataset.utils import load_rel, TextTokenIdsCache

from transformers import (
    AutoConfig,
    AutoTokenizer
)

from arguments import (
    DataTrainingArguments,
    ModelArguments,
    FaissArguments,
    MyTrainingArguments
)

from dataset.dataset import (
    TrainInbatchDataset,
    TrainInbatchWithHardDataset,
    TrainInbatchWithRandDataset
)

from dataset.collation import (
    dual_get_collate_function,
    triple_get_collate_function
)

from models.models import get_model_class

logger = logging.Logger(__name__)


def main():
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, FaissArguments, MyTrainingArguments))
    data_args, model_args, faiss_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # device = torch.device('cuda' if torch.cuda.is_available() and not training_args.no_cuda else 'cpu')
    # training_args.n_gpus = torch.cuda.device_count() if not training_args.no_cuda else 0
    # training_args.device = device

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed, training_args.n_gpu)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task="msmarco",
        gradient_checkpointing=model_args.gradient_checkpointing
    )
    config.gradient_checkpointing = model_args.gradient_checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )

    rel_dict = load_rel(data_args.label_path)
    if data_args.data_type == 0:
        data_type = 'documents'
    else:
        data_type = 'passages'

    if training_args.hard_neg:
        train_dataset = TrainInbatchWithHardDataset(
            rel_file=data_args.label_path,
            rank_file=data_args.hardneg_path,
            queryids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix="train-query"),
            docids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix=data_type),
            max_query_length=data_args.max_query_length,
            max_doc_length=data_args.max_doc_length,
            hard_num=training_args.per_query_hard_num
        )
        data_collator = triple_get_collate_function(
            data_args.max_query_length, data_args.max_doc_length,
            rel_dict=rel_dict, padding=training_args.padding)
        model_class = get_model_class(model_args.model_name_or_path, hard_neg=True)
    elif training_args.rand_neg:
        train_dataset = TrainInbatchWithRandDataset(
            rel_file=data_args.label_path,
            rand_num=training_args.per_query_hard_num,
            queryids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix="train-query"),
            docids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix=data_type),
            max_query_length=data_args.max_query_length,
            max_doc_length=data_args.max_doc_length
        )
        data_collator = triple_get_collate_function(
            data_args.max_query_length, data_args.max_doc_length,
            rel_dict=rel_dict, padding=training_args.padding)
        model_class = get_model_class(model_args.model_name_or_path, rand_neg=True)
    else:
        train_dataset = TrainInbatchDataset(
            rel_file=data_args.label_path,
            queryids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix="train-query"),
            docids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix=data_type),
            max_query_length=data_args.max_query_length,
            max_doc_length=data_args.max_doc_length
        )
        data_collator = dual_get_collate_function(
            data_args.max_query_length, data_args.max_doc_length,
            rel_dict=rel_dict, padding=training_args.padding)
        model_class = get_model_class(model_args.model_name_or_path)

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )


if __name__ == '__main__':
    main()
