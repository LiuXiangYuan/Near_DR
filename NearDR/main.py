import os
import torch
import logging
import transformers

from transformers import HfArgumentParser

from utils import set_seed

from Arguments import (
    DataTrainingArguments,
    ModelArguments,
    FaissArguments,
    MyTrainingArguments
)

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

    device = torch.device('cuda' if torch.cuda.is_available() and not training_args.no_cuda else 'cpu')
    training_args.n_gpu = torch.cuda.device_count()
    training_args.device = device

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


if __name__ == '__main__':
    main()
