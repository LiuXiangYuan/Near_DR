from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class DataTrainingArguments:
    data_dir: str = field()  # "./data/passage or doc/preprocess"
    preprocess_dir: str = field()  # use prepare_hardneg.py to generate
    label_path: str = field(metadata={"help: the path of qrel.tsv file"})
    max_seq_length: int = field(default=64)
    max_query_length: int = field(default=24)  # 24
    max_doc_length: int = field(default=120)  #  512 for doc and 120 for passage


@dataclass
class ModelArguments:
    init_path: str = field()  # please use bm25 warmup model or roberta-base
    model_name_or_path: str = field()
    gradient_checkpointing: bool = field(default=False)


@dataclass
class FaissArguments:
    pembed_path: str = field()
    index_path: str = field()
    faiss_omp_num_threads: int = field(default=16)
    index_cpu: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default='./model_output')  # where to output
    padding: bool = field(default=False)
    optimizer_str: str = field(default="adamw")  # or lamb
    hard_neg: bool = field(default=False)
    per_query_hard_num: int = field(default=0)
    rand_neg: bool = field(default=False)
    neg_topk: int = field(default=200)
    overwrite_output_dir: bool = field(default=False)

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})

    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Run evaluation during training at each logging step."}, )

    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}, )

    learning_rate: float = field(default=5e-6, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=100.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=99999999999, metadata={"help": "Save checkpoint every X updates steps."})

    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    metric_cut: int = field(default=None)
