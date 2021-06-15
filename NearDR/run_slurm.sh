# code below is for tokenize

# we choice model [roberta-base, distilbert-base-uncased,cross-encoder/ms-marco-TinyBERT-L-2-v2]

# MODEL_NAME_OR_PATH=roberta-base
# DATA_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/raw_data/marco_passage/
# OUT_DATA_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/roberta-base-ts2
# DATA_TYPE=1
# # 120 for passage, 512 for document
# MAX_SEQ_LENGTH=120

# RUN_NAME="preprocess.sh"

# export MODEL_NAME_OR_PATH DATA_DIR OUT_DATA_DIR DATA_TYPE MAX_SEQ_LENGTH

# bash ${RUN_NAME}

######################################
# code below is for training

# RUN_NAME="train_model.sh"

# # data args
# DATA_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/roberta-base-ts2/
# LABEL_PATH=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/roberta-base-ts2/train-qrel.tsv
# # 1 for passage, 0 for document
# DATA_TYPE=1
# # 120 for passage, 512 for document
# MAX_SEQ_LENGTH=120
# MAX_QUERY_LENGTH=24

# export DATA_DIR LABEL_PATH DATA_TYPE MAX_SEQ_LENGTH MAX_QUERY_LENGTH

# # model args
# MODEL_NAME_OR_PATH=roberta-base

# export MODEL_NAME_OR_PATH

# # train args
# OUTPUT_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/model_output/marco_passage/roberta-base-ts2
# OPTIMIZER_STR=lamb
# HARD_NEG=False
# RAND_NEG=False
# EXAM_MODEL=other
# PER_BATCH_SIZE=128
# LEARNING_RATE=2e-4
# GRADIENT_ACCUMULATION_STEPS=2
# OVERWRITE=True
# DDP=False
# NUM_EPOCH=10

# export OUTPUT_DIR OPTIMIZER_STR HARD_NEG RAND_NEG EXAM_MODEL PER_BATCH_SIZE LEARNING_RATE GRADIENT_ACCUMULATION_STEPS OVERWRITE DDP NUM_EPOCH

# bash ${RUN_NAME}


######################################
# code below is for inferencing

RUN_NAME="infer_model.sh"

PREPROCESS_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/roberta-base-ts2/
MODEL_PATH=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/model_output/marco_passage/roberta-base-ts2/
OUTPUT_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/infer_output/marco_passage/roberta-base-ts2/
MAX_QUERY_LENGTH=24
MAX_SEQ_LENGTH=120
BATCH_SIZE=1024
MODE=dev
TOPK=100
FAISS_GPUS=[0]

export PREPROCESS_DIR MODEL_PATH OUTPUT_DIR MAX_QUERY_LENGTH MAX_SEQ_LENGTH BATCH_SIZE MODE TOPK FAISS_GPUS

bash ${RUN_NAME}
