MODEL_NAME_OR_PATH=roberta-base
DATA_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/raw_data/marco_passage/
OUT_DATA_DIR=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/
DATA_TYPE=1
# 120 for passage, 512 for document
MAX_SEQ_LENGTH=120

RUN_NAME="preprocess.sh"

export MODEL_NAME_OR_PATH DATA_DIR OUT_DATA_DIR DATA_TYPE MAX_SEQ_LENGTH

bash ${RUN_NAME}