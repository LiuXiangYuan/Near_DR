#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DR_train_passage
#SBATCH -o ./model_logs/marco_passage/train-model-roberta-base-ts4.%A.out
#SBATCH -e ./model_logs/marco_passage/train-model-roberta-base-ts4.%A.err
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu04
#SBATCH -c10
#SBATCH --mem=80G
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate dpr

PYTHONPATH=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/

# CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
--data_dir ${DATA_DIR} \
--label_path ${LABEL_PATH} \
--data_type ${DATA_TYPE} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--max_query_length ${MAX_QUERY_LENGTH} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--output_dir ${OUTPUT_DIR} \
--optimizer_str ${OPTIMIZER_STR} \
--hard_neg ${HARD_NEG} \
--rand_neg ${RAND_NEG} \
--exam_mode ${EXAM_MODEL} \
--per_device_train_batch_size ${PER_BATCH_SIZE} \
--learning_rate ${LEARNING_RATE}

EOT