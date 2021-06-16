#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DR_infer
#SBATCH -o ./model_logs/marco_passage/infer-model-roberta-base-ts4.%A.out
#SBATCH -e ./model_logs/marco_passage/infer-model-roberta-base-ts4.%A.err
#SBATCH -p debug
#SBATCH --gres=gpu:6
#SBATCH --nodelist=gpu03
#SBATCH -c5
#SBATCH --mem=100G
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate dpr

PYTHONPATH=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/

# CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u inference.py \
--preprocess_dir ${PREPROCESS_DIR} \
--model_path ${MODEL_PATH} \
--output_dir ${OUTPUT_DIR} \
--max_query_length ${MAX_QUERY_LENGTH} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--eval_batch_size ${BATCH_SIZE} \
--mode ${MODE} \
--topk ${TOPK}

EOT