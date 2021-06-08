#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DR_preprocess_passage
#SBATCH -o ./preprocess_log/marco_passage/preprocess-passage-roberta-base-ts4.%A.out
#SBATCH -e ./preprocess_log/marco_passage/preprocess-passage-roberta-base-ts4.%A.err
#SBATCH -p debug
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gpu04
#SBATCH -c10
#SBATCH --mem=80G
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate dpr

PYTHONPATH=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/

# CUDA_VISIBLE_DEVICES=0

python -u preprocess.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_dir ${DATA_DIR} \
--out_data_dir ${OUT_DATA_DIR} \
--data_type ${DATA_TYPE} \
--max_seq_length ${MAX_SEQ_LENGTH}

EOT