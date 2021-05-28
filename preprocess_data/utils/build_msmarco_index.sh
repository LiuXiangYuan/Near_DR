#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=build_msmarco_index
#SBATCH -o ./ance_doc_index.%A.out
#SBATCH -e ./ance_doc_index.%A.err
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu06
#SBATCH -c2
#SBATCH --mem=50G
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate dpr

PYTHONPATH=/data/liuxiangyuan-slurm/work2021/DR_projection/data/utils/

# CUDA_VISIBLE_DEVICES=3 python -u encode_corpus_msmarco_passage.py \
# --encoder castorini/ance-msmarco-passage \
# --corpus /data/liuxiangyuan-slurm/work2021/DR_projection/data/marco_passage/passage_json \
# --index /data/liuxiangyuan-slurm/work2021/DR_projection/data/marco_passage/ance_passage_index \
# --batch 256 \
# --device cuda:0

CUDA_VISIBLE_DEVICES=0 python -u encode_corpus_msmarco_doc.py \
--encoder castorini/ance-msmarco-doc-maxp \
--corpus /data/liuxiangyuan-slurm/work2021/DR_projection/data/marco_doc/document_json \
--index /data/liuxiangyuan-slurm/work2021/DR_projection/data/marco_doc/ance_document_index \
--batch 512 \
--device cuda:0

EOT