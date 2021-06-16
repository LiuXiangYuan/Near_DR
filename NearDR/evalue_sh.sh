INPUT_NAME=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/infer_output/marco_passage/roberta-base-ts4/dev.rank.tsv
LABEL_PATH=/data/liuxiangyuan-slurm/work2021/Near_DR/dataset/preprocessed/marco_passage/roberta-base-ts2/dev-qrel.tsv
CONVERT_NAME=/data/liuxiangyuan-slurm/work2021/Near_DR/NearDR/infer_output/marco_passage/roberta-base-ts4/dev.rank.trec

python -m pyserini.eval.msmarco_passage_eval ${LABEL_PATH} ${INPUT_NAME}

python -m pyserini.eval.convert_msmarco_run_to_trec_run --input ${INPUT_NAME} --output ${CONVERT_NAME}
python -m pyserini.eval.trec_eval -c -mrecall.100 -mmap ${LABEL_PATH} ${CONVERT_NAME}