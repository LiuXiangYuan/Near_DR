import os
import pickle

from .utils import wc_cmd
from .utils import multi_file_process
from .utils import merging_split_dir
from .preprocessingfn import PassagePreprocessingFn, QueryPreprocessingFn


def write_query_rel(args, pid2offset, qid2offset_file,
                    query_file, positive_id_file, out_query_file, standard_qrel_file):

    print("Writing query files " + str(out_query_file) +
          " and " + str(standard_qrel_file))
    query_collection_path = os.path.join(args.data_dir, query_file)
    if positive_id_file is None:
        query_positive_id = None
        query_positive_id_path = None
        valid_query_num = int(wc_cmd(query_collection_path).split()[0])
    else:
        query_positive_id = set()
        query_positive_id_path = os.path.join(
            args.data_dir,
            positive_id_file,
        )

        print("Loading query_2_pos_docid")
        for line in open(query_positive_id_path, 'r', encoding='utf8'):
            query_positive_id.add(int(line.split()[0]))
        valid_query_num = len(query_positive_id)

    out_query_path = os.path.join(args.out_data_dir, out_query_file)

    print('start query file split processing')
    splits_dir_lst, _ = multi_file_process(
        args, args.threads, query_collection_path,
        out_query_path, QueryPreprocessingFn,
        args.max_query_length
        )

    print('start merging splits')
    qid2offset, idx = merging_split_dir(
        splits_dir_lst, out_query_path,
        valid_query_num, args.max_query_length,
        merge_query=True, query_positive_id=query_positive_id
    )

    qid2offset_path = os.path.join(args.out_data_dir, qid2offset_file)
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)
    print("done saving qid2offset")

    if positive_id_file is None:
        print("No qrels file provided")
        return
    print("Writing qrels")
    with open(os.path.join(args.out_data_dir, standard_qrel_file), "w", encoding='utf-8') as qrel_output: 
        out_line_count = 0
        for line in open(query_positive_id_path, 'r', encoding='utf-8'):
            topicid, _, docid, rel = line.split()
            topicid = int(topicid)
            if args.data_type == 0:
                docid = int(docid[1:])
            else:
                docid = int(docid)
            qrel_output.write(str(qid2offset[topicid]) +
                              "\t0\t" + str(pid2offset[docid]) +
                              "\t" + rel + "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))


def preprocess(args):

    if args.data_type == 0:
        in_passage_path = os.path.join(args.data_dir, "msmarco-docs.tsv")
    else:
        in_passage_path = os.path.join(args.data_dir, "collection.tsv")

    out_passage_path = os.path.join(args.out_data_dir, "passages")

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    print('start passage file split processing')
    splits_dir_lst, all_linecnt = multi_file_process(
        args, args.threads, in_passage_path,
        out_passage_path, PassagePreprocessingFn,
        args.max_seq_length
        )

    print('start merging splits')
    pid2offset, idx = merging_split_dir(
        splits_dir_lst, out_passage_path,
        all_linecnt, args.max_seq_length
    )
    
    pid2offset_path = os.path.join(args.out_data_dir, "pid2offset.pickle")
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    print("done saving pid2offset")
    
    if args.data_type == 0:
        write_query_rel(
            args,
            pid2offset,
            "train-qid2offset.pickle",
            "msmarco-doctrain-queries.tsv",
            "msmarco-doctrain-qrels.tsv",
            "train-query",
            "train-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-docs.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "msmarco-docdev-queries.tsv",
            "msmarco-docdev-qrels.tsv",
            "dev-query",
            "dev-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "docleaderboard-queries.tsv",
            None,
            "lead-query",
            None)
    else:
        write_query_rel(
            args,
            pid2offset,
            "train-qid2offset.pickle",
            "queries.train.tsv",
            "qrels.train.tsv",
            "train-query",
            "train-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "queries.dev.small.tsv",
            "qrels.dev.small.tsv",
            "dev-query",
            "dev-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-pass.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "queries.eval.small.tsv",
            None,
            "lead-query",
            None)
