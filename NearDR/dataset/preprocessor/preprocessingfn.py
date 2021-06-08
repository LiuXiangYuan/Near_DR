from .utils import pad_input_ids


def PassagePreprocessingFn(args, line, tokenizer):
    # 0 is for document, 1 is for passage
    if args.data_type == 0:
        line_arr = line.split('\t')
        p_id = int(line_arr[0][1:])  # remove "D"

        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()
        # NOTE: This linke is copied from ANCE,
        # but I think it's better to use <s> as the separator,
        if args.is_roberta:
            full_text = url + "<sep>" + title + "<sep>" + p_text
        else:
            full_text = url + "[SEP]" + title + "[SEP]" + p_text
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    else:
        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])

        p_text = line_arr[1].rstrip()
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:args.max_doc_character]
    passage = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id, input_id_b, passage_len


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length,
        truncation=True)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id, input_id_b, passage_len
