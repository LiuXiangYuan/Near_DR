import os
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file path', required=True)
    parser.add_argument('--output', type=str, help='output file path', required=True)
    args = parser.parse_args()

    collection_name = args.input
    output_name = args.output

    if not os.path.exists(collection_name):
        print('the input dir path does not exists, please check your input')
        assert False
    
    if not os.path.exists(output_name):
        os.mkdir(output_name)

    with open(collection_name, 'r', encoding='utf-8') as f, open(output_name, 'w', encoding='utf-8') as fw:
        for line in f:
            passage_id, passage = line.strip().split('\t')

            fw.write(json.dumps(dict(id=str(passage_id), contents=passage)) + '\n')

    print('json format done. format example like this.')

    con = 0
    with open(output_name, 'r', encoding='utf-8') as f:
        for line in f:
            print(line)
            con += 1
            if con == 5:
                break