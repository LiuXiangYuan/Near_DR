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

    con = 0
    with open(collection_name, 'r', encoding='utf-8') as f, open(output_name, 'w', encoding='utf-8') as fw:
        for line in f:
            info = line.strip().split('\t')
            if len(info) != 4:
                continue
            else:
                docid, url, title, body = info[0], info[1], info[2], info[3]
                con += 1

            fw.write(json.dumps(dict(id=str(docid), contents='\n'.join([url.strip(), title.strip(), body.strip()]))) + '\n')

    print('all dataset :', con)
    
    print('json format done. format example like this.')

    con = 0
    with open(output_name, 'r', encoding='utf-8') as f:
        for line in f:
            print(line)
            con += 1
            if con == 5:
                break