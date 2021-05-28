import argparse
import json
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer
from pyserini.dsearch import AnceEncoder


@torch.no_grad()
def encode_passage(texts, tokenizer, model, name, device='cuda:0'):
    max_length = 512  # hardcode for now
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding='longest',
        truncation=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    inputs.to(device)
    if 'ance' in name:
        embeddings = model(inputs["input_ids"]).detach().cpu().numpy()
    else:
        embeddings = model(**inputs)[0][:, 0, :].detach().cpu().numpy()
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    parser.add_argument('--dimension', type=int, help='dimension of passage embeddings', required=False, default=768)
    parser.add_argument('--corpus', type=str,
                        help='directory that contains corpus files to be encoded, in jsonl format.', required=True)
    parser.add_argument('--index', type=str, help='directory to store brute force index of corpus', required=True)
    parser.add_argument('--batch', type=int, help='batch size', default=8)
    parser.add_argument('--device', type=str, help='device cpu or cuda [cuda:0, cuda:1...]', default='cuda:0')
    args = parser.parse_args()

    if 'ance' in args.encoder:
        tokenizer = RobertaTokenizer.from_pretrained(args.encoder)
        model = AnceEncoder.from_pretrained(args.encoder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        model = AutoModel.from_pretrained(args.encoder)
    model_name = args.encoder
    model.to(args.device)

    index = faiss.IndexFlatIP(args.dimension)

    if not os.path.exists(args.index):
        os.mkdir(args.index)

    texts = []
    with open(os.path.join(args.index, 'docid'), 'w') as id_file:
        for file in sorted(os.listdir(args.corpus)):
            file = os.path.join(args.corpus, file)
            if file.endswith('json') or file.endswith('jsonl'):
                print(f'Loading {file}')
                with open(file, 'r') as corpus:
                    for idx, line in enumerate(tqdm(corpus.readlines())):
                        info = json.loads(line)
                        docid = info['id']
                        text = info['contents']
                        id_file.write(f'{docid}\n')
                        url, title, text = text.split('\n')
                        text = f"{url} <sep> {title} <sep> {text}"
                        texts.append(text.lower())
    for idx in tqdm(range(0, len(texts), args.batch)):
        text_batch = texts[idx: idx+args.batch]
        embeddings = encode_passage(text_batch, tokenizer, model, model_name, args.device)
        index.add(np.array(embeddings))
    faiss.write_index(index, os.path.join(args.index, 'index'))
