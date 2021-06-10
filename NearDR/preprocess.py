import os

from arguments import TokenizeArguments
from transformers import HfArgumentParser
from dataset.preprocessor.preprocess import preprocess


def main():
    parser = HfArgumentParser(TokenizeArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)

    if 'roberta' in args.model_name_or_path:
        args.is_roberta = True
    else:
        args.is_roberta = False

    preprocess(args)


if __name__ == '__main__':
    main()
