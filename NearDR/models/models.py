from ._Roberta import RobertaDot_InBatch, RobertaDot_Rand
from ._TinyBert import TinyBertDot_InBatch, TinyBertDot_Rand
from ._Distrilbert import DistillbertDot_InBatch, DistillbertDot_Rand


MODEL_MAP = {
    'roberta_in_batch': RobertaDot_InBatch,
    'roberta_rand': RobertaDot_Rand,
    'tinybert_in_batch': TinyBertDot_InBatch,
    'tinybert_rand': TinyBertDot_Rand,
    'distrilbert_in_batch': DistillbertDot_InBatch,
    'distrilbert_rand': DistillbertDot_Rand
}


def get_model_class(model_name_or_path, hard_neg=False, rand_neg=False):
    if 'roberta' in model_name_or_path:
        if rand_neg:
            return MODEL_MAP['roberta_rand']
        else:
            return MODEL_MAP['roberta_in_batch']

    if 'tiny' in model_name_or_path:
        if rand_neg:
            return MODEL_MAP['tinybert_rand']
        else:
            return MODEL_MAP['tinybert_in_batch']

    if 'distril' in model_name_or_path:
        if rand_neg:
            return MODEL_MAP['distrilbert_rand']
        else:
            return MODEL_MAP['distrilbert_in_batch']
