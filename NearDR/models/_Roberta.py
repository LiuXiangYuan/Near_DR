import torch.nn as nn

from ._models import BaseModelDot
from .model_forward import inbatch_train, randneg_train
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel


class RobertaDot(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        config.return_dict = False
        self.roberta = RobertaModel(config=config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1


class RobertaDot_InBatch(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask=None,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)


class RobertaDot_Rand(RobertaDot):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask=None,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             hard_pair_mask)
