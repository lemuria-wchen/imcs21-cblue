import torch
import torch.nn as nn
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF


class SeqClassifier(nn.Module):
    """序列标注分类器"""
    def __init__(self, input_dim, num_seq_labels, dropout_rate=0.):
        super(SeqClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_seq_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class NERBERT(BertPreTrainedModel):
    """NERBERT模型"""
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config, args, seq_label_lst):
        super(NERBERT, self).__init__(config)
        self.args = args
        self.num_seq_labels = len(seq_label_lst)
        self.bert = BertModel(config=config)

        self.seq_classifier = SeqClassifier(config.hidden_size, self.num_seq_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_seq_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, seq_labels_ids):
        """前向传播"""
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        seq_logits = self.seq_classifier(sequence_output)

        total_loss = 0

        if seq_labels_ids is not None:
            if self.args.use_crf:
                seq_loss = self.crf(seq_logits, seq_labels_ids, mask=attention_mask.byte(), reduction='mean')
                seq_loss = -1 * seq_loss  # negative log-likelihood
            else:
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # pad部分不计算loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = seq_logits.view(-1, self.num_seq_labels)[active_loss]
                    active_labels = seq_labels_ids.view(-1)[active_loss]
                    seq_loss = seq_loss_fct(active_logits, active_labels)
                else:
                    seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_seq_labels), seq_labels_ids.view(-1))
            total_loss += self.args.seq_loss_coef * seq_loss

        outputs = ((seq_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
