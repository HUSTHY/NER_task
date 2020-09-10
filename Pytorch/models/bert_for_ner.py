import torch
import torch.nn as nn
from .crf import CRF
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertModel
from torch.nn import CrossEntropyLoss

class BertCrfOneStageForNer(BertPreTrainedModel):
    def __init__(self,config):
        super(BertCrfOneStageForNer,self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size,config.num_labels)
        self.crf = CRF(num_tags=config.num_labels,batch_first=True)
        self.init_weights()

    def forward(self,input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores

class BertOneStageForNer(BertPreTrainedModel):
    def __init__(self,config):
        super(BertOneStageForNer,self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,input_ids, attention_mask=None,labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)

        if labels is not  None:
            loss_fct = CrossEntropyLoss(ignore_index=0)

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits,active_labels)
            else:
                loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

class BertCrfTwoStageForNer(BertPreTrainedModel):
    def __init__(self,config):
        super(BertCrfTwoStageForNer,self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_bio = nn.Linear(config.hidden_size,config.bio_num_labels)
        self.cls_att = nn.Linear(config.hidden_size, config.att_num_labels)
        self.crf = CRF(num_tags=config.num_labels,batch_first=True)
        self.init_weights()

    def forward(self,input_ids, attention_mask=None,bio_labels=None,att_labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        bio_logits = self.cls_bio(sequence_output)
        att_logits = self.cls_att(sequence_output)
        outputs = (bio_logits,att_logits)
        if bio_labels is not None and att_labels is not None:
            bio_loss = self.crf(emissions = bio_logits, tags=bio_labels, mask=attention_mask)
            att_loss = self.crf(emissions = att_logits, tags=att_labels, mask=attention_mask)
            loss = bio_loss + att_loss
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores
