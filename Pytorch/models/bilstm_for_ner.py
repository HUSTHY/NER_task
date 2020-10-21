import torch
import torch.nn as nn
from .crf import CRF

class BiLstmCRFNer(nn.Module):
    """
    vocab_size:词典的大小
    embedding_size：sequence_length 可以是句子的长度也可以比句子长
    hidden_size：字向量的维度
    biso_num_labels：标注的labels
    bilstm后接一个分类器，然后再接一个CRF控制输出
    nn.Embedding:随机进行字向量构建
    """
    def __init__(self,vocab_size,embedding_size,hidden_size,biso_labels,drop_p):
        super(BiLstmCRFNer,self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size,embedding_size)

        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,batch_first=True,num_layers=2,dropout=drop_p,bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, len(biso_labels))
        self.crf = CRF(num_tags = len(biso_labels),batch_first=True)

    def forward(self,inputs_ids, input_mask,bio_labels=None):
        embs = self.embedding(inputs_ids)
        embs = embs*input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        features = self.classifier(seqence_output)
        outputs = (features,)
        if bio_labels is not None:
            loss = self.crf(emissions=features, tags=bio_labels, mask=input_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores

