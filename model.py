import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class sentencesVec(nn.Module):
    def __init__(self, bert: str="vinai/bertweet-base", layer_index: int=-2):
        super(sentencesVec, self).__init__()
        self.bert_tweet = AutoModel.from_pretrained(bert, output_hidden_states=True)
        self.idx = layer_index

    def forward(self, sent):
        """
        transform sentences to vector by BERT
            * sent: (batch_size, seq_len)
        """

        # object : tuple = (the output of the embeddings, the output of each layer)
        object = self.bert_tweet(sent)

        hd_states = object.hidden_states[self.idx] # (batch, seq_len, hidden_dim)

        return hd_states # (batch_size, seq_len, hidden_dim)

class Classifier_bert(nn.Module):
    def __init__(
        self, 
        src_bert: str="vinai/bertweet-base", 
        bert_layerChoice: int=-2, 
        MAX_SEQ_LEN: int=150, 
        attn_num :int=2, 
        out_dim: int=2
    ):
        super(Classifier_bert, self).__init__()
        self.sourceBert = src_bert
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.attentionNum = attn_num
        self.outNum = out_dim
    
        self.cvtr_layer = sentencesVec(bert=self.sourceBert, layer_index=bert_layerChoice)

        # Attention
        self.d_sqrt = 768**0.5
        self.Q = nn.Linear(768, 768)
        self.K = nn.Linear(768, 768)
        self.V = nn.Linear(768, 768)

        # self.clsfr = nn.Sequential(
        #     nn.Linear(768, 384),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(384, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(64, 16),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(16, 8),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(8, self.outNum, bias=False),
        #     nn.LeakyReLU(negative_slope=0.1)
        # )

        self.clsfr = nn.Sequential(
            nn.Linear(768, 16),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(16, self.outNum, bias=False),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # output
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        """
        text: (batch_size, seq_len)
        """
        # print('0: ', end='')
        # print(str(text))

        # get contextualized embedding
        sentVec = self.cvtr_layer(text) # sentVec: (batch_size, seq_len, 768)
        # print('1: ', end='')
        # print(sentVec)

        # self-attention
        # for i in range(self.attentionNum):

        #     q = self.Q(sentVec) # q: (batch_size, seq_len, 768)
        #     k = self.K(sentVec) # k: (batch_size, seq_len, 768)
        #     v = self.V(sentVec) # v: (batch_size, seq_len, 768)

        #     a = torch.bmm(q, k.permute(0, 2, 1))/self.d_sqrt # a: (batch_size, seq_len, seq_len)
        #     print('4_a: ', end='')
        #     print(a)
        #     weight = F.softmax(a, dim=2) # weight: (batch_size, seq_len, seq_len)
        #     print('4_weight: ', end='')
        #     print(weight)
        #     sentVec = torch.bmm(weight, v) # newVec: (batch_size, seq_len, 768)
        #     print('4_sentVec: ', end='')
        #     print(sentVec)


        sentVec = sentVec.mean(dim=1).unsqueeze(1) # sentVec: (batch_size, 1, 768)
        # print('2: ', end='')
        # print(sentVec)


        # classifier
        sentVec = self.clsfr(sentVec) # newVec: (batch_size, 1, self.outNum)
        # print('5: ', end='')
        # print(sentVec)
        sentVec = sentVec.squeeze(1) # newVec: (batch_size, self.outNum)
        # print('6: ', end='')
        # print(sentVec)
        output = self.logSoftmax(sentVec) # output: (batch_size, 2)
        # print('7: ', end='')
        # print(output)

        return output 
