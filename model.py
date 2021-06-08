import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class sentencesVec(nn.Module):
    def __init__(self, bert: str="vinai/bertweet-base", layer_index: int=-2):
        super(sentencesVec, self).__init__()
        self.bert_tweet = AutoModel.from_pretrained(bert, output_hidden_states=True)
        self.idx = layer_index

    def forward(self, text):
        """
        transform sentences to vector by BERT
            text: (batch_size, seq_len)
        """

        # object : tuple = (the output of the embeddings, the output of each layer)
        object = self.bert_tweet(text)

        hd_states = object.hidden_states[self.idx] # (batch, seq_len, hidden_dim)

        return hd_states # (batch_size, seq_len, hidden_dim)

class Classifier_bert(nn.Module):
    def __init__(
        self, 
        src_bert: str="vinai/bertweet-base", 
        bert_layerChoice: int=-2, 
        attn_num :int=3, 
        out_dim: int=2
    ):
        super(Classifier_bert, self).__init__()
        self.sourceBert = src_bert
        self.attentionNum = attn_num
        self.outNum = out_dim
    
        self.cvtr_layer = sentencesVec(bert=self.sourceBert, layer_index=bert_layerChoice)

        # Attention
        self.d_sqrt = 768**0.5
        self.Q = nn.Linear(768, 768)
        self.K = nn.Linear(768, 768)
        self.V = nn.Linear(768, 768)

        self.clsfr = nn.Sequential(
            nn.Linear(768, 384),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(384, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(8, self.outNum),
            nn.LeakyReLU(negative_slope=0.04)
        )


        # output
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        """
        text: (batch_size, seq_len)
        """

        # contextualized embedding
        sentVec = self.cvtr_layer(text) # sentVec: (batch_size, seq_len, 768)
        
        # self-attention
        for i in range(self.attentionNum):

            q = self.Q(sentVec) # q: (batch_size, seq_len, 768)
            k = self.K(sentVec) # k: (batch_size, seq_len, 768)
            v = self.V(sentVec) # v: (batch_size, seq_len, 768)

            a = torch.bmm(q, k.permute(0, 2, 1))/self.d_sqrt # a: (batch_size, seq_len, seq_len)
            
            weight = F.softmax(a, dim=2) # weight: (batch_size, seq_len, seq_len)
            
            sentVec = torch.bmm(weight, v) # newVec: (batch_size, seq_len, 768)

        # sentence embedding
        sentVec = sentVec.mean(dim=1).unsqueeze(1) # sentVec: (batch_size, 1, 768)

        # classifier
        sentVec = self.clsfr(sentVec) # newVec: (batch_size, 1, self.outNum)
        sentVec = sentVec.squeeze(1) # newVec: (batch_size, self.outNum)
        output = self.logSoftmax(sentVec) # output: (batch_size, 2)

        return output 
