import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from transformers import AutoModel


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

        self.tweet_extractor = nn.Sequential(
            nn.Linear(768, 384),
            nn.Tanh(),
            nn.Linear(384, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        # Attention
        self.Q = nn.Linear(16, 16)
        self.K = nn.Linear(16, 16)
        self.V = nn.Linear(16, 16)

        self.clsfr = nn.Sequential(
            nn.Linear(self.MAX_SEQ_LEN, 128),
            nn.Tanh(),
            nn.Linear(128, 64), 
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )

        # output
        self.out = nn.Linear(16, self.outNum)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        """
        text: (batch_size, MAX_SEQ_LEN)
        """

        # get contextualized embedding
        sentVec = self.cvtr_layer(text) # sentVec: (batch_size, MAX_SEQ_LEN, 768)

        # compression
        newVec = self.tweet_extractor(sentVec) # newVec: (batch_size, MAX_SEQ_LEN, 16)

        # self-attention
        for i in range(self.attentionNum):
            q = self.Q(newVec) # q: (batch_size, MAX_SEQ_LEN, 16)
            k = self.K(newVec) # k: (batch_size, MAX_SEQ_LEN, 16)
            v = self.V(newVec) # v: (batch_size, MAX_SEQ_LEN, 16)

            A = torch.tanh(torch.bmm(q, k.permute(0, 2, 1))) # A: (batch_size, MAX_SEQ_LEN, seq_len)
            newVec = torch.bmm(A, v) # newVec: (batch_size, MAX_SEQ_LEN, 16)

        # classifier
        newVec = newVec.permute(0, 2, 1) # newVec: (batch_size, 16, MAX_SEQ_LEN)
        newVec = self.clsfr(newVec) # newVec: (batch_size, 16, 1)
        newVec = newVec.squeeze(2) # newVec: (batch_size, 16)
        output = self.logSoftmax(self.out(newVec)) # output: (batch_size, 2)

        return output 
