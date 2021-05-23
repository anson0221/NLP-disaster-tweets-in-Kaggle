import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class sentencesVec(nn.Module):
    def __init__(self, bert: str, layer_index: int=-2):
        self.bert_tweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.idx = layer_index

    def forward(self, kw, sent):
        """
        transform sentences to vector by BERT
            * keyword: (batch_size)
            * sentence: (batch_size, seq_len)
        """
        self.idx = -2 # get the second-to-last layer output of BERT


        # object : tuple = (the output of the embeddings, the output of each layer)
        object = self.bert_tweet(sent)

        hd_states = object.hidden_states[self.idx] # (batch, seq_len, hidden_dim)

        return hd_states # (batch, seq_len, hidden_dim)

class Classifier_bert(nn.Module):
    def __init__(self, kw_embd_dim: int=10, src_bert: str="vinai/bertweet-base", attn_num :int=2, out_dim: int=2):
        self.kw_embedding_dim = kw_embd_dim
        self.sourceBert = src_bert
        self.attentionNum = attn_num
        self.outNum = out_dim
    
        self.cvtr_layer = sentencesVec(self.sourceBert)

        self.tweet_extractor = nn.Sequential(
            nn.Linear(768, 384),
            nn.Tanh(),
            nn.Linear(384, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid()
        )

        self.kw_encoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.Tanh()
        )

        # Attention
        self.Q = nn.Linear(16, 16)
        self.K = nn.Linear(16, 16)
        self.V = nn.Linear(16, 16)

        # output
        self.reduce_dim = nn.Linear(16, 1, bias=False)
        self.out = nn.Linear(1, self.outNum)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, text, keyword, kw_len):
        """
        text: (batch_size)
        keyword: (batch_size)
        """

        # get contextualized embedding
        batch_size = text.shape[0]
        sentVec = self.cvtr_layer(keyword + text) # sentVec: (batch_size, total_seq_len, 768)

        tweet_vec = sentVec[:, kw_len:, :] # tweet_vec: (batch_size, tweet_seq_len, 768)
        tweet_vec = self.tweet_extractor(tweet_vec) # tweet_vec: (batch_size, tweet_seq_len, 16)

        kwVec = sentVec[:, :kw_len, :] # kwVec: (batch_size, keyword_seq_len, 768)
        kwVec = self.kw_encoder(kwVec) # kwVec: (batch_size, keyword_seq_len, 16)

        newVec = torch.cat((kwVec, tweet_vec), dim=1) # newVec: (batch_size, total_seq_len, 16)

        # self-attention
        for i in range(self.attentionNum):
            q = self.Q(newVec) # q: (batch_size, total_seq_len, 16)
            k = self.K(newVec) # k: (batch_size, total_seq_len, 16)
            v = self.V(newVec) # v: (batch_size, total_seq_len, 16)

            A = torch.tanh(torch.bmm(q, k.permute(0, 2, 1))) # A: (batch_size, total_seq_len, total_seq_len)
            newVec = torch.bmm(A, v) # newVec: (batch_size, total_seq_len, 16)

        # output
        newVec = self.reduce_dim(newVec) # newVec: (batch_size, total_seq_len, 1)
        newVec = newVec.reshape(batch_size, 1, -1) # newVec: (batch_size, 1, total_seq_len)
        newVec = newVec.squeeze(1) # newVec: (batch_size, total_seq_len)
        newVec = torch.sum(newVec, dim=1).unsqueeze(1) # newVec: (batch_size, 1)
        output = self.logSoftmax(self.out(newVec)) 

        return output # output: (batch_size, 2)
