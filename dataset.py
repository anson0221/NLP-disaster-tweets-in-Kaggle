import pandas as pd
from torch.utils.data.dataset import Dataset
import re
import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.corpus import stopwords
import string

class collater():
    def __init__(self, pad_value):
        self.padding = pad_value

    def __call__(self, sample :list):
        """
        sample (a list for a batch of samples) : 
            [
                (sentence_0: list[int], target_0: int), 
                (sentence_1: list[int], target_1: int),
                ..., 
                (sentence_n: list[int], target_n: int),
            ]

        return values:
            * sentence: (batch_size, sent_seq_len)
            * target: (batch_size)
        """

        

        sentence = []
        target = torch.zeros(len(sample))
        i = 0
        for sent, tgt in sample:
            sentence.append(sent)
            target[i] = tgt
            i += 1

        sentence = pad_sequence(sentence, batch_first=True, padding_value=self.padding)
        return sentence, target
    

class DisasTweet_ds(Dataset):
    def __init__(self, file_path: str='./data/train.csv', src_bert :str="vinai/bertweet-base", mode: str='train'):
        self.MAX_SEQ_LEN = 150

        df = pd.read_csv(file_path)
        print(df.info())
        print(df.describe())
        self.keyword = df['keyword'].fillna('none')
        self.text = df['text'].fillna('none text.')

        self.tknzr_tweet = AutoTokenizer.from_pretrained(src_bert, use_fast=False)
        self.vocab = self.tknzr_tweet.get_vocab()
        self.pad_token_id = self.vocab[self.tknzr_tweet.pad_token]

        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')

        self.punc = []
        for p in string.punctuation:
            self.punc.append(self.vocab[p])

        self.mode = mode
        if self.mode=='train':
            self.target = df['target'].fillna(0)
        elif self.mode=='test':
            self.id = df['id']

        self._preprocess_text()


    def _preprocess_text(self):
        html = re.compile(r'((http)|(https))://\S+') # \S: all non-space symbols
        for i in range(len(self.text)):
            self.text[i] = html.sub(repl=r'', string=self.text[i]) # remove URL

            # remove all stop-words
            for s in self.stop_words:
                self.text[i] = re.sub(pattern=r'\s'+s+r'\s+', repl=r' ', string=self.text[i])

    def _padding(self, sent):
        if sent.shape[0]>=self.MAX_SEQ_LEN:
            return sent[:self.MAX_SEQ_LEN]
        else:
            apd = torch.zeros(self.MAX_SEQ_LEN-sent.shape[0], dtype=torch.int32)
            for i in range(apd.shape[0]):
                apd[i] = self.pad_token_id
            return torch.cat((sent, apd), dim=0)
            
    def __getitem__(self, idx):
        kw = torch.tensor(self.tknzr_tweet.encode('['+self.keyword[idx]+']: '))
        txt = self.tknzr_tweet.encode(self.text[idx])
        txt = torch.tensor([wp for wp in txt if wp not in self.punc]) # remove all punctuation
        sent = self._padding(torch.cat((kw, txt), dim=0))

        if self.mode=='train':
            tgt = torch.tensor(self.target[idx])
            return sent, tgt
        elif self.mode=='test':
            return sent

    def __len__(self):
        return len(self.text)
