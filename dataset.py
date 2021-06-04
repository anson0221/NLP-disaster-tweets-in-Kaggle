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
                (keyword_0: list[int], sentence_0: list[int], target_0: int), 
                (keyword_1: list[int], sentence_1: list[int], target_1: int),
                ..., 
                (keyword_n: list[int], sentence_n: list[int], target_n: int),
            ]

        return values:
            * keyword: (batch_size, kw_seq_len)
            * sentence: (batch_size, sent_seq_len)
            * target: (batch_size)
        """

        

        keyword = []
        sentence = []
        target = torch.zeros(len(sample))
        i = 0
        for kw, sent, tgt in sample:
            keyword.append(kw)
            sentence.append(sent)
            target[i] = tgt
            i += 1

        keyword = pad_sequence(keyword, batch_first=True, padding_value=self.padding) 
        sentence = pad_sequence(sentence, batch_first=True, padding_value=self.padding)
        return keyword, sentence, target

        # elif self.mode=='test':
        #     for kw, sent, tgt in sample:
        #         keyword.append(kw)
        #         sentence.append(sent)

        #     keyword = pad_sequence(keyword, batch_first=True, padding_value=self.padding) 
        #     sentence = pad_sequence(sentence, batch_first=True, padding_value=self.padding)
        #     return keyword, sentence

        


class DisasTweet_ds(Dataset):
    def __init__(self, file_path: str='./data/train.csv', src_bert :str="vinai/bertweet-base", mode: str='train'):
        df = pd.read_csv(file_path)
        print(df.info())
        print(df.describe())
        self.keyword = df['keyword'].fillna('none')
        self.text = df['text'].fillna('none text.')

        self.tknzr_tweet = AutoTokenizer.from_pretrained(src_bert, use_fast=False)
        self.vocab = self.tknzr_tweet.get_vocab()
        self.pad_token = self.vocab[self.tknzr_tweet.pad_token]

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
        for sent in self.text:
            sent = html.sub(repl=r'', string=sent) # remove URL

            # remove all stop-words
            for s in self.stop_words:
                sent = re.sub(pattern=r'\s'+s+r'\s+', repl=r' ', string=sent)
            
    def __getitem__(self, idx):
        kw = torch.tensor(self.tknzr_tweet.encode('['+self.keyword[idx]+']: '))
        sent = self.tknzr_tweet.encode(self.text[idx])
        sent = torch.tensor([wp for wp in sent if wp not in self.punc]) # remove all punctuation

        if self.mode=='train':
            tgt = torch.tensor(self.target[idx])
            return kw, sent, tgt
        elif self.mode=='test':
            return kw, sent

    def __len__(self):
        return len(self.text)
