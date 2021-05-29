import pandas as pd
from torch.utils.data.dataset import Dataset
import re
import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

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

        batch_size = len(sample)
        
        keyword = []
        sentence = []
        target = torch.zeros(batch_size)
        i = 0
        for kw, sent, tgt in sample:
            keyword.append(kw)
            sentence.append(sent)
            target[i] = tgt
            i += 1

        keyword = pad_sequence(keyword, batch_first=True, padding_value=self.padding) 
        sentence = pad_sequence(sentence, batch_first=True, padding_value=self.padding)

        return keyword, sentence, target


class DisasTweet_ds(Dataset):
    def __init__(self, train_file_path: str='./data/train.csv', src_bert :str="vinai/bertweet-base"):
        df = pd.read_csv(train_file_path)

        self.keyword = df['keyword'].fillna('none')
        self.text = df['text'].fillna('none text.')
        self.target = df['target']

        self._preprocess_text()

        self.tknzr_tweet = AutoTokenizer.from_pretrained(src_bert, use_fast=False)
        self.pad_token = self.tknzr_tweet.get_vocab()[self.tknzr_tweet.pad_token]
    
    def _preprocess_text(self):
        html = re.compile(r'((http)|(https))://\S+') # \S: all non-space symbols
        for sent in self.text:
            html.sub(repl=r'', string=sent)

    def __getitem__(self, idx):
        kw = torch.tensor(self.tknzr_tweet.encode('['+self.keyword[idx]+']:'))
        sent = torch.tensor(self.tknzr_tweet.encode(self.text[idx]))
        tgt = torch.tensor(self.target[idx])

        return kw, sent, tgt

    def __len__(self):
        return len(self.text)
