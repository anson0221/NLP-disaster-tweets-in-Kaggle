import pandas as pd
from torch.utils.data.dataset import Dataset
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
                (keyword_0: str, len_of_keyword_0, sentence_0: list[int], target_0: int), 
                (keyword_1: str, len_of_keyword_1, sentence_1: list[int], target_1: int),
                ..., 
                (keyword_n: str, len_of_keyword_n, sentence_n: list[int], target_n: int),
            ]

        return values:
            * keyword: (batch_size)
            * sentence: (batch_size, seq_len)
            * target: (batch_size)
        """

        keyword = []
        len_kw = []
        sentence = []
        target = []
        for kw, len_of_kw, sent, tgt in sample:
            keyword.append(kw)
            len_kw.append(len_of_kw)
            sentence.append(sent)
            target.append(tgt)

        keyword = pad_sequence(keyword, batch_first=True, padding_value=self.padding)
        sentence = pad_sequence(sentence, batch_first=True, padding_value=self.padding)
        target = pad_sequence(sentence, batch_first=True)

        return keyword, len_kw, sentence, target


class DisasTweet_ds(Dataset):
    def __init__(self, train_file_path: str='./data/train.csv', src_bert :str="vinai/bertweet-base"):
        df = pd.read_csv(train_file_path)

        self.keyword = df['keyword'].fillna('none')
        self.text = df['text'].fillna('none text.')
        self.target = df['target']

        self.tknzr_tweet = AutoTokenizer.from_pretrained(src_bert, use_fast=False)
        self.pad_token = self.tknzr_tweet.get_vocab()[self.tknzr_tweet.pad_token]

    def __getitem__(self, idx):
        kw = torch.tensor(self.tknzr_tweet.encode(self.keyword[idx]+':'))
        kw_len = kw.size()[0]
        sent = torch.tensor(self.tknzr_tweet.encode(self.text[idx]))
        tgt = torch.tensor(self.target[idx])

        return kw, kw_len, sent, tgt

    def __len__(self):
        return len(self.text)
