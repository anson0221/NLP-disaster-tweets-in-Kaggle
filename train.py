from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from torch import optim
from dataset import DisasTweet_ds, collater
from tqdm import tqdm
import os
import sys

def train(
            from_ck_point :bool,
            model_path :str,
            optimizer__ :str='SGD',
            criterion=nn.NLLLoss(),
            source_bert="vinai/bertweet-base",
            batch_size_=64, 
            epochs=20,
            clip=1,
            learning_rate=0.009, 
            device='cpu'
):
    # dataset
    dataset_ = DisasTweet_ds(src_bert=source_bert)
    collate_fn_ = collater(pad_value=dataset_.pad_token)
    train_loader = DataLoader(dataset=dataset_, batch_size=batch_size_, collate_fn=collate_fn_, shuffle=True, drop_last=True)