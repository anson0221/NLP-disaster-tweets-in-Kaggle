from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from torch import optim
from dataset import DisasTweet_ds, collater
from model import Classifier_bert
from tqdm import tqdm
import sys

def train(
            from_ck_point :bool,
            model_path :str,
            data_path :str='./data/trasin.csv',
            criterion=nn.NLLLoss(),
            source_bert='vinai/bertweet-base',
            choose_bert_layer_as_embd: int=-2,
            attentionNUM: int=2,
            tgt_categoryNum: int=2, 
            batch_size_=64, 
            epochs=20,
            clip=1,
            learning_rate=0.009, 
            device='cpu'
):
    # dataset
    dataset_ = DisasTweet_ds(file_path=data_path, src_bert=source_bert, mode='train')
    collate_fn_ = collater(pad_value=dataset_.pad_token_id)
    train_loader = DataLoader(dataset=dataset_, batch_size=batch_size_, collate_fn=collate_fn_, shuffle=True, drop_last=True)

    # model
    model = Classifier_bert(
        src_bert=source_bert, 
        bert_layerChoice=choose_bert_layer_as_embd,
        MAX_SEQ_LEN=dataset_.MAX_SEQ_LEN, 
        attn_num=attentionNUM, 
        out_dim=tgt_categoryNum
    ).to(device)

    if from_ck_point:
        check_point = torch.load(model_path, map_location=device)
        model.load_state_dict(check_point['model_state_dict'])
        BEST_LOSS = check_point['loss']
    else:
        BEST_LOSS = 999999

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    model.train()

    # training
    for epoch in range(epochs):
        epoch_loss = 0
        print()
        print('Epoch #'+str(epoch))
        for sentence, target in tqdm(train_loader):
            """
            * sentence: (batch_size, seq_len)
            * target: (batch_size)
            """

            sentence = sentence.to(device)
            target = target.to(device, dtype=torch.long)
            
            out = model(text=sentence)
            
            loss = 0
            optimizer.zero_grad()
            loss += criterion(out, target)

            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=clip) # clipping gradient
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= (len(train_loader))
        print('Loss : '+str(epoch_loss))

        if epoch_loss<BEST_LOSS:
            # save the model
            BEST_LOSS = epoch_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, model_path)


if __name__=='__main__':
    from_check_point = False
    modelPath = './experiment/'+sys.argv[1]
    dataPath = './data/train.csv'
    srcBERT = 'vinai/bertweet-base'
    bertlayer_idx = -2
    attn_layerNum = 2
    batchSize = 8
    epoch_num = 30
    lr = 0.00007
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        from_ck_point=from_check_point,
        model_path=modelPath,
        data_path=dataPath,
        source_bert=srcBERT,
        choose_bert_layer_as_embd=bertlayer_idx,
        attentionNUM= attn_layerNum,
        batch_size_=batchSize,
        epochs=epoch_num,
        learning_rate=lr,
        device=device_
    )