import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import Classifier_bert
from dataset import DisasTweet_ds
from transformers import AutoTokenizer
from tqdm import tqdm
import csv
import os
import sys

def predict(model: Classifier_bert, test_filePath: str, source_bert: str, out_file: str):
    dataset = DisasTweet_ds(file_path=test_filePath, src_bert=source_bert, mode='test')
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

    ans = []
    with torch.no_grad():
        for sent in tqdm(testloader):
            out = model(text=sent)
                
            prediction = out.argmax(1)
            ans.append(prediction[0])

    with open(out_file, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'target']) # head row

        for i in range(len(ans)):
            writer.writerow([dataset.id[i], ans[i].numpy()])

def test_on_training_set(model: Classifier_bert, train_filePath: str, source_bert: str, out_file: str):
    dataset = DisasTweet_ds(file_path=train_filePath, src_bert=source_bert, mode='train')
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

    with torch.no_grad():
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        
        TP_num = 0
        TN_num = 0
        FP_num = 0
        FN_num = 0

        output = []
        i = 0
        for sent, tgt in tqdm(testloader):
            out = model(text=sent)
            prediction = out.argmax(1)[0]

            # True
            if prediction==tgt:
                if prediction==0: # Negative
                    TN_num += 1
                else: # Positive
                    TP_num += 1
            # False
            else:
                output.append(i)
                if prediction==0: # Negative
                    FN_num += 1
                else: # Positive
                    FP_num += 1

            i += 1


        accuracy = (TN_num+TP_num)/(TN_num+TP_num+FN_num+FP_num)
        if (TP_num+FP_num)>0:
            precision = TP_num/(TP_num+FP_num)
            recall = TP_num/(TP_num+FN_num)
        if precision+recall>0:
            f1_score = 2*precision*recall/(precision+recall)

        print('Accuracy: '+str(accuracy))
        print('precision: '+str(precision))
        print('recall: '+str(recall))
        print('F1-score: '+str(f1_score))

        with open(out_file, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'keyword', 'text', 'target']) # head row

            for i in output:
                writer.writerow([dataset.id[i], dataset.keyword[i], dataset.ori_text[i], dataset.target[i]])


if __name__ == '__main__':
    path = os.path.join('./experiment/', sys.argv[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_bert = "vinai/bertweet-base"
    model = Classifier_bert(src_bert=source_bert)

    check_point = torch.load(path, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()

    if sys.argv[2]=='predict':
        predict(model=model, test_filePath='./data/test.csv', source_bert=source_bert, out_file='./data/submission.csv')
    elif sys.argv[2]=='test':
        test_on_training_set(model=model, train_filePath='./data/train.csv', source_bert=source_bert, out_file='wrong_prediction.csv')
