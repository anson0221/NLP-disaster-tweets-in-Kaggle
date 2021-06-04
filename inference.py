import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import Classifier_bert
from dataset import DisasTweet_ds, collater
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

if __name__ == '__main__':
    path = os.path.join('./experiment/', sys.argv[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_bert = "vinai/bertweet-base"
    model = Classifier_bert(src_bert=source_bert)

    check_point = torch.load(path, map_location=device)
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()

    predict(model=model, test_filePath='./data/test.csv', source_bert=source_bert, out_file='./data/submission.csv')
