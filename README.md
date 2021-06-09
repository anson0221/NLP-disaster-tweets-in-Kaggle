# NLP-disaster-tweets-in-Kaggle
## Kaggle Competition
* 此模型結構在test.csv的最佳F1-score
    * 0.81642
        * BERT: 
            * 'vinai/bertweet-base'
        * attn_layerNum = 2
        * batch_size = 64
        * lr = 7e-4
        * epochs: 約300
## 使用方法
* 訓練
    * 請先至train.py修改欲實驗之參數
    * python3 train.py [model_name]  
* 預測
    * predict on the test.csv
        * python3 inference.py [model_name] predict
    * test on the train.csv
        * python3 inference.py [model_name] test