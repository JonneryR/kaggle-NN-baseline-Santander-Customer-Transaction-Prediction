import pandas as pd
import numpy as np
import lightgbm as lgb
import time,datetime
import catboost as cgb
import xgboost as xgb
from util import *
device = tr.device('cuda')

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')


target = df_train['target'].values
del df_train['target']

sub = df_test[['ID_code']]
feature_lists = [i for i in df_train.columns if i not in ['ID_code']]

train_set = df_train[feature_lists]
test_set = df_test[feature_lists]

class MLP(nn.Module):
    def __init__(self,input_size):
        super(MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(200,100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(100,50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(50,1),
        )
    def forward(self,x):
        output = F.sigmoid(self.fc(x)).view(-1)
        return output

NN = MLP(train_set.shape[1]).to(device)
NNmodel(NN,train_set,target,test_set,sub,num_epoch=100)
