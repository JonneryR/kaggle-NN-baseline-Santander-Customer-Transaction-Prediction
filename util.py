import pandas as pd
import numpy as np
import lightgbm as lgb
import time,datetime
import re,os
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split,RepeatedKFold
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LinearRegression,BayesianRidge
import gc
import warnings
import torch as tr
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data
warnings.filterwarnings('ignore')
device = tr.device('cuda')


def baseline_para_lgb(X_train,y_train,X_test,test,seed = 2019,round = 10000,n_folds = 5):
    res = test.copy()
    features = X_train.columns
    feature_importance_df = pd.DataFrame()
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
                                    n_estimators=round, subsample=0.9, colsample_bytree=0.77, 
                                    subsample_freq=1, learning_rate=0.05,random_state=1500, n_jobs=16, 
                                    min_child_weight=4, min_child_samples=30, min_split_gain=0)
    
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(X_train.shape[0])
    predictions_lgb = np.zeros(X_test.shape[0])

    for index, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold:", index+1)

        lgb_model.fit(X_train.iloc[trn_idx],y_train[trn_idx], 
                      eval_set=[(X_train.iloc[trn_idx],y_train[trn_idx]),
                                (X_train.iloc[val_idx],y_train[val_idx])],eval_metric = 'auc',
                                verbose=50, early_stopping_rounds=100)
        
        oof_lgb[val_idx] = lgb_model.predict(X_train.iloc[val_idx], num_iteration=lgb_model.best_iteration_)

        predictions_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_) / n_folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = lgb_model.feature_importances_
        fold_importance_df["fold"] = index + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    auc = roc_auc_score(y_train,oof_lgb)
    print('Training set and test set shape：',X_train.shape,X_test.shape) 
    print('AUC:', auc)
    res['target'] = predictions_lgb
    mean = res['target'].mean()
    print('mean:',mean)
    res.to_csv("./submit/lgb_base.csv", index=False,encoding='utf-8',sep=',')
    return res
      


def NNmodel(model_input,X_train,y_train,X_test,test,n_folds = 5,seed = 2019,num_epoch = 1):
    res = test.copy()
    learning_rate = 0.0001
    model = model_input
    criterion = nn.BCELoss()
    optimizer = tr.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 15000

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_NN = np.zeros(X_train.shape[0])
    predictions_NN = np.zeros(X_test.shape[0])

    test_tr = tr.from_numpy(X_test.values).float().to(device)
    test_dataset = torch.utils.data.TensorDataset(test_tr,tr.zeros(test_tr.size(0)).long())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=False)

    for index, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold:", index+1) 
        trn_data,trn_y = X_train.iloc[trn_idx],y_train[trn_idx]
        val_data,val_y = X_train.iloc[val_idx],y_train[val_idx]


        train_tr = tr.from_numpy(trn_data.values).float().to(device)
        label_tr = tr.from_numpy(trn_y).float().to(device)

        val_tr = tr.from_numpy(val_data.values).float().to(device)

        train_dataset = torch.utils.data.TensorDataset(train_tr,label_tr)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(val_tr,tr.zeros(val_tr.size(0)).long())
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=False)
        

        model.train()
        num_epochs = num_epoch
        for epoch in range(num_epochs):
            for i, (trains, labels) in enumerate(train_loader):
                trains = trains.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(trains)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, num_epoch, i+1, len(train_loader), loss.item()))

        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        ans = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                out = outputs.cpu()
                ans.extend(list(out.numpy()))
        ans = np.array(ans)
        oof_NN[val_idx] = ans

        ans = []
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = model(images)
                out = outputs.cpu()
                ans.extend(list(out.numpy()))
        ans = np.array(ans)
        predictions_NN += ans

    auc = roc_auc_score(y_train,oof_NN)
    print('Training set and test set shape：',X_train.shape,X_test.shape) 
    print('AUC:', auc)
    res['target'] = predictions_NN/n_folds
    mean = res['target'].mean()
    print('mean:',mean)
    res.to_csv("./submit/NN_base.csv", index=False,encoding='utf-8',sep=',')

