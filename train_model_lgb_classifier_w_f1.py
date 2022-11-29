import joblib
import numpy as np
import pandas as pd
import gc
import time
import os
import sys
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV 
from optbinning import OptimalBinning
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
from DMS_202211.seed_everything import seed_everything


# """
# baseline model 
# """

# drop_features_vc2= ['ZN_MO_plus', 'MO_minus_MO_YEAR_max', 'FE_divide_FE_psudo_id2_YEAR_sum', 'FE_divide_FE_YEAR_COMPONENT_median', 'V40_minus_V40_YEAR_COMPONENT_min', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_max', 'CU_NI_logdiff', 'ZN_minus_ZN_YEAR_COMPONENT_median', 'ZN_ANONYMOUS_1_div', 'FE_minus_FE_psudo_id2_YEAR_COMPONENT_median', 'V40_minus_V40_psudo_id2_YEAR_COMPONENT_min', 'ANONYMOUS_1_V40_mul', 'FE_divide_FE_YEAR_median', 'V40_minus_V40_psudo_id2_median', 'PQINDEX_mul_ZN', 'CR_MN_div', 'ZN_psudo_id2_YEAR_max', 'H2O_divide_H2O_YEAR_max', 'CU_divide_CU_psudo_id2_YEAR_COMPONENT_sum', 'PQINDEX_NI_div', 'ZN_minus_ZN_psudo_id2_YEAR_COMPONENT_min', 'CU_NI_div', 'PQINDEX_divide_PQINDEX_psudo_id2_YEAR_COMPONENT_sum', 'ZN_psudo_id2_YEAR_sum', 'FE_NI_minus', 'PQINDEX_divide_PQINDEX_YEAR_COMPONENT_max', 'V40_MN_div', 'ANONYMOUS_1_V40_div', 'FE_divide_FE_YEAR_sum', 'ANONYMOUS_1_NI_plus', 'ZN_minus_ZN_YEAR_COMPONENT_min', 'PQINDEX_divide_PQINDEX_YEAR_sum', 'ZN_AG_minus', 'MN_divide_MN_YEAR_COMPONENT_sum', 'FE_divide_FE_YEAR_COMPONENT_sum', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_median', 'ANONYMOUS_1_MO_div', 'ANONYMOUS_2_divide_ANONYMOUS_2_psudo_id2_YEAR_median', 'PQINDEX_divide_PQINDEX_YEAR_max', 'FE_NI_mul', 'ZN_minus_ZN_psudo_id2_YEAR_COMPONENT_max', 'MO_divide_MO_YEAR_max', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_median', 'FE_divide_FE_YEAR_COMPONENT_max', 'ANONYMOUS_2_divide_ANONYMOUS_2_YEAR_COMPONENT_max', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_median', 'AG_divide_AG_YEAR_COMPONENT_sum', 'V40_ANONYMOUS_2_plus', 'ANONYMOUS_1_NI_div', 'V40_divide_V40_YEAR_sum', 'ZN_FE_mul', 'NI_mul_ZN', 'ANONYMOUS_2_psudo_id2_YEAR_median', 'PQINDEX_divide_PQINDEX_psudo_id2_median', 'PQINDEX_ANONYMOUS_2_plus', 'FE_divide_FE_YEAR_COMPONENT_min', 'PQINDEX_divide_PQINDEX_psudo_id2_YEAR_min', 'CU_NI_minus', 'V40_divide_V40_YEAR_median', 'V40_MO_div', 'PQINDEX_FE_div', 'V40_CO_div', 'ZN_ANONYMOUS_1_mul', 'ANONYMOUS_1_divide_ANONYMOUS_1_YEAR_COMPONENT_sum', 'ZN_minus_ZN_psudo_id2_YEAR_median', 'V40_divide_V40_psudo_id2_min', 'V40_psudo_id2_YEAR_COMPONENT_max', 'CR_divide_CR_YEAR_COMPONENT_max', 'V40_divide_V40_YEAR_min', 'ZN_divide_ZN_YEAR_COMPONENT_max', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_sum', 'ZN_ANONYMOUS_2_plus', 'ZN_minus_ZN_psudo_id2_max', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_min', 'FE_divide_FE_psudo_id2_median', 'ANONYMOUS_2_divide_ANONYMOUS_2_psudo_id2_YEAR_COMPONENT_median', 'PQINDEX_minus_PQINDEX_psudo_id2_YEAR_COMPONENT_max', 'PQINDEX_divide_PQINDEX_YEAR_COMPONENT_sum', 'V_PQINDEX_minus', 'PQINDEX_divide_PQINDEX_psudo_id2_YEAR_median', 'PQINDEX_minus_PQINDEX_YEAR_COMPONENT_max', 'ANONYMOUS_2_YEAR_COMPONENT_sum', 'ANONYMOUS_1_H2O_plus', 'ANONYMOUS_2_psudo_id2_YEAR_sum', 'ANONYMOUS_1_psudo_id2_YEAR_max', 'CR_ANONYMOUS_1_mul', 'ZN_PQINDEX_div', 'V40_divide_V40_psudo_id2_YEAR_min', 'MO_psudo_id2_YEAR_max', 'V40_CU_div', 'MN_divide_MN_psudo_id2_sum', 'TI_V40_plus', 'ANONYMOUS_1_PQINDEX_plus', 'PQINDEX_minus_PQINDEX_YEAR_max', 'ANONYMOUS_1_CO_minus', 'V40_FE_plus', 'ANONYMOUS_1_log1', 'V40_PQINDEX_div', 'FE_mul_ZN', 'ZN_divide_ZN_psudo_id2_YEAR_COMPONENT_max', 'ZN_ANONYMOUS_2_div', 'CR_CU_minus', 'CR_divide_CR_psudo_id2_YEAR_COMPONENT_sum', 'MN_divide_MN_psudo_id2_YEAR_max', 'V40_CU_minus', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_YEAR_median', 'ANONYMOUS_1_CU_plus', 'V40_divide_V40_YEAR_COMPONENT_min', 'FE_NI_div', 'CO_FE_minus', 'ANONYMOUS_1_MN_plus', 'CU_divide_CU_YEAR_COMPONENT_sum', 'ZN_minus_ZN_psudo_id2_YEAR_max', 'ZN_divide_ZN_psudo_id2_max', 'FE_divide_FE_psudo_id2_YEAR_COMPONENT_median', 'CR_NI_minus', 'MO_divide_MO_psudo_id2_YEAR_COMPONENT_max', 'ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_sum', 'ANONYMOUS_2_divide_ANONYMOUS_2_psudo_id2_YEAR_COMPONENT_sum', 'V_CU_plus', 'MO_minus_MO_YEAR_COMPONENT_median', 'CU_minus_CU_psudo_id2_YEAR_COMPONENT_median', 'FE_psudo_id2_YEAR_median', 'ANONYMOUS_1_CU_div', 'ZN_CR_div', 'PQINDEX_psudo_id2_YEAR_COMPONENT_min', 'NI_divide_NI_psudo_id2_YEAR_COMPONENT_sum', 'ANONYMOUS_1_MN_minus', 'ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_median', 'CR_ANONYMOUS_2_minus', 'FE_divide_FE_psudo_id2_YEAR_max', 'ANONYMOUS_1_divide_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_sum', 'ANONYMOUS_1_FE_plus', 'FE_divide_FE_psudo_id2_YEAR_COMPONENT_sum', 'ZN_minus_ZN_psudo_id2_YEAR_COMPONENT_median', 'cluster_no_by_ANONYMOUS_2_mean', 'ANONYMOUS_1_TI_minus', 'MO_divide_MO_YEAR_COMPONENT_sum', 'ANONYMOUS_2_psudo_id2_YEAR_max', 'ANONYMOUS_2_NI_mul', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_max', 'ZN_psudo_id2_YEAR_median', 'V40_divide_V40_psudo_id2_YEAR_COMPONENT_median']
# drop_features_vc3= ['ZN_MO_plus', 'MO_minus_MO_YEAR_max', 'FE_divide_FE_psudo_id2_YEAR_sum', 'FE_divide_FE_YEAR_COMPONENT_median', 'V40_minus_V40_YEAR_COMPONENT_min', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_max', 'CU_NI_logdiff', 'ZN_minus_ZN_YEAR_COMPONENT_median', 'ZN_ANONYMOUS_1_div', 'FE_minus_FE_psudo_id2_YEAR_COMPONENT_median', 'V40_minus_V40_psudo_id2_YEAR_COMPONENT_min', 'ANONYMOUS_1_V40_mul', 'FE_divide_FE_YEAR_median', 'V40_minus_V40_psudo_id2_median', 'PQINDEX_mul_ZN', 'CR_MN_div', 'ZN_psudo_id2_YEAR_max', 'H2O_divide_H2O_YEAR_max', 'CU_divide_CU_psudo_id2_YEAR_COMPONENT_sum', 'PQINDEX_NI_div', 'ZN_minus_ZN_psudo_id2_YEAR_COMPONENT_min', 'CU_NI_div', 'PQINDEX_divide_PQINDEX_psudo_id2_YEAR_COMPONENT_sum', 'ZN_psudo_id2_YEAR_sum', 'FE_NI_minus', 'PQINDEX_divide_PQINDEX_YEAR_COMPONENT_max', 'V40_MN_div', 'ANONYMOUS_1_V40_div', 'FE_divide_FE_YEAR_sum', 'ANONYMOUS_1_NI_plus', 'ZN_minus_ZN_YEAR_COMPONENT_min', 'PQINDEX_divide_PQINDEX_YEAR_sum', 'ZN_AG_minus', 'MN_divide_MN_YEAR_COMPONENT_sum', 'FE_divide_FE_YEAR_COMPONENT_sum', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_median', 'ANONYMOUS_1_MO_div', 'ANONYMOUS_2_divide_ANONYMOUS_2_psudo_id2_YEAR_median']
# drop_features_vc4= ['ZN_MO_plus', 'MO_minus_MO_YEAR_max', 'FE_divide_FE_psudo_id2_YEAR_sum', 'FE_divide_FE_YEAR_COMPONENT_median', 'V40_minus_V40_YEAR_COMPONENT_min', 'ANONYMOUS_1_minus_ANONYMOUS_1_psudo_id2_YEAR_COMPONENT_max', 'CU_NI_logdiff', 'ZN_minus_ZN_YEAR_COMPONENT_median']

# print(len(drop_features_vc2))

# params = {
    
#     'task': 'train', 
#     'boosting_type': 'gbdt', 
#     'objective': 'binary', 
#     'metric': 'auc', 
    
#     'num_iteration': 5000, 
#     'early_stopping_rounds': 200,
#     'verbose': -1 ,    
    
#     'learning_rate': 0.07398,
#     'max_depth': 4,
#     'colsample_bytree': 0.4028,
#     'subsample': 0.4278,
#     'min_child_samples': 26,
#     'min_child_weight': 0.6138,
#     'min_split_gain': 0.7354,
#     'num_leaves': 62,
#     'reg_alpha': 0.2889,
#     'reg_lambda': 7.875
    
# }

# train_model_lgb_classifier_w_f1(train,test,params,True,5,drop_features_vc2,seed_num=11)

def train_model_lgb_classifier_w_f1(train,test,params,stratified,num_folds,drop_features,seed_num):
    
    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))
    
    seed_everything(seed_num)
    
    # Divide in training/validation and test data
    train_df = train.copy()
    test_df = test.copy()

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        test_df[each] = encoder.fit_transform(test_df[each])

    # set training options
    stratified = stratified
    num_folds = num_folds

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed_num)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed_num)

    # Create arrays and dataframes to store results
    oof_preds_lgb = np.zeros(train_df.shape[0])
    sub_preds_lgb = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
    
    params['random_state'] = seed_num
    
    def f1_eval(yhat,data):
    
        y = data.get_label()
        thred = 0.1636 # 0.125~0.13
        pred = np.where(yhat>thred,1,0)

        return 'f1', f1_score(y, pred, average='macro'),True    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
        
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]

        categorical_feature = []        
        lgb_train = lgb.Dataset(data=train_x, label=train_y, categorical_feature=categorical_feature)
        lgb_valid = lgb.Dataset(data=valid_x, label=valid_y, reference=lgb_train, categorical_feature=categorical_feature)
        
        with warnings.catch_warnings():
            
            warnings.filterwarnings('ignore')  
            
            clf = lgb.train(

                params,                
                train_set = lgb_train,
                valid_sets = [lgb_train,lgb_valid],
                feval = f1_eval,
                verbose_eval = 200,
                categorical_feature = categorical_feature
            )

        oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds_lgb += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["importance_split"] = clf.feature_importance(importance_type='split')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb))

    # Write submission file and plot feature importance
    test_df['Y_LABEL_lgb'] = sub_preds_lgb

    # vi
    # print('-'*50)
    display(feature_importance_df.groupby(['feature'])['importance_gain'].sum().sort_values(ascending=False).head())
    # print('-'*50)
    # display_importances(feature_importance_df)
    
    # train auc
    oof_auc = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)
    
    # find the best thred for f1-score
    f1_score_df = pd.DataFrame()
    for thred in [i/10000 for i in range(0,10000,1) if (i/10000>0.1) & (i/10000<0.3)]:

        a1 = pd.DataFrame()
        f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>=thred,1,0), average='macro')
        a1['f1'] = [f1]
        a1['thred'] = [thred]
        f1_score_df = pd.concat([f1_score_df, a1], axis=0)

    thred = f1_score_df.loc[f1_score_df['f1']==f1_score_df['f1'].max(),'thred'].tolist()[0]
    print('thred:',thred)
    print('ncol',len(feats))

    # train f1
    print('auc:',oof_auc)
    oof_f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
    print('f1:',oof_f1)
    a1 = train_df['Y_LABEL'].value_counts()/len(train_df)
    print('Target ratio(real):',(a1[1]))

    # test err
    test_df['TARGET'] = np.where(test_df['Y_LABEL_lgb']>thred,1,0)
    a1 = test_df['TARGET'].value_counts()/len(test_df)
    print('Target ratio(pred):',(a1[1]))
    target_sum = test_df['TARGET'].sum()
    print('Target sum:',target_sum)

    # save 
    train_df['Y_LABEL_lgb'] = oof_preds_lgb
    a1 = train_df[['ID','YEAR','COMPONENT_ARBITRARY','Y_LABEL_lgb','Y_LABEL']].copy()
    a1.to_csv('train_pred_'+str(seed_num)+'_'+str(np.round(oof_f1,10))+'.csv', index= False)    
    a1 = test_df[['ID','YEAR','COMPONENT_ARBITRARY','Y_LABEL_lgb']].copy()
    a1.to_csv('test_pred_'+str(seed_num)+'_'+str(np.round(oof_f1,10))+'.csv', index= False)

    # submit
    a1 = test_df[['ID', 'TARGET']].copy()
    a1 = a1.rename(columns={'TARGET':'Y_LABEL'})
    submission_file_name = 'sample_submission_lgb_'+str(np.round(oof_f1,4))+'.csv'
    a1.to_csv(submission_file_name, index= False)
