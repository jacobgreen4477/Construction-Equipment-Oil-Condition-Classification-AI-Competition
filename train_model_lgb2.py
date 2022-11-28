def train_model(train,test,params,stratified,num_folds,drop_features,seed_num):
    
    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))
    
    seed_everything(1)
    
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
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)

    # Create arrays and dataframes to store results
    oof_preds_lgb = np.zeros(train_df.shape[0])
    sub_preds_lgb = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]

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
                verbose_eval = -1,
                num_boost_round = 10000,
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
    display(feature_importance_df.groupby(['feature'])['importance_gain'].sum().sort_values(ascending=False).head(30))
    # print('-'*50)
    # display_importances(feature_importance_df)
    
    # train auc
    oof_auc = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)

    
    if oof_auc>=0.703:

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
        
        if (target_sum>=510) & (target_sum<590):

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
