def find_best_seed_f1(train,test,params,stratified,num_folds,drop_features,seed_num):
        
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
                verbose_eval = False,
                categorical_feature = categorical_feature
            )

        oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds_lgb += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb))

    # Write submission file and plot feature importance
    test_df['Y_LABEL_lgb'] = sub_preds_lgb

    # vi
    # print('-'*50)
    # display(feature_importance_df.groupby(['feature'])['importance'].sum().sort_values(ascending=False).head(20))
    # print('-'*50)
    # display_importances(feature_importance_df)

    # find the best thred for f1-score
    f1_score_df = pd.DataFrame()
    for thred in [i/1000 for i in range(0,1000,1)]:

        a1 = pd.DataFrame()
        f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
        a1['f1'] = [f1]
        a1['thred'] = [thred]
        f1_score_df = pd.concat([f1_score_df, a1], axis=0)

    thred = f1_score_df.loc[f1_score_df['f1']==f1_score_df['f1'].max(),'thred'].tolist()[0]
    print('thred:',thred)
    print('ncol',len(feats))

    # train err
    oof_auc = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)
    print('auc:',oof_auc)
    oof_f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
    print('f1:',oof_f1)
    test_df['TARGET'] = np.where(test_df['Y_LABEL_lgb']>thred,1,0)
    target_sum = test_df['TARGET'].sum()
    print('target_sum:',target_sum)
    
    return {'seed_num':[seed_num],'auc':[oof_auc],'f1':[oof_f1],'target_sum':[target_sum]}
