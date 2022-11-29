def bayes_parameter_opt_lgb_f1(
    train, 
    opt_params, 
    init_round=15, 
    opt_round=25, 
    n_folds=3, 
    random_seed=1, 
    n_estimators=10000, 
    output_process=False, 
    drop_features=[]
    ):   
    
    seed_everything(1)
    
    train_df = train.copy()

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        
    def f1_eval(yhat,data):

        y = data.get_label()
        thred = 0.127 # 0.125~0.13
        pred = np.where(yhat>thred,1,0)

        return 'f1', f1_score(y, pred, average='macro'),True    
    
    # parameters
    def lgb_eval(
          learning_rate
        , num_leaves
        , colsample_bytree
        , subsample
        , max_depth
        , reg_alpha
        , reg_lambda
        , min_split_gain
        , min_child_weight
        , min_child_samples

    ):
        
        params = {
            'application':'binary', 
            'metric':'auc',
            'verbose':-1
        }
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params['max_depth'] = int(round(max_depth))        
        params['reg_alpha'] = reg_alpha
        params['reg_lambda'] = reg_lambda        
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['min_child_samples'] = int(round(min_child_samples))
        
        # cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        
        # -----        
        
        stratified = True
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
        else:
            folds = KFold(n_splits= 5, shuffle=True, random_state=1)

        # Create arrays and dataframes to store results
        oof_preds_lgb = np.zeros(train_df.shape[0])

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
                    num_boost_round = 10000,
                    verbose_eval = False,
                    feval  = f1_eval,
                    early_stopping_rounds = 200,
                    categorical_feature = categorical_feature
                )

            oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

        cv_result = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)        
       
        # ----

        return cv_result

    lgbBO = BayesianOptimization(lgb_eval, opt_params, random_state=1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        lgbBO.maximize(init_points=init_round, n_iter=opt_round, acq='ucb')
    
    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    a1 = lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']
    file_name = 'res_tune_'+str(len(drop_features))+'_'+str(a1[0])+'.joblib'    
    joblib.dump(a1[1],file_name)
    
    return a1
