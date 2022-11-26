"""
make psudo id
"""

group_features = num_features13
group_features = list(set(group_features))

for c1, c2 in [i for i in itertools.combinations(group_features,2)]:
    new_c = f'{c1}_{c2}_id'
    train[new_c] = train[[c1,c2]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    test[new_c] = test[[c1,c2]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    
    a1 = train[new_c].value_counts().reset_index(name='cnt')
    low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

    a1 = train.groupby([new_c])['Y_LABEL'].mean().reset_index(name='ratio')
    zero_ratio = a1.loc[a1['ratio']==0,new_c].tolist()

    others = list(set(low_cnt+zero_ratio))

    train[new_c] = np.where(train[new_c].isin(others),'others',train[new_c])
    test[new_c] = np.where(test[new_c].isin(others),'others',test[new_c])
    
for c1, c2, c3 in [i for i in itertools.combinations(group_features,3)]:
    new_c = f'{c1}_{c2}_{c3}_id'
    train[new_c] = train[[c1,c2,c3]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    test[new_c] = test[[c1,c2,c3]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    
    a1 = train[new_c].value_counts().reset_index(name='cnt')
    low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

    a1 = train.groupby([new_c])['Y_LABEL'].mean().reset_index(name='ratio')
    zero_ratio = a1.loc[a1['ratio']==0,new_c].tolist()

    others = list(set(low_cnt+zero_ratio))
    
    train[new_c] = np.where(train[new_c].isin(others),'others',train[new_c])
    test[new_c] = np.where(test[new_c].isin(others),'others',test[new_c])
    
    
"""
make psudo id2 (중앙값보다 높은거)
"""

group_features = num_features13
group_features = list(set(group_features))

med_dict = train[group_features].apply(lambda x: x.median()).to_dict()
for i,j in med_dict.items():
    
    # greater than median
    train[i+'_gtm'] = np.where(train[i]>j,1,0)
    test[i+'_gtm'] = np.where(test[i]>j,1,0)

    
group_features = train.columns[train.columns.str.contains('_gtm')].tolist()
group_features = list(set(group_features))

for c1, c2 in [i for i in itertools.combinations(group_features,2)]:
    new_c = f'{c1}_{c2}_id2'
    train[new_c] = train[[c1,c2]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    test[new_c] = test[[c1,c2]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    
    a1 = train[new_c].value_counts().reset_index(name='cnt')
    low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

    a1 = train.groupby([new_c])['Y_LABEL'].mean().reset_index(name='ratio')
    zero_ratio = a1.loc[a1['ratio']==0,new_c].tolist()

    others = list(set(low_cnt+zero_ratio))

    train[new_c] = np.where(train[new_c].isin(others),'others',train[new_c])
    test[new_c] = np.where(test[new_c].isin(others),'others',test[new_c])
    
for c1, c2, c3 in [i for i in itertools.combinations(group_features,3)]:
    new_c = f'{c1}_{c2}_{c3}_id2'
    train[new_c] = train[[c1,c2,c3]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    test[new_c] = test[[c1,c2,c3]].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
    
    a1 = train[new_c].value_counts().reset_index(name='cnt')
    low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

    a1 = train.groupby([new_c])['Y_LABEL'].mean().reset_index(name='ratio')
    zero_ratio = a1.loc[a1['ratio']==0,new_c].tolist()

    others = list(set(low_cnt+zero_ratio))
    
    train[new_c] = np.where(train[new_c].isin(others),'others',train[new_c])
    test[new_c] = np.where(test[new_c].isin(others),'others',test[new_c])
