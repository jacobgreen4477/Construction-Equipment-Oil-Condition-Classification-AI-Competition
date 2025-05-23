
"""
psudo_id1
"""

extra_num_features = ['ANONYMOUS_1', 'ANONYMOUS_2','PQINDEX'] 
num_features = ['AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI',  'TI', 'V', 'V40', 'ZN']
vi_num_features = ['V40','FE','ZN','CU']

group_features = num_features

train['psudo_id1'] = train[group_features].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)
test['psudo_id1'] = test[group_features].apply(lambda x: ''.join(np.where(x==0,'1','0')),axis=1)

a1 = train['psudo_id1'].value_counts().reset_index(name='cnt')
low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

a1 = train.groupby(['psudo_id1'])['Y_LABEL'].mean().reset_index(name='ratio')
zero_ratio = a1.loc[a1['ratio']==0,'psudo_id1'].tolist()

others = list(set(low_cnt+zero_ratio))

train['psudo_id1'] = np.where(train['psudo_id1'].isin(others),'others',train['psudo_id1'])
test['psudo_id1'] = np.where(test['psudo_id1'].isin(others),'others',test['psudo_id1'])

print('# of element:',len(train['psudo_id1'].value_counts()))

"""
psudo_id2
"""

extra_num_features = ['ANONYMOUS_1', 'ANONYMOUS_2','PQINDEX'] 
num_features = ['AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI',  'TI', 'V', 'V40', 'ZN']
vi_num_features = ['V40','FE','ZN','CU']

group_features = num_features

train['psudo_id2'] = train[group_features].apply(lambda x: ''.join(np.where(x>x.median(),'1','0')),axis=1)
test['psudo_id2'] = test[group_features].apply(lambda x: ''.join(np.where(x>x.median(),'1','0')),axis=1)

a1 = train['psudo_id2'].value_counts().reset_index(name='cnt')
low_cnt = a1.loc[a1['cnt']<10,'index'].to_list()

a1 = train.groupby(['psudo_id2'])['Y_LABEL'].mean().reset_index(name='ratio')
zero_ratio = a1.loc[a1['ratio']==0,'psudo_id2'].tolist()

others = list(set(low_cnt+zero_ratio))

train['psudo_id2'] = np.where(train['psudo_id2'].isin(others),'others',train['psudo_id2'])
test['psudo_id2'] = np.where(test['psudo_id2'].isin(others),'others',test['psudo_id2'])

print('# of element:',len(train['psudo_id2'].value_counts()))


