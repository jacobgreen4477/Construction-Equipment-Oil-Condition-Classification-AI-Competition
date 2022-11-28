from DMS_202211 import common
from DMS_202211.seed_everything import seed_everything

def make_group_features(train, test, numeric_features, group):
    
    print('group_feature:',group)

    for numeric_var in numeric_features:

        train = train.copy()
        test = test.copy()

        a1 = train.groupby([group])[numeric_var].median().to_dict()

        train[numeric_var+'_'+group+'_'+'median'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'median'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'median'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'median']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'median'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'median']
        
        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'median'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'median']+0.00001) 
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'median'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'median']+0.00001)


        a1 = train.groupby([group])[numeric_var].max().to_dict()

        train[numeric_var+'_'+group+'_'+'max'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'max'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'max'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'max']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'max'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'max']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'max'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'max']+0.00001)
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'max'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'max']+0.00001)


        a1 = train.groupby([group])[numeric_var].min().to_dict()

        train[numeric_var+'_'+group+'_'+'min'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'min'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'min'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'min']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'min'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'min']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'min'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'min']+0.00001) 
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'min'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'min']+0.00001) 


        a1 = train.groupby([group])[numeric_var].sum().to_dict()

        train[numeric_var+'_'+group+'_'+'sum'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'sum'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'sum'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'sum']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'sum'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'sum']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'sum'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'sum']+0.00001)
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'sum'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'sum']+0.00001)
            
    return train, test
