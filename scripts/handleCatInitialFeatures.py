
import pandas as pd
import numpy as np

def handleCatInitialFeatures(data,X_train,X_test):
        
    """
    handle categorical variables according to the number of levels that they have
    in order to use it in ML models
    
    Args:
    
        X_train (dataFrame): initial feature table (training data)
        X_test (dataFrame): initial feature table (train data)
        
    Returns:
    
        trainData: "the new training data only with the transformed features "
        testData:"the new test data with the principal component features"

    """
    
    trainData_cat = data[data['UID'].isin(X_train['UID'].unique())].select_dtypes(exclude=np.number)
    testData_cat = data.loc[data['UID'].isin(X_test['UID'].unique())].select_dtypes(exclude=np.number)

    candidateFeatures = [x for x in trainData_cat.columns if 'UID' not in x]
    catFeatureType = pd.DataFrame()
    for tempFeature in candidateFeatures:
        #print(tempFeature)
        feature_levels = trainData_cat[tempFeature].nunique()
        if feature_levels<=4:
            
            data = pd.get_dummies(data, columns=[tempFeature], drop_first=True)
            newFeature = data[['UID']+[x for x in data.columns if tempFeature in x]]
            trainData_cat = pd.merge(trainData_cat.drop(columns=[tempFeature],axis=1),newFeature)
            testData_cat = pd.merge(testData_cat.drop(columns=[tempFeature],axis=1),newFeature)

            catFeatureType = pd.concat([catFeatureType,pd.DataFrame({'feature':tempFeature,'encoding':'onehotEncoding','numLevels':feature_levels},
                                                    index=[0])],ignore_index=True)
        elif feature_levels>4:
            frequency_encoding = trainData_cat[tempFeature].value_counts().to_dict()
            trainData_cat[tempFeature] = trainData_cat[tempFeature].map(frequency_encoding)
            testData_cat[tempFeature] = testData_cat[tempFeature].map(frequency_encoding)
            testData_cat.loc[testData_cat[tempFeature].isnull(),tempFeature]=0
            catFeatureType = pd.concat([catFeatureType,pd.DataFrame({'feature':tempFeature,'encoding':'FrequencyEncoding','numLevels':feature_levels},
                                                    index=[0])],ignore_index=True)

    trainData_cat = pd.merge(X_train[['UID']],trainData_cat,on='UID')
    testData_cat = pd.merge(X_test[['UID',]],testData_cat,on='UID')
    
    
    return  trainData_cat,testData_cat,catFeatureType