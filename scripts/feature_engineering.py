import pandas as pd
from scripts import splitCustomers
from scripts import reduceDimensionalityPCA
from scripts import handleCatInitialFeatures
from scripts import create_datetime_features
from scripts import create_features_openbalance_per_class
from scripts import createRealResponseRate_testData

from scripts import omitNAFeatures
from scripts import omitConstantFeatures

def feature_engineering(clientBureauInfo,clientInformation,mergedData,X,y,realResponseRate):
    
    clientInformation = clientInformation[clientInformation['UID'].isin(mergedData['UID'].tolist())]
    
    clientInformation = omitNAFeatures.omitNAFeatures(clientInformation)

    clientInformation = omitConstantFeatures.omitConstantFeatures(clientInformation)

    
    X_train, X_test, y_train, y_test = splitCustomers.splitCustomers(mergedData,randomState = 10,testPercent=0.2)

    #keep only the modeling UID from the client Information table
    clientInformation = clientInformation[clientInformation['UID'].isin(mergedData['UID'].unique())]

    #The already exist features in client level I will handle in two steps
        # 1rst step: apply appropriate encoder to categorical features in order to convert it to numeric
        #2nd step: apply PCA in order to reduce the dimensionality

    train_clientInformation_catEncoding,\
        test_clientInformation_catEncoding,catFeatureType = handleCatInitialFeatures.handleCatInitialFeatures(clientInformation,X_train,X_test)

    clientInformation_beforePCA_train = pd.merge(train_clientInformation_catEncoding,clientInformation.drop(columns=catFeatureType['feature'].tolist(),axis=1),on='UID',how='inner')
    clientInformation_beforePCA_test = pd.merge(test_clientInformation_catEncoding,clientInformation.drop(columns=catFeatureType['feature'].tolist(),axis=1),on='UID',how='inner')

    train_clientInformation_PCA,test_clientInformation_PCA = reduceDimensionalityPCA.reduceDimensionalityPCA(clientInformation_beforePCA_train,clientInformation_beforePCA_test,0.99)

    dateColumns = [x for x in X_train.columns if X_train[x].dtypes=='datetime64[ns]']
    X_train_1 = create_datetime_features.create_datetime_features(X_train,dateColumns)
    X_test_1 = create_datetime_features.create_datetime_features(X_test,dateColumns)

    X_train_2 = create_features_openbalance_per_class.create_features_openbalance_per_class(X_train_1)
    X_test_2 = create_features_openbalance_per_class.create_features_openbalance_per_class(X_test_1)

    initialData_columns = ['ACCSTARTDATE', 'FIRST', 'LAST', 'SEARCHDATE']
    dropColumns = [x for x in X_train_2.columns if x.split('_')[0] in initialData_columns]

    data = pd.get_dummies(clientBureauInfo, columns=['CLASS'], drop_first=True)
    newFeature = data[['UID']+[x for x in data.columns if 'CLASS' in x]]
    X_train_2 = pd.merge(X_train_2.drop(columns=['CLASS'],axis=1),newFeature,how='left')
    X_test_2 = pd.merge(X_test_2.drop(columns=['CLASS'],axis=1),newFeature,how='left')

    X_train_final = X_train_2.drop(columns=dropColumns,axis=1)
    X_test_final = X_test_2.drop(columns=dropColumns,axis=1)
    
    medianOPENBALANCE_mean = round(X_train_final['OPENBALANCE_mean'].median())
    medianOPENBALANCE_max = round(X_train_final['OPENBALANCE_max'].median())
    medianOPENBALANCE_min = round(X_train_final['OPENBALANCE_min'].median())
    

    medianRepayPeriod_max = round(X_train_final['REPAYPERIOD_max'].median())
    medianRepayPeriod_min = round(X_train_final['REPAYPERIOD_min'].median())
    
    
    X_train_final['OPENBALANCE_mean'] = X_train_final['OPENBALANCE_mean'].fillna(medianOPENBALANCE_mean)
    X_train_final['OPENBALANCE_max'] = X_train_final['OPENBALANCE_max'].fillna(medianOPENBALANCE_max)
    X_train_final['OPENBALANCE_min'] = X_train_final['OPENBALANCE_min'].fillna(medianOPENBALANCE_min)
    
    
    X_test_final['OPENBALANCE_mean'] = X_test_final['OPENBALANCE_mean'].fillna(medianOPENBALANCE_mean)
    X_test_final['OPENBALANCE_max'] = X_test_final['OPENBALANCE_max'].fillna(medianOPENBALANCE_max)
    X_test_final['OPENBALANCE_min'] = X_test_final['OPENBALANCE_min'].fillna(medianOPENBALANCE_min)

    
    X_train_final = X_train_final.fillna(0)
    X_test_final = X_test_final.fillna(0)


    X_train_final = pd.merge(X_train_final,train_clientInformation_PCA,how='left')
    X_test_final = pd.merge(X_test_final,test_clientInformation_PCA,how='left')


    realResponseRate_X_test,realResponseRate_y_test = createRealResponseRate_testData.createRealResponseRate_testData(X_test_final,y_test,realResponseRate)
    
    return X_train_final,X_test_final,y_train,y_test,realResponseRate_X_test,realResponseRate_y_test
