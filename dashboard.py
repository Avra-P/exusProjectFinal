import numpy as np
import pandas as pd
from scripts import loadData
from scripts import exporeData_plots
from scripts import constructMergedDataset
from scripts import evaluate_models_with_weighting
from scripts import tune_xgboost_with_grid_search
from scripts import evaluate_xgboost_with_metrics
from scripts import feature_engineering
from dateutil.relativedelta import relativedelta

# exporeData_plots.exporeData_plots() 

defaultData_rowData,\
    clientBureauInfo_rowData,\
        clientInformation_rowData,\
            loanInformation_rowData = loadData.load_input_files(Default_Data = ".\\assignment_datasets\\Default_Data.csv",
                                                                ClientBureau_Info = ".\\assignment_datasets\\Client_Bureau_Information.csv", 
                                                                Loan_Information = ".\\assignment_datasets\\Loan_Information.csv",
                                                                Client_Information = ".\\assignment_datasets\\Client_Information.csv")



#create a flag indicates the default and non default loans
defaultData_rowData['target'] = np.where(defaultData_rowData['DVAL'].isnull(),0,1)


#fill na values of DVAL,DMON (non default loans) with 0 
# (0 not exist as a value on these columns, hence we can use it as neutral case)
#DMON and DVAL have na on the some rows (non default loans)

defaultData_rowData.loc[defaultData_rowData['DVAL'].isnull(),'DVAL'] = 0
defaultData_rowData.loc[defaultData_rowData['DVAL']==0,'DMON'] = 0

#omitte cases that defaultDate is after 2019-10-01
def omittedActiveLoansAfterBusinessDate(defaultData_rowData,loanInformation_rowData,Date):
    
    defaultData_rowData = pd.merge(defaultData_rowData,loanInformation_rowData[['UID','RECORDNUMBER','ACCSTARTDATE']],on=['UID','RECORDNUMBER'],how='left')
    defaultData_rowData['ACCSTARTDATE'] = pd.to_datetime(defaultData_rowData['ACCSTARTDATE'])
    defaultData_rowData['defaultMonth'] = pd.to_datetime(defaultData_rowData['ACCSTARTDATE']+pd.to_timedelta(defaultData_rowData['DMON']*31,unit='d'))
    testCutData = defaultData_rowData.loc[defaultData_rowData['defaultMonth']>=pd.to_datetime(Date),['UID','RECORDNUMBER','ACCSTARTDATE','DMON']]

    omittedIndexes = [] 
    for index,temp1,temp2 in zip(list(testCutData.index),testCutData['ACCSTARTDATE'].tolist(),testCutData['DMON'].tolist()):
        
        date = pd.to_datetime(temp1)
        new_date = date + relativedelta(months=temp2)
        if new_date>=pd.to_datetime(Date):
            
            omittedIndexes = omittedIndexes+[index]

    defaultData_rowData = defaultData_rowData[~defaultData_rowData.index.isin(omittedIndexes)]
    defaultData_rowData = defaultData_rowData.drop(columns=['ACCSTARTDATE','defaultMonth'],axis=1)

    return defaultData_rowData

defaultData = omittedActiveLoansAfterBusinessDate(defaultData_rowData,loanInformation_rowData,'2019-10-01')
    
#defaultData = pd.merge(defaultData,defaultData.groupby('UID',as_index=False).size().rename(columns={'size':'numProducts'}),on='UID',how='left')
    
defaultData = pd.merge(defaultData.drop(columns=['target'],axis=1),defaultData.groupby('UID',as_index=False)['target'].max(),on='UID',how='left')

defaultDataPerUser = defaultData[['UID','target']].drop_duplicates()

clientBureauInfo = clientBureauInfo_rowData.copy()

loanInformation = loanInformation_rowData.copy()
    
loanInformation['FIRST_MONTH'] = pd.to_datetime(loanInformation['FIRST_MONTH'])
loanInformation['LAST_MONTH'] = pd.to_datetime(loanInformation['LAST_MONTH'])

loanInformation['ACCSTARTDATE'] = pd.to_datetime(loanInformation['ACCSTARTDATE'])

loanInformation['SEARCHDATE'] = pd.to_datetime(loanInformation['SEARCHDATE'],  format='%Y%m%d')

#because we do not have any business sense how to fill repayperiod the data point are exclude from the analysis
#loanInformation = loanInformation[(loanInformation['REPAYPERIOD'].notnull())]

#exclude data based on the project scope 
loanInformation = loanInformation[(loanInformation['REPAYPERIOD']<=60) & (loanInformation['REPAYPERIOD']>=12)]

loanInformationPerUID = loanInformation.groupby('UID',as_index=False).\
    agg({'OPENBALANCE':[pd.Series.min,pd.Series.max,'mean'],
         'FIRST_MONTH':[pd.Series.min],
         'LAST_MONTH':[pd.Series.max],
         'REPAYPERIOD':[pd.Series.min,pd.Series.max],
         'ACCSTARTDATE':[pd.Series.min,pd.Series.max],
         'SEARCHDATE':[pd.Series.min,pd.Series.max]})


loanInformationPerUID.columns = loanInformationPerUID.columns.map('_'.join).str.strip('_')

mergedData= constructMergedDataset.constructMergedDataset(defaultDataPerUser,clientBureauInfo,loanInformationPerUID)

X = mergedData.copy()
y = mergedData[['UID','target']] 
realResponseRate = 0.2

X_train_final,X_test_final,y_train, y_test,realResponseRate_X_test,realResponseRate_y_test = feature_engineering.feature_engineering(clientBureauInfo,clientInformation_rowData,mergedData,X,y,realResponseRate)

# X= X_train_final.drop(columns=['UID','target'],axis=1)
# y = y_train[['target']]
# X_test = X_test_final.drop(columns=['UID','target'],axis=1)
# yTest = y_test[['target']]
# X_test_real = realResponseRate_X_test.drop(columns=['UID'],axis=1)
# y_test_real = realResponseRate_y_test[['target']]
#sum(realResponseRate_test['target']==1)/sum(realResponseRate_test['target']==0)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

evaluation_results,feature_importances = evaluate_models_with_weighting.evaluate_models_with_weighting(X_train_final.drop(columns=['UID'],axis=1),
                                                    y_train[['target']],
                                                    realResponseRate,
                                                    X_test_final.drop(columns=['UID'],axis=1),
                                                     y_test[['target']],
                                                     realResponseRate_X_test.drop(columns=['UID'],axis=1),
                                                     realResponseRate_y_test[['target']])
print(evaluation_results)
evaluation_results.to_csv('./data/GridSearchEvaluationResults.csv')
feature_importances.to_csv('./data/feature_importances.csv')
print('-----------------------------------------------------')
best_xgb_model = tune_xgboost_with_grid_search.tune_xgboost_with_grid_search(realResponseRate,X_train_final.drop(columns=['UID'],axis=1), y_train[['target']])

evaluation_metrics = evaluate_xgboost_with_metrics.evaluate_xgboost_with_metrics(X_train_final.drop(columns=['UID'],axis=1),
                                                    y_train[['target']],
                                                    realResponseRate_X_test.drop(columns=['UID'],axis=1),
                                                    realResponseRate_y_test[['target']], best_xgb_model)
        # AUC  F1-Score    Recall  Precision
# 0  0.960044  0.674772  0.991071   0.511521

print(evaluation_metrics)
evaluation_metrics.to_csv('./data/finalMOdel EvaluationResults.csv')
