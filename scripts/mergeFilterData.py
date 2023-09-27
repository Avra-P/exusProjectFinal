import pandas as pd

def mergeFilterData(defaultData,loanInformation,clientBureauInfo,clientInformation):
    
    mergedData = pd.merge(defaultData,loanInformation,on=['UID','RECORDNUMBER'],how='left')

    #exclude data based on the project scope 
    mergedData = mergedData[(loanInformation['REPAYPERIOD']<=60) & (loanInformation['REPAYPERIOD']>=12) | (loanInformation['REPAYPERIOD'].isnull())]

    #exclude na values (only one exist) in order to use this column to split data stratified by the account start date
    mergedData = mergedData[mergedData['ACCSTARTDATE'].notnull()]

    #merge clientInfo table on the main data table on UID level
    mergedData = pd.merge(mergedData,clientInformation,on=['UID'],how='left')

    clientBureauInfo = clientBureauInfo.copy()

    #merge client score table on the main data table on UID level
    mergedData = pd.merge(mergedData,clientBureauInfo,on=['UID'],how='left')

    mergedData.to_pickle('./data/mergedData.plk')
    
    return mergedData