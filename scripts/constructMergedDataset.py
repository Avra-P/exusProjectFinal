import pandas as pd

def constructMergedDataset(defaultData,clientBureauInfo,loanInformation):
    
    mergedData = pd.merge(defaultData,clientBureauInfo,on=['UID'],how='left')
        
    mergedData = pd.merge(loanInformation,mergedData,on=['UID'],how='left')
    mergedData = mergedData[mergedData['target'].notnull()]
    
    responseRate = len(mergedData[mergedData['target']==1])/len(mergedData[mergedData['target']==0])
    
    notExistOnLoanData  = len(list(set(set(mergedData['UID'])-set(loanInformation['UID']))))

    notExistOnDefaultData = len(list(set(set(loanInformation['UID'])-set(mergedData['UID']))))

    UIDexistOnDefaultNotExistOnLoan = len(list(set(set(mergedData['UID'])-set(loanInformation['UID']))))

    print(f'there are {notExistOnLoanData} use cases that do not exist on loan data but exist on default data')

    print(f'there are {notExistOnDefaultData} use cases that do not exist on default data but exist on loan data')

    return mergedData
