import pandas as pd
import numpy as np

def transformAndSaveData(defaultData,clientBureauInfo,clientInformation,loanInformation):
    
    """
    manipulate the input tables and save it on a binary form (pickle)
    
    """
    #create a flag indicates the default and non default loans
    defaultData['target'] = np.where(defaultData['DVAL'].isnull(),0,1)
        
    loanInformation['FIRST_MONTH'] = pd.to_datetime(loanInformation['FIRST_MONTH'])
    loanInformation['LAST_MONTH'] = pd.to_datetime(loanInformation['LAST_MONTH'])
    loanInformation['ACCSTARTDATE'] = pd.to_datetime(loanInformation['ACCSTARTDATE'])
    loanInformation['SEARCHDATE'] = pd.to_datetime(loanInformation['SEARCHDATE'],  format='%Y%m%d')
    
    #fill na values of DVAL,DMON (non default loans) with 0 
    # (0 not exist as a value on these columns, hence we can use it as neutral case)
    #DMON and DVAL have na on the some rows (non default loans)
    defaultData.loc[defaultData['DVAL'].isnull(),'DVAL'] = 0
    defaultData.loc[defaultData['DVAL']==0,'DMON'] = 0
    
    defaultData.to_pickle('./data/defaultData.plk')
    loanInformation.to_pickle('./data/loanInformation.plk')
    clientBureauInfo.to_pickle('./data/clientBureauInfo.plk')
    clientInformation.to_pickle('./data/clientInformation.plk')