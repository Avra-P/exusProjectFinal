import pandas as pd

def loadTransformedData():
    
    defaultData = pd.read_pickle('./data/defaultData.plk')
    loanInformation = pd.read_pickle('./data/loanInformation.plk')
    clientBureauInfo = pd.read_pickle('./data/clientBureauInfo.plk')
    clientInformation = pd.read_pickle('./data/clientInformation.plk')
    
    return defaultData,loanInformation,clientBureauInfo,clientInformation