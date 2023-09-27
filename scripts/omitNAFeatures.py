import pandas as pd
def omitNAFeatures(clientInformation_rowData):
        
    clientInfo = clientInformation_rowData.drop(columns=['UID'],axis=1).describe()
    clientInfo_descr = pd.melt(clientInfo)
    clientInfo = clientInfo.reset_index()
    clientInfo_descr['metric']=list(clientInfo['index'])*(len(clientInfo.columns)-1)
    
    #exclude features that do not have any value at all 
    clientInfoNull = clientInformation_rowData.isnull().sum().reset_index()
    cutNAFeatures = clientInfoNull.loc[clientInfoNull[0]!=0,'index'].tolist()

    if len(cutNAFeatures)>0:
        clientInformation_rowData = clientInformation_rowData.drop(columns=cutNAFeatures,axis=1)

    return clientInformation_rowData
    