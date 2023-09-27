import pandas as pd
from scripts import DVAL_DMON
from scripts import prod_target
from scripts import class_score
from scripts import openBalance_repayPeriod
import numpy as np

def exporeData_plots():
    
    """ 
    
    explore the data handling non values,outliers,explore patterns, distributions and apply the appropriate transformations
    
     Args:
        DefaultData_rowData (str): "data describing defaulters and no defaulters"
        clientBureauInfo_rowData (str): "data providing risk-based information"
        loanInformation_rowData (str): "data providing information related to loans"

    Returns:
        tuple: A tuple containing four Pandas DataFrames.
    """
    
    print('ok')
    
    defaultData_rowData = pd.read_pickle('./data/Default_Data.plk')
    clientBureauInfo_rowData = pd.read_pickle('./data/clientBureauInfo.plk')
    loanInformation_rowData = pd.read_pickle('./data/loanInformation.plk')
    
    
    # #create a flag indicates the default and non default loans
    # defaultData_rowData['target'] = np.where(defaultData_rowData['DVAL'].isnull(),0,1)


    # #fill na values of DVAL,DMON (non default loans) with 0 
    # # (0 not exist as a value on these columns, hence we can use it as neutral case)
    # #DMON and DVAL have na on the some rows (non default loans)

    # defaultData_rowData.loc[defaultData_rowData['DVAL'].isnull(),'DVAL'] = 0
    # defaultData_rowData.loc[defaultData_rowData['DVAL']==0,'DMON'] = 0
    
    # DVAL_DMON.DVAL_DMON(defaultData_rowData,'initial')
    
    # #scatter plot between default value and default month indicate that
    #     # the majority of the defaulted loans are less than 10000
    #     # there are some outliers, based on the default value (more tha 35000)
    
    # #defaultData = defaultData_rowData[defaultData_rowData['DVAL']<35000]
    
    
    # DVAL_DMON.DVAL_DMON(defaultData_rowData[defaultData_rowData['DVAL']<35000],'DVAL_low3500')
    
    # DVAL_DMON.DVAL_DMON(defaultData_rowData[defaultData_rowData['DVAL']<10000],'DVAL_low_10000')
    
    # #the histogram between products and default flag depict a paterns.
    # # products with biggest order number have more defaulted loans 
    # prod_target.prod_target(defaultData_rowData,'initial')
    # prod_target.prod_target(defaultData_rowData[(defaultData_rowData['DVAL']<10000)],'dval_less10000')
    
    # class_score.class_score(clientBureauInfo_rowData,flag='initial')
    
    # # The distribution of SCORE by class is almost the same normal distribution
    
    
    #almost the one third of the loans are premium. 
    
    openBalance_repayPeriod.openBalance_repayPeriod(loanInformation_rowData[loanInformation_rowData['OPENBALANCE']<20000],flag='initial')
    
    openBalance_repayPeriod.openBalance_repayPeriod(loanInformation_rowData[(loanInformation_rowData['REPAYPERIOD']>=12) & (loanInformation_rowData['REPAYPERIOD']<60)],flag='initial')
    
    #probably the loan with original term more than 400 is error values. 
    #also the 1000 value for term is a fill in value almost for sure. 
    #the majority of loans are less than 200 months as we expect.
    
    #further analysis bassed on the date columns    
    
    

  