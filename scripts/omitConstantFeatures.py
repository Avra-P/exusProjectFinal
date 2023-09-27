import pandas as pd

def omitConstantFeatures(data):
    
    """
    Get unique values per column in a DataFrame.

    Parameters:
    - data: DataFrame for which constant features should be ommited.

    Returns:
    - DataFrame with a subset of features that have at least 2 values across all UID.
    """
    unique_values = []
    
    for column in data.columns:
    #    print(column)
        unique_values = unique_values+[data[column].nunique()]
    
    unique_values_df = pd.DataFrame({'columnName':list(data.columns),'uniqueValues':unique_values})
    
    NonConstantFeatures = unique_values_df.loc[unique_values_df['uniqueValues']>1,'columnName'].tolist()
    
    if len(NonConstantFeatures)>0:
        data = data[NonConstantFeatures]
        
    return data
