import matplotlib.pyplot as plt

def DVAL_DMON(defaultData,flag):
        
    '''
    investigate the distribution and the relation od default value and default month
    
    Args:
        DefaultData_rowData (dataFrame): "data describing defaulters and no defaulters"
    
    Returns:
        histogram:Default value (DVAL) distribution
        histrogram: Default month (DMON) distribution
    
    '''
    
    defaultData.loc[defaultData['target']==1,['DVAL', 'DMON']]\
        .hist(bins=20, figsize=(12, 6))
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(f'./misc/graphs/DVAL_DMON_distr_{str(flag)}.png')

    # Create a scatter plot

    plt.figure(figsize=(10, 6))  # Set the figure size (adjust as needed)
    plt.scatter(defaultData['DVAL'], defaultData['DMON'], alpha=0.5) 
    # Add labels and title

    plt.xlabel('default value')
    plt.ylabel('default month')
    plt.title('examine correlation between default value and month')

    plt.savefig(f'./misc/graphs/DVAL_DMON_corr_{str(flag)}.png')
    
    # plt.show()
    