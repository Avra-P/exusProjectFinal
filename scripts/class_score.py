import matplotlib.pyplot as plt

def class_score(clientBureauInfo_rowData,flag): 
    
    '''
    Investigate the score distribution across customers' class
    
    Args:
        clientBureauInfo_rowData (dataFrame): "data providing risk-based information"
    
    Returns:
        boxplots: default value per product
    
    '''
    
    # Create a boxplot for 'SCORE' grouped by 'CLASS'
    plt.figure(figsize=(8, 6))  # Set the figure size (adjust as needed)

    # Group the data by 'CLASS' and select the 'SCORE' column
    data_grouped = clientBureauInfo_rowData.groupby('CLASS')['SCORE']

    # Create the boxplot
    plt.boxplot([data_grouped.get_group('STANDARD'), 
                data_grouped.get_group('PREMIUM')], labels=['STANDARD', 'PREMIUM'])

    # Add labels and title
    plt.xlabel('customer type')
    plt.ylabel('acquisition Score')
    plt.title('acquisition score distribution by customer type')

    plt.savefig(f'./misc/graphs/score_perCustomer {str(flag)}.png')

    # Show the plot
    plt.show()        

