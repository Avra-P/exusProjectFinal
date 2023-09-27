import matplotlib.pyplot as plt
import numpy as np
def prod_target(defaultData_rowData,flag):
        
    """
    explore the relation between products (RECORDNUMBER) and target
    
    
    Args:
        DefaultData_rowData (dataFrame): data describing defaulters and no defaulters
        flag (str): describe the plot (used on saving name of the plots)
    Returns:
        boxplot: Investigate the default value distribution per product
        multi-histogram: Investigate the distribution of default and non default loans per product
    
    """
    
    # Group the data by 'RECORDNUMBER' and 'target' and count the occurrences
    grouped = defaultData_rowData.groupby(['RECORDNUMBER', 'target'])\
        .size().unstack()

    # Extract unique 'RECORDNUMBER' values
    record_numbers = grouped.index

    # Create an array of indices for the x-axis (one for each unique 'RECORDNUMBER')
    x = np.arange(len(record_numbers))

    # Width of each bar
    width = 0.35

    # Create separate bars for 'target' 0 and 1
    plt.figure(figsize=(12, 6))  # Set the figure size (adjust as needed)

    plt.bar(x - width/2, grouped[0], width, label='target 0', color='b')
    plt.bar(x + width/2, grouped[1], width, label='target 1', color='r')

    # Add labels and title
    plt.xlabel('products')
    plt.ylabel('# loans')
    plt.title('# default/non default loans per product')

    # Rotate x-axis labels for readability (adjust angle as needed)
    plt.xticks(x, record_numbers, rotation=45, ha='right')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    
    plt.savefig(f'./misc/graphs/defaultFlag_perProd{str(flag)}.png')

    plt.show()
    

    
    # Filter the data to exclude rows where 'DVAL' is equal to 0
    defaultData_rowData = defaultData_rowData[defaultData_rowData['DVAL'] != 0]
    
    plt.figure(figsize=(12, 6))  # Set the figure size (adjust as needed)

    boxplot_data = []

    for record_number in sorted(defaultData_rowData['RECORDNUMBER'].unique()):
        # Filter the data for the current 'RECORDNUMBER'
        data = defaultData_rowData[defaultData_rowData['RECORDNUMBER'] == record_number]['DVAL']
        boxplot_data.append(data)

    # Create the boxplots
    plt.boxplot(boxplot_data, labels=sorted(defaultData_rowData['RECORDNUMBER'].unique()))

    # Add labels and title
    plt.xlabel('product')
    plt.ylabel('default value')
    plt.title('default value distribution by product')

    # Rotate x-axis labels for readability (adjust angle as needed)
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    
    plt.savefig(f'./misc/graphs/boxplotDVal_perProd{str(flag)}.png')
    
    plt.show()
    