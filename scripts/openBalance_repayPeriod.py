
import matplotlib.pyplot as plt
def openBalance_repayPeriod(loanInformation,flag):
        
    '''
    Args:
        loanInformation (dataFrame): "data providing information related to loans"
    
    Returns:
        histogram: original loan term (REPAYPERIOD) distribution
        histrogram:loan opening balance (OPENBALANCE) distribution
    
    '''
    
    loanInformation[['REPAYPERIOD','OPENBALANCE']]\
        .hist(bins=20, figsize=(12, 6))
    plt.tight_layout()
    
    plt.savefig(f'./misc/graphs/repayPeriod_onBalance_distr {str(flag)}.png')

    plt.show()
    
    
    # Create a scatter plot

    plt.figure(figsize=(10, 6))  # Set the figure size (adjust as needed)
    plt.scatter(loanInformation['REPAYPERIOD'], loanInformation['OPENBALANCE'], alpha=0.5) 
    # Add labels and title

    plt.xlabel('original loan term')
    plt.ylabel('loan opening balance')
    plt.title('examine correlation between repay period and open balance')
    # Show the plot

    plt.savefig(f'./misc/graphs/openbalance_perRepayPeriod {str(flag)}.png')

    plt.show()
        
