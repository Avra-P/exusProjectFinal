import matplotlib.pyplot as plt
import pandas as pd

def defaultPerYearAnalysis():
    
    mergedData = pd.read_pickle('./data/mergedData.plk')
    
    mergedData['issueLoanYear'] = mergedData['ACCSTARTDATE'].dt.year.astype(str)

    mergedData['defaultMonth'] = pd.to_datetime(mergedData['ACCSTARTDATE']+pd.to_timedelta(mergedData['DMON']*30,unit='d')).dt.year.astype(str)
    
    loansPerYear = pd.merge(mergedData.groupby(['defaultMonth'],as_index=False)['target'].sum().rename(columns={'defaultMonth':'Year'}),
                            mergedData.groupby(['issueLoanYear'],as_index=False).size().rename(columns={'issueLoanYear':'Year'}))

    loansPerYear = loansPerYear.rename(columns={'target':'numDefaultPerDefaultDate','size':'newIssueLoans'})

    loansPerYear = pd.merge(loansPerYear,
                            mergedData.groupby(['issueLoanYear'],as_index=False)['target'].sum().rename(columns={'issueLoanYear':'Year','target':'defaultPerIssueDate'}),
                            )

    plt.plot(loansPerYear['Year'], loansPerYear['numDefaultPerDefaultDate'],label ='num Default per DefaultDate')
    plt.plot(loansPerYear['Year'], loansPerYear['newIssueLoans'],label ='new Issue Loans')
    plt.plot(loansPerYear['Year'], loansPerYear['defaultPerIssueDate'],label ='default per issue Date')

    plt.xlabel("year")
    plt.ylabel("number of loans")
    plt.legend()
    plt.title('loan default analysis per year')
    plt.show()
    plt.save('./misc/graphs/loan default analysis per year.png')
    
    return loansPerYear