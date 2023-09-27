
import random
import pandas as pd


def createRealResponseRate_testData(X_test,y_test,realResponseRate):
    
    balanceResponseRate = sum(y_test['target']==1)/sum(y_test['target']==0)
    realNumPositiveCase_test = int(realResponseRate*sum(y_test['target']==1)/balanceResponseRate)
    random_numbers = [random.randint(0,sum(y_test['target']==1)-1) for _ in range(realNumPositiveCase_test)]
    positiveCases = y_test[y_test['target']==1]
    positiveCases = positiveCases.reset_index()
    negativeCasesCases = y_test[y_test['target']==0]
    negativeCasesCases = negativeCasesCases.reset_index()
    realResponseRate_y_test = pd.concat([negativeCasesCases[['UID','target']],positiveCases[['UID','target']].iloc[random_numbers]],ignore_index=True)

    realResponseRate_X_test = pd.merge(realResponseRate_y_test[['UID']],X_test,on=['UID'],how='inner')

    return realResponseRate_X_test,realResponseRate_y_test
