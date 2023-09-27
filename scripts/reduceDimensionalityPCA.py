import pandas as pd

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
def reduceDimensionalityPCA(X_train,X_test,variance_threshold):

    """
    # Select a number of components that explain a sufficiently high percentage (variance_threshold) of the total variance 

    Args:
        X_train (dataFrame): "initial feature table (training data)"
        X_test (dataFrame): "initial feature table (test data)"
        variance_threshold: "min explained variance value after selected n_component in PCA"
        
    Returns:
    
        trainData: "the new training data only with the principal component features from PCA"
        testData:"the new test data with the principal component features from PCA"

    """

    trainData = X_train.drop(columns=['UID'],axis=1)
    testData = X_test.drop(columns=['UID'],axis=1)
    
    pca = PCA()
    pca.fit(trainData)
    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    
    
    # plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance Ratio')
    # plt.title('Explained Variance vs. Number of Components')
    # plt.show()
    # plt.savefig('./misc/graphs/PCAgraph.png')

    varianceTable = pd.DataFrame(cumulative_variance_ratio).reset_index()
    
    varianceTable['index'] = varianceTable['index']+1
    erbowSelectedComponentNum = varianceTable.loc[varianceTable[0]>variance_threshold,'index'].min()
        
    pca = PCA(n_components=erbowSelectedComponentNum)
    rd_TrainData = pca.fit_transform(trainData.select_dtypes(include=np.number))
    rd_TestData = pca.fit_transform(testData.select_dtypes(include=np.number))

    rd_TrainData = pd.DataFrame(rd_TrainData, columns=[f'PC_{i+1}' for i in range(erbowSelectedComponentNum)])
    rd_TestData = pd.DataFrame(rd_TestData, columns=[f'PC_{i+1}' for i in range(erbowSelectedComponentNum)])

    excludeInitialFeatures = [x for x in X_train.columns if "F_" not in x ]  
    trainData = pd.merge(X_train[excludeInitialFeatures],rd_TrainData,left_index=True,right_index=True)
    testData = pd.merge(X_test[excludeInitialFeatures],rd_TestData,left_index=True,right_index=True)
    
    return trainData,testData
