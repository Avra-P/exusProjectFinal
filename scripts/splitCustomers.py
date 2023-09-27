import numpy as np

def splitCustomers(mergedData,randomState = 10,testPercent=0.2):
    
    numberOfCustomers = mergedData.UID.nunique()
    testSampleNumCustomers = int(numberOfCustomers*testPercent)
    allCustomers = list(mergedData.UID.unique())
    
    # Seed NumPy's random number generator
    np.random.seed(randomState)

    # Shuffle the input list randomly using NumPy
    np.random.shuffle(allCustomers)
    
    testSampleCustomers = allCustomers[0:testSampleNumCustomers]
    trainSampleCustomers = allCustomers[testSampleNumCustomers+1:]

    trainData = mergedData[mergedData['UID'].isin(trainSampleCustomers)]
    testData = mergedData[mergedData['UID'].isin(testSampleCustomers)]

    X_train_rowData = trainData.drop(columns=['target'],axis=1 )
    X_test_rowData  = testData.drop(columns=['target'],axis=1 )
    y_train = trainData[['UID','target']]
    y_test = testData[['UID','target']]
    
    X_train_rowData.to_pickle("./data/X_train_rowData.plk")
    X_test_rowData.to_pickle("./data/X_test_rowData.plk")
    y_train.to_pickle("./data/y_train.plk")
    y_test.to_pickle("./data/y_test.plk")
    
    #print(allCustomers[0])

    return X_train_rowData, X_test_rowData, y_train, y_test
