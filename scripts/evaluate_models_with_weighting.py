import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def evaluate_models_with_weighting(X,
                                   y,
                                   realResponseRate,
                                   X_test,
                                   yTest,
                                   X_test_real,
                                   y_test_real):
    
    """
    Apply multiple machine learning models with adjusted class weights and compare them based on evaluation metrics.

    Parameters:
    - X: DataFrame containing features.
    - y: Series containing the target variable.

    Returns:
    - DataFrame with evaluation metrics for each model.
    
    """
    # Define class weights based on the real response rate
    class_weights = {0: 1 - realResponseRate, 1: realResponseRate}

    # Define classifiers with adjusted class weights
    classifiers = {
        'Logistic Regression': LogisticRegression(class_weight=class_weights),
        'Random Forest': RandomForestClassifier(class_weight=class_weights),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=(1 / realResponseRate))
    }

    # Initialize a DataFrame to store evaluation metrics
    metrics_df = pd.DataFrame(columns=['Classifier', 'ROC-AUC_test', 'Recall_test', 'Precision_test', 'F1-Score_test','ROC-AUC_train', 'Recall_train', 'Precision_train', 'F1-Score_train','ROC-AUC_test_real', 'Recall_test_real', 'Precision_test_real', 'F1-Score_test_real',])

    # Iterate through classifiers and evaluate them
    for name, clf in classifiers.items():
        
        clf.fit(X, y)
        
        if name=='Random Forest':
            # Get feature importances
            importances = clf.feature_importances_
            
            # Create a DataFrame to store feature names and their importances
            feature_importances = pd.DataFrame({'Feature': X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1]),
                                        'Importance': importances})
        
        # Make predictions on the test data
        y_pred = clf.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        roc_auc_test = roc_auc_score(yTest, y_pred)
        recall_test = recall_score(yTest, y_pred > 0.5)
        precision_test = precision_score(yTest, y_pred > 0.5)
        f1_test = f1_score(yTest, y_pred > 0.5)
        
        y_pred = clf.predict_proba(X)[:, 1]
        
        # Calculate evaluation metrics
        roc_auc_train = roc_auc_score(y, y_pred)
        recall_train = recall_score(y, y_pred > 0.5)
        precision_train = precision_score(y, y_pred > 0.5)
        f1_train = f1_score(y, y_pred > 0.5)

        y_pred = clf.predict_proba(X_test_real)[:, 1]
        
        # Calculate evaluation metrics
        roc_auc_test_real = roc_auc_score(y_test_real, y_pred)
        recall_test_real = recall_score(y_test_real, y_pred > 0.5)
        precision_test_real = precision_score(y_test_real, y_pred > 0.5)
        f1_test_real = f1_score(y_test_real, y_pred > 0.5)

        # Append metrics to the DataFrame
        metrics_df = pd.concat([metrics_df,pd.DataFrame({
            'Classifier': name,
            'ROC-AUC_test': roc_auc_test,
            'Recall_test': recall_test,
            'Precision_test': precision_test,
            'F1-Score_test': f1_test,
            'ROC-AUC_train': roc_auc_train,
            'Recall_train': recall_train,
            'Precision_train': precision_train,
            
            'ROC-AUC_test_real': roc_auc_test_real,
            'Recall_test_real': recall_test_real,
            'Precision_test_real': precision_test_real,
            'F1-Score_test_real': f1_test_real,
            
            'F1-Score_train': f1_train},index=[0])],ignore_index=True)

    return metrics_df,feature_importances
