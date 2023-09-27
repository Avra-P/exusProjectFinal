import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

def evaluate_xgboost_with_metrics(X_train, y_train, X_test, y_test, best_xgb_model):
    """
    Run XGBoost with the best model obtained from hyperparameter tuning and evaluate it using various metrics.

    Parameters:
    - X_train: DataFrame containing training features.
    - y_train: Series containing training target variable.
    - X_test: DataFrame containing test features.
    - y_test: Series containing test target variable.
    - best_xgb_model: Best XGBoost classifier with tuned hyperparameters.

    Returns:
    - Dictionary containing evaluation metrics (AUC, F1-score, recall, precision).
    """
    # Fit the best XGBoost model on the training data
    
    best_xgb_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_prob = best_xgb_model.predict_proba(X_test)[:, 1]
    y_pred = best_xgb_model.predict(X_test)

    # Calculate evaluation metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Create a dictionary to store the metrics
    metrics_dict = pd.DataFrame({
        'AUC': auc,
        'F1-Score': f1,
        'Recall': recall,
        'Precision': precision
    },index=[0])

    return metrics_dict
