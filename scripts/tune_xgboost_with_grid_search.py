
import xgboost as xgb
from pickle import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

def tune_xgboost_with_grid_search(realResponseRate,X, y):
    """
    Tune hyperparameters of the XGBoost model using grid search with cross-validation.

    Parameters:
    - X: DataFrame containing features.
    - y: Series containing the target variable.

    Returns:
    - Best XGBoost classifier with tuned hyperparameters.
    """
    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0,4,0,6,0.8],
        'scale_pos_weight': [1/realResponseRate]
    }

    # Create an XGBoost classifier
    xgb_classifier = xgb.XGBClassifier()

    # Create a custom scoring function (ROC-AUC) for grid search
    custom_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

    # Create a StratifiedKFold cross-validation object
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring=custom_scorer,
                               cv=skf, verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best XGBoost classifier with tuned hyperparameters
    best_xgb_classifier = grid_search.best_estimator_

    dump(best_xgb_classifier, open('./data/selected_model.pkl', 'wb'))
    
    return best_xgb_classifier
