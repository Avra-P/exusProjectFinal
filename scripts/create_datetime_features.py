import pandas as pd

def create_datetime_features(data,dateColumns):
    """
    Create features based on datetime columns in the input DataFrame.

    Parameters:
    - data: DataFrame containing datetime columns.

    Returns:
    - DataFrame with additional datetime-based features.
    """
    for tempColumn in dateColumns:
        # Feature 1: Time Since Account Start (in days)
        data[f'DaysSince_{tempColumn}'] = (pd.to_datetime('now') - data[tempColumn]).dt.days

        # Feature 2: Month of Account Start
        data[f'MonthOf_{tempColumn}'] = data[tempColumn].dt.month

        # Feature 3: Year of Account Start
        data[f'YearOf_{tempColumn}'] = data[tempColumn].dt.year

    return data
