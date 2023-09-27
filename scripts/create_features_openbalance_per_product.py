
def create_features_openbalance_per_product(data):
    """
    Create statistical features per 'RECORDNUMBER' and 'OPENBALANCE'.

    Parameters:
    - data: DataFrame containing 'RECORDNUMBER' and 'OPENBALANCE'.

    Returns:
    - DataFrame with statistical features per 'RECORDNUMBER' and 'OPENBALANCE'.
    """
    # Calculate statistical features for 'OPENBALANCE' per 'RECORDNUMBER'
    record_stats = data.groupby('RECORDNUMBER')['OPENBALANCE'].agg([
        ('OpenBalance_Mean_perProduct', 'mean'),
        ('OpenBalance_Sum_perProduct', 'sum'),
        ('OpenBalance_Max_perProduct', 'max'),
        ('OpenBalance_Min_perProduct', 'min'),
        ('OpenBalance_Std_perProduct', 'std')
    ]).reset_index()

    # Merge the statistical features back to the original DataFrame based on 'RECORDNUMBER'
    data = data.merge(record_stats, on='RECORDNUMBER', how='left')

    return data
