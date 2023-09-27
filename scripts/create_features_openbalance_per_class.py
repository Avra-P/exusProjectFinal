
def create_features_openbalance_per_class(data):
    """
    Create statistical features per 'CLASS' and 'OPENBALANCE_max'.

    Parameters:
    - data: DataFrame containing 'CLASS' and 'OPENBALANCE_max'.

    Returns:
    - DataFrame with statistical features per 'CLASS' and 'OPENBALANCE_max'.
    """
    # Calculate statistical features for 'OPENBALANCE' per 'CLASS'
    class_stats = data.groupby('CLASS')['OPENBALANCE_mean'].agg([
        ('OpenBalance_Mean_perCLASS', 'mean'),
        ('OpenBalance_Sum_perCLASS', 'sum'),
        ('OpenBalance_Max_perCLASS', 'max'),
        ('OpenBalance_Min_perCLASS', 'min'),
        ('OpenBalance_Std_perCLASS', 'std')
    ]).reset_index()

    # Merge the statistical features back to the original DataFrame based on 'CLASS'
    data = data.merge(class_stats, on='CLASS', how='left')

    return data
