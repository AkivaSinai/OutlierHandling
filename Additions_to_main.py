import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def replace_outlier_by_median(df, column, outlier):
    new_df= df
    median= df[column].median()
    #df.loc[[~b for b in outlier], column] = median
    #df[column] = np.where([~b for b in outlier], median, df[column])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        new_df[column]= new_df[column].where(outlier, other=median)
    return new_df


def replace_outlier_by_flooring_and_capping(df, column, outlier, min_percentile=0.01, top_percentile= 0.99):
    median= df[column].median()
    temp_df= df
    temp_df["outlier"]= outlier
    percentiles = df[column].quantile([min_percentile, top_percentile]).values
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        temp_df[column][(temp_df[column] > median ) & (temp_df ["outlier"]== False)]= percentiles[0]
        temp_df[column][(temp_df[column] < median ) & (temp_df ["outlier"]== False)]= percentiles[1]
    return temp_df.drop(["outlier"], axis=1)

#
# #todo- make treatment also for not percentile detection  ( pass the outlier mask)
# def quantile_based_flooring_and_capping(df, column,  min_percentile=0.01, top_percentile= 0.99 ):
#     percentiles = df[column].quantile([min_percentile, top_percentile]).values
#     df[column][df[column] <= percentiles[0]] = percentiles[0]
#     df[column][df[column] >= percentiles[1]] = percentiles[1]
#     return df
def get_treated_datasets(X_train_copy, col, outlier_mask, y_train_copy, treatment_methods= ["drop", "median", "floored_and_capped"]):
    new_datasets_dict={}
    if("drop" in treatment_methods):
        X_train_masked, y_train_masked = X_train_copy.iloc[outlier_mask, :], y_train_copy.iloc[outlier_mask]
        new_datasets_dict["drop"]= [X_train_masked, y_train_masked]
    if("median" in treatment_methods):
        X_train_median_replaced = replace_outlier_by_median(X_train_copy.copy(), col, outlier_mask)
        new_datasets_dict["median"]= [X_train_median_replaced, y_train_copy]
    if("floored_and_capped" in treatment_methods):
        X_train_capped_and_floored = replace_outlier_by_flooring_and_capping(X_train_copy.copy(), col, outlier_mask)
        new_datasets_dict["floored_and_capped"]= [X_train_capped_and_floored, y_train_copy]
    # if (detection_method == "percentile"):
    #     X_train_capped = quantile_based_flooring_and_capping(X_train_copy, col)
    #     new_datasets_dict["capped_floored"] = [X_train_capped, y_train_copy]
    #     # summarize the shape of the updated training dataset
    return new_datasets_dict