from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier




# a dictionary of datasets- with the following information: 1. path 2. type of problem (regression/ binary/ multiple) 3. (optional: the name of the label feature. by default this is the last column)
data_sets_info_dict ={
    #"german": {"path": "data_sets\\german_credit.csv", "type": "binary", "label": "Creditability"},  # https://online.stat.psu.edu/stat857/node/215/
    "life expectancy": {"path": "data_sets\\life_expectancy_data.csv", "type": "regression"},
    "college graduates": {"path": "data_sets\\college_graduates.csv", "type": "regression"},
    "breast cancer": {"path":"data_sets\\breast_cancer_dataset.csv",  "type": "binary_class" },
    "nba rookie": {"path":"data_sets\\nba_rookie.csv",  "type": "binary_class" },
    #"boston housing":{"path": "data_sets\\boston_pricing.csv", "type":"regression"}
# boston data set- https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html  - the features describes towns around boston, the target us the median proce

}
REGRESSIOM_MODEL= XGBRegressor(use_label_encoder =False,verbosity = 0)
CLASSIFICATION_MODEL= XGBClassifier(use_label_encoder =False,verbosity = 0)
TREATMENT_METHODS=["drop", "median_replaced", "capped_floored"]
UNIVARIATE_DETECTION_METHODS=["mad", "percentile", "iqr"]
IQR_THRESHOLD=1.5
PERCENTILE_THRESHOLD= 99
MAD_THRESHOLD= 3.5
FOLD_K= 5
#ROUND_DECIMAL_DIGITS=3

