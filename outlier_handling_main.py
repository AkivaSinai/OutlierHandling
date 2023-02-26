import  pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, precision_score, roc_auc_score, recall_score
import warnings
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from operator import itemgetter

from Univariate_outlier_treatment import replace_outlier_by_median, replace_outlier_by_flooring_and_capping, \
    get_treated_datasets
from multivariate_outliers import identify_multivariate_outliers

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from Configurations import data_sets_info_dict, REGRESSIOM_MODEL, CLASSIFICATION_MODEL, FOLD_K
from Univariate_Outier_Detection import univariate_outlier_mask
import os

import time

# A binary flag- when True- use K-fold, otherwise only run once (on first fold)
USE_K_FOLD= True

# create a directory for the results and create files for the output
outdir = os.getcwd() + '/output/' + str(time.time())
if not os.path.exists(outdir):
    os.umask(0)
    os.makedirs(outdir)
f = open(outdir + '/results.csv', 'w')
f2 = open(outdir + '/iterations.csv', 'w')



print("", file=f)





def evaluate_regression_model(predictions,y_test):
    return mean_absolute_error(predictions, y_test)


#todo - meanwhile this only works for binary classification- because we calculate the regular AUC
# we can easily make it more general- see for example AUC for multiple classes
def evaluate_classification_model(predictions,y_test):
    precision_w = precision_score(y_test, predictions, average='weighted')
    recall_w = recall_score(y_test, predictions, average='weighted')
    auc= roc_auc_score(y_test, predictions)
    return auc, precision_w, recall_w

# A simple imputer for missing data. Meanwhile using mean imputation
def impute(df):
    simple_imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    imputed_df= pd.DataFrame(simple_imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
    return  imputed_df


def detect_and_treat_outliers( X, y, IS_REGRESSION):
    best_results={}
    print("***************************************************************")
    #select the model to use for evaluation
    model= REGRESSIOM_MODEL if IS_REGRESSION else CLASSIFICATION_MODEL
    feature_columns= list(X.columns)
    # create lists for all of the results of the folds (the list lendth will eventually be k)
    mul_accuracy_list , mul_method_list, uni_accuracy_list, uni_detialed_dict_list= [], [], [],[]
    # create a k-fold for cross validation
    kf = KFold(n_splits=FOLD_K, shuffle=True, random_state=2)
    #iterate over all folds
    best_results["uni"]={"average_result": None, "filtered_col":[]}
    best_results["multi"] ={"average_result": None, "results":[], "best_methods":[]}
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        f_fold = open(outdir + '/iterations_fold ' +str(i)+'.csv', 'w')

        if (not USE_K_FOLD and i==1):
            break
        # get X and y according to fold's indexes
        X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
        # evaluate  multivariate methods, and save best method and accuracy
        best_method, best_result_multivariate, number_of_outliers_dict = multivariate_outliers(
            IS_REGRESSION, X_test, X_train, model, y_test, y_train)
        mul_accuracy_list.append(best_result_multivariate)
        mul_method_list.append(best_method)

        print("best model and best result multivariate outlier detection: \n"+best_method+'\n'+str(best_result_multivariate)  , file=f)

        best_results["multi"]["best_methods"].append(best_method)
        best_results["multi"]["results"].append(best_result_multivariate)

        best_accuracy_global, filtered_col, filtered_columns_dict = univariate_outliers(X_train, X_test, y_train,
                                                                                        y_test, feature_columns, model,
                                                                                        IS_REGRESSION)

        # print the treated (filtered)  columns by order and statitics
        if(len(filtered_col)>0):
            print(",".join(filtered_columns_dict[filtered_col[0]].keys()), file= f_fold)
            for col in filtered_col:
                print(",".join([str(x) for x in filtered_columns_dict[col].values()]), file=f_fold)
        best_results["uni"]["filtered_col"].append(filtered_columns_dict)
        f_fold.close()
        uni_accuracy_list.append(best_accuracy_global)
        uni_detialed_dict_list.append(filtered_columns_dict)
    print("best model and best result multivariate outlier detection with k fold:")
    print(mul_method_list)
    best_results["multi"]["average_result"]=round(np.average(mul_accuracy_list), 3)
    print( best_results["multi"]["average_result"])
    print("multivariate result without k fold : " + str(round(best_result_multivariate, 3)))
    print("best model and best result univariate outlier detection with k fold:")
    best_results["uni"]["average_result"]= round(np.average(uni_accuracy_list) ,3)
    print(best_results["uni"]["average_result"])

    print(uni_detialed_dict_list)

    print("univariate result without k fold : " + str(round(best_accuracy_global, 3)))
    return best_results



"""
The function splits the dataset in samples and labels 
the name of the label may be in the metadata (Configuration file)
by default- we assume it is the last column

input: df- the dataframe (pandas), info- the metadata
output- X,y - two dataframes 
"""
def split_into_X_and_y(df, info):
    if "label" in info:
        return df.drop([info["label"]], axis=1), df[info["label"]]
    else:
        return df[list(df.columns)[:-1]], df[list(df.columns)[-1]]

"""
Another baseline method.
This function evaluates a method that uses a single combination of univariate detection and treatments:
Unlike the detect_and_treat_outliers function- here only one combination of detection and treatment is considered during the iterative process
"""
def detect_and_treat_outliers_one_method(X,y, IS_REGRESSION):
    #todo- make functions for duplicate code
    #todo - also try one treatment detection_method each time
    print("***************************************************************")
    #select the model to use for evaluation
    model= REGRESSIOM_MODEL if IS_REGRESSION else CLASSIFICATION_MODEL
    # split into feature and label elements
    feature_columns= list(X.columns)
    # split into train and test sets   #todo- change to k-fold
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


    best_accuracy= np.inf if IS_REGRESSION else 0
    for detection_method in ["mad", "percentile", "iqr"]:
        for treatment_method in ["drop", "median", "floored_and_capped"]:
            # todo -also loop over treatments

            print("now only "+ detection_method + "   ********  and " + treatment_method )
            best_accuracy_for_treatment, filtered_col, filtered_columns_dict = univariate_outliers(X_train, X_test,
                                                                                                   y_train, y_test,
                                                                                                   feature_columns,
                                                                                                   model, IS_REGRESSION,
                                                                                                   [detection_method],
                                                                                                   [treatment_method])
            if (IS_REGRESSION and best_accuracy_for_treatment < best_accuracy) or (not IS_REGRESSION and best_accuracy_for_treatment > best_accuracy) :
                    best_accuracy= best_accuracy_for_treatment

    print("best accuracy for univarate outlier detection and treatment one detection_method (baseline)")
    print(best_accuracy)

    print("best accuracy for univarate outlier detection and treatment one detection_method", file= f)
    print(best_accuracy, file= f)

    return best_accuracy




def accuracy_was_improved(best_accuracy_for_col, best_accuracy_for_round, best_accuracy_global, IS_REGRESSION):
    if (IS_REGRESSION and best_accuracy_for_col < best_accuracy_global and best_accuracy_for_col < best_accuracy_for_round) :
        return True
    if(not IS_REGRESSION and best_accuracy_for_col > best_accuracy_global and best_accuracy_for_col > best_accuracy_for_round):
        return True
    return False


def get_best_result(col_results, IS_REGRESSION):
    if IS_REGRESSION:
        return min(enumerate(col_results.values()),key=itemgetter(1))
    else:
        return max( enumerate(col_results.values()), key=itemgetter(1))

""""
univariate outliers treatment 

The function implements the Iterative Univaraite Outlier Process
Input:
The function recieves the dataset divided into train and test (X_train, X_test, y_train, y_test,)
In 

"""
def univariate_outliers(X_train, X_test, y_train, y_test, columns, model, IS_REGRESSION,
                        outlier_detecting_methods=["mad", "percentile", "iqr"],
                        outlier_treating_methods=["drop", "median", "floored_and_capped"]):

    filtered_columns_dict = {}
    filtered_col = []
    improved_flag = True
    # best_accuracy_global- the best accuracy found in all iterations
    best_accuracy_global = np.infty if IS_REGRESSION else 0
    X_train_copy, y_train_copy = X_train, y_train
    # The main iteration loop: as long as their are more columns to try filtering, and as long a the previous iteration improved
    while len(filtered_col) < len(columns) and improved_flag:
        # best_accuracy_for_round- the best accuracy in the current iteration
        best_accuracy_for_round = np.infty if IS_REGRESSION else 0
        best_col = None
        improved_flag = False
        for col in [c for c in columns if c not in filtered_col]:
            data = X_train_copy[col]
            # this dict will contain all the results for all combinations of detection and treating methods
            col_results = {}
            for detection_method in outlier_detecting_methods:
                # get the outlier mask of current detection method
                outlier_mask = univariate_outlier_mask(data, detection_method)
                number_of_outliers = list(outlier_mask).count(False)
                if (number_of_outliers > len(data) / 2):
                    continue
                treated_datasets_dict = get_treated_datasets(X_train_copy.copy(), col, outlier_mask,
                                                             y_train_copy, outlier_treating_methods)
                # try every possible treating method and save the accuracy
                for outlier_treating_method, Xy_train in treated_datasets_dict.items():
                    accuracy = fit_model_and_evaluate_accuracy(Xy_train[0], Xy_train[1], X_test, y_test, model,
                                                               IS_REGRESSION)
                    col_results[(detection_method, outlier_treating_method)] = accuracy
                # choose the result of the best treatment (in terms of accuracy)
            # find key and value of best result (min for regression and max for classification)
            if len(col_results) == 0:
                continue
            index_of_best_combo, best_accuracy_for_col = get_best_result(col_results, IS_REGRESSION)

            best_detection_and_treatment = list(col_results.keys())[index_of_best_combo]
            if (accuracy_was_improved(best_accuracy_for_col,best_accuracy_for_round, best_accuracy_global, IS_REGRESSION )):
                best_accuracy_for_round = best_accuracy_for_col
                improved_flag = True
                best_col = col
                best_column_treatment_and_detection = best_detection_and_treatment
                #best_outlier_mask= outlier_mask
        # if found a way to improve prediction- meaning  that in this iteration we found a column
        # that by treating it's outliiers- the prediction was improved.
        if improved_flag:
            # get info about the best columns in this round
            improvement = abs(best_accuracy_for_round - best_accuracy_global) # the delta of accuracy  improvement
            best_accuracy_global = best_accuracy_for_round
            best_column_detection = best_column_treatment_and_detection[0]
            best_column_treatment = best_column_treatment_and_detection[1]
            # get  the outlier mask of the best detection method
            outlier_mask = univariate_outlier_mask(X_train_copy[best_col], best_column_detection)
            number_of_outliers = list(outlier_mask).count(False)
            X_train_copy, y_train_copy = treat_best_column(X_train_copy, best_column_treatment, col, outlier_mask,
                                                           y_train_copy)
            filtered_col.append(best_col)
            # save the information of the column that was treated in this iteration
            filtered_columns_dict[best_col] = {"column name": best_col, "number of outliers": number_of_outliers,
                                               "best detection": best_column_detection,
                                               "best treatment": best_column_treatment,
                                               "performance after treating": best_accuracy_global,
                                               "improvement": improvement}
    return best_accuracy_global, filtered_col, filtered_columns_dict

"""
The function gets a dataset and  one column from the dataset, with the column's outliers and the expected treatment
The function returns that the dataset after treating the outlier 
"""
def treat_best_column(X_train_copy, best_column_treatment, col, outlier_mask, y_train_copy):
    if (best_column_treatment == "drop"):
        X_train_copy, y_train_copy = X_train_copy.iloc[outlier_mask, :], y_train_copy.iloc[
            outlier_mask]
    elif (best_column_treatment == "median"):
        X_train_copy = replace_outlier_by_median(X_train_copy.copy(), col, outlier_mask)
    else:
        X_train_copy = replace_outlier_by_flooring_and_capping(X_train_copy.copy(), col, outlier_mask)
    return X_train_copy, y_train_copy

"""
The function gets a dataset and tries various methods for multivariate outlier detection. For each method- the
function drops the outliers and evaluates a given prediciton model with the modified dataset.

The function return the method that that gave the best accuracy results and the accuracy itself. In addition in 
return the number of outlier that were detected (and dropped) """
def multivariate_outliers(IS_REGRESSION, X_test, X_train, model, y_test, y_train):
    number_of_outliers_dict={}
    outlier_masks = identify_multivariate_outliers(X_train)
    X_train_copy, y_train_copy = X_train, y_train
    method_performance_dict = {}
    # for each possible detection detection_method- evaluate the model
    for detection_method, outlier_mask in outlier_masks.items():
        # print(detection_method)
        # print("outlier count:")
        number_of_outliers = list(outlier_mask).count(False)
        #print(list(outlier_mask).count(False))
        number_of_outliers_dict[detection_method]= number_of_outliers
        X_train_masked, y_train_masked = X_train_copy.iloc[outlier_mask, :], y_train_copy.iloc[outlier_mask]
        if (len(X_train_masked) < 100):
            continue
        accuracy = fit_model_and_evaluate_accuracy(X_train_masked, y_train_masked, X_test, y_test, model, IS_REGRESSION)
        # fit the model
        method_performance_dict[detection_method] = accuracy
        # scores_after_treatment.append({"detection detection_method": detection_method, "number of outliers": number_of_outliers,
        #                                "treatment detection_method": treatment_method,
        #                                "score": method_performance_dict[detection_method]})
    sorted_performance = {k: v for k, v in sorted(method_performance_dict.items(), key=lambda item: item[1])}
    if (IS_REGRESSION):
        best_method, best_result = list(sorted_performance.items())[0]
    else:
        best_method, best_result = list(sorted_performance.items())[-1]
    return best_method, best_result, number_of_outliers_dict


"""
The function get a dataset, after splitting to train-test and a prediction model
 The function fits and  evaluates the model with the given dataset and returns the accuracy
  
"""
def fit_model_and_evaluate_accuracy( X_train,y_train,  X_test,y_test, model, IS_REGRESSION):
    # fit the model
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    if (IS_REGRESSION):
        accuracy = evaluate_regression_model(yhat, y_test)
    else:
        auc, precision, recall = evaluate_classification_model(yhat, y_test)
        accuracy = auc
    return accuracy



"""
The function evaluates the given dataset with baseline methods: 1. no outlier detection or treatment 2. normalization of all the data
"""
def evaluate_baseline(X, y, info, IS_REGRESSION):
    model = REGRESSIOM_MODEL if IS_REGRESSION else CLASSIFICATION_MODEL
    # split into train and test sets using cross validation technique
    kf = KFold(n_splits=FOLD_K, shuffle=True, random_state=2)
    #normalization_accuracy_list= []
    no_treatment_accuracy_list=[]
    # for each fold - divide so that the train will be the remaining
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if (not USE_K_FOLD and i==1):
            continue
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test =  X.iloc[test_index], y.iloc[test_index]
        # get and save the accuracy of both baselines
        no_treatment_accuracy= fit_model_and_evaluate_accuracy(X_train,y_train,   X_test,y_test, model,IS_REGRESSION  )
        #normalization_accuracy= evaluate_with_simple_normailization(X_train, y_train, X_test, y_test, model, IS_REGRESSION)

        no_treatment_accuracy_list.append(no_treatment_accuracy)
        #normalization_accuracy_list.append(normalization_accuracy)
    average_accuracy_no_treament= round(np.mean(no_treatment_accuracy), 3)
    #average_accuracy_normalization = round(np.mean(normalization_accuracy), 3)

    return average_accuracy_no_treament

def main():
    results_title= ["dataset","baseline accuracy", "multivariate  accuracy"," multivariate method", "univariate accuracy"]
    print(" & ".join(results_title))
    # iterate over all datasets
    for dataset, info in data_sets_info_dict.items():
        print(dataset)
        print(dataset, file=f)
        IS_REGRESSION= (info["type"] == "regression")
        print("**************************************")
        print("**************************************", file=f)
        print(dataset, file=f2)
        print("**************************************", file=f2)

        df = read_csv(info["path"])
        df = impute(df)  # todo- more imputation methods ?- FWork
        # split into feature and label elements
        X, y = split_into_X_and_y(df, info)
        #evaluiate without outlier handing and with normalization
        accuracy_with_outliers= evaluate_baseline(X, y, info, IS_REGRESSION)
        print("accuracy with outliers (baseline)")
        print("accuracy with outliers (baseline)", file = f)
        print(accuracy_with_outliers)

        # print("accuracy with normalization (baseline)")
        # print("accuracy with normalization (baseline)", file = f)
        # print(accuracy_with_normalization)


        print("accuracy without outliers")
        print("accuracy without outliers", file=f)
        best_results= detect_and_treat_outliers(X,y, IS_REGRESSION)
        multivariate_methods, multivariate_accuracy = best_results["multi"]["best_methods"],  best_results["multi"]["average_result"]
        most_common_mul_method=  max(set(multivariate_methods), key=multivariate_methods.count)

        #
        univariate_accuracy = best_results["uni"]["average_result"]
        # univariate_combos= [str((x["best detection"], x["best treatment"])) for x in best_results["uni"]["filtered_columns_dict"].values()]
        results_list= [dataset, str(accuracy_with_outliers),str(multivariate_accuracy), most_common_mul_method, str(univariate_accuracy)]
        print(multivariate_methods)
        print (" & ".join(results_list))
main()

f.close()
f2.close()
