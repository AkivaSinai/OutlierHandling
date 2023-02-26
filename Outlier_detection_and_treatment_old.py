# evaluate model performance with outliers removed using isolation forest
import  pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, precision_score, roc_auc_score, recall_score
import warnings
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from Univariate_outlier_treatment import replace_outlier_by_median, replace_outlier_by_flooring_and_capping, \
    get_treated_datasets
from multivariate_outliers import identify_multivariate_outliers

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from Configurations import data_sets_info_dict, REGRESSIOM_MODEL, CLASSIFICATION_MODEL
from Univariate_Outier_Detection import univariate_outlier_mask
import os

import time


outdir = os.getcwd() + '/output/' + str(time.time())
if not os.path.exists(outdir):
    os.umask(0)
    os.makedirs(outdir)
f = open(outdir + '/results.csv', 'w')
f2 = open(outdir + '/iterations.csv', 'w')



print("", file=f)





def evaluate_regression_model(predictions,y_test):
    return mean_absolute_error(predictions, y_test)


def evaluate_classification_model(predictions,y_test):
    precision_w = precision_score(y_test, predictions, average='weighted')
    recall_w = recall_score(y_test, predictions, average='weighted')
    auc= roc_auc_score(y_test, predictions)
    return auc, precision_w, recall_w


def impute(df):
    simple_imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    imputed_df= pd.DataFrame(simple_imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
    return  imputed_df


def detect_and_treat_outliers(df, info):

    best_results={}
    #scores_after_treatment = []
    print("***************************************************************")
    data_type= info["type"]
    IS_REGRESSION= True if data_type== "regression" else False
    #select the model to use for evaluation
    model= REGRESSIOM_MODEL if IS_REGRESSION else CLASSIFICATION_MODEL
    # split into feature and label elements,  #todo assuming the last column is the label- need to make more generic (save label in info of dataset)
    X, y = df[list(df.columns)[:-1]], df[list(df.columns)[-1]]
    feature_columns= list(X.columns)
    # split into train and test sets   #todo- k-fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    # summarize the shape of the training dataset
    print(X_train.shape, y_train.shape)
    best_method, best_result_multivariate, number_of_outliers_dict = multivariate_outliers(
        IS_REGRESSION, X_test, X_train, model, y_test, y_train)

    print("best model and best result multivariate outlier detection:")
    print(best_method)
    print(best_result_multivariate)
    print(number_of_outliers_dict[best_method])

    print("best model and best result multivariate outlier detection:", file=f)
    print(best_method, file=f)
    print(best_result_multivariate, file= f)

    best_results["multi"]= {"best_method":best_method, "best_result": best_result_multivariate}
                                                                                    #todo -
    best_accuracy_global, filtered_col, filtered_columns_dict = univariate_outliers(X_test, X_train, feature_columns, data_type,
                                                                                    model,
                                                                                    y_test, y_train, IS_REGRESSION)
    print("best accuracy for univariate outlier detection and treatment")
    print(best_accuracy_global)

    print("best accuracy for univariate outlier detection and treatment", file= f)
    print(best_accuracy_global, file= f)

    # print the treated columns by order and statitics
    if(len(filtered_col)>0):
        print(",".join(filtered_columns_dict[filtered_col[0]].keys()), file= f2)
        for col in filtered_col:
            print(",".join([str(x) for x in filtered_columns_dict[col].values()]), file=f2)
    best_results["uni"]= { "best_result": best_accuracy_global, "filtered_columns_dict": filtered_columns_dict}

    return best_results

def detect_and_treat_outliers_one_method(df, info):
    #todo- make functions for duplicate code
    #todo - also try one treatment detection_method each time
    #scores_after_treatment = []
    print("***************************************************************")
    data_type= info["type"]
    IS_REGRESSION= True if data_type== "regression" else False
    #select the model to use for evaluation
    model= REGRESSIOM_MODEL if IS_REGRESSION else CLASSIFICATION_MODEL
    # split into feature and label elements
    X, y = df[list(df.columns)[:-1]], df[list(df.columns)[-1]]
    feature_columns= list(X.columns)
    # split into train and test sets   #todo- why before outliers ?? if necessary- write in the document
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    # summarize the shape of the training dataset
    print(X_train.shape, y_train.shape)


    best_accuracy= np.inf if IS_REGRESSION else 0
    for detection_method in ["mad", "percentile", "iqr"]:
        for treatment_method in ["drop", "median", "floored_and_capped"]:
            # todo -also loop over treatments

            print("now only "+ detection_method + "   ********  and " + treatment_method )
            best_accuracy_for_treatment, filtered_col, filtered_columns_dict = univariate_outliers(X_test, X_train, feature_columns, data_type, model,
                                                                                 y_test, y_train, IS_REGRESSION, [detection_method], [treatment_method])
            if (IS_REGRESSION and best_accuracy_for_treatment < best_accuracy) or (not IS_REGRESSION and best_accuracy_for_treatment > best_accuracy) :
                    best_accuracy= best_accuracy_for_treatment

    print("best accuracy for univarate outlier detection and treatment one detection_method (baseline)")
    print(best_accuracy)

    print("best accuracy for univarate outlier detection and treatment one detection_method", file= f)
    print(best_accuracy, file= f)

    return best_accuracy

def univariate_outliers(X_test, X_train, columns, data_type, model, y_test, y_train,
                        IS_REGRESSION, outlier_detecting_methods= ["mad", "percentile", "iqr"], outlier_treating_methods= ["drop", "median", "floored_and_capped"]):
    """"
    univariate outliers

    """
    filtered_columns_dict = {}
    filtered_col = []
    improved_flag = True
    best_accuracy_global = np.infty if data_type == "regression" else 0
    X_train_copy, y_train_copy = X_train, y_train
    # as long as their are more columns to try filtering, and as long a the previous iteration improved
    while len(filtered_col) < len(columns) and improved_flag:
        best_accuracy_for_round = np.infty if IS_REGRESSION else 0
        best_col = None
        best_outlier_mask = None
        improved_flag = False
        for col in [c for c in columns if c not in filtered_col]:
            data = X_train_copy[col]
            # this dict will contain all the results for all combinations of detection and treating methods
            col_results = {}
            for detection_method in outlier_detecting_methods:
                # print("&&&&&&&&&&&&&&&&&&&&&&&&")
                # print("detection_method:  " + str(detection_method))
                outlier_mask = univariate_outlier_mask(data, detection_method)
                number_of_outliers = list(outlier_mask).count(False)
                if (number_of_outliers > len(data) / 2):
                    continue
                # print("number of outliers")
                # print(number_of_outliers)
                treated_datasets_dict = get_treated_datasets(X_train_copy.copy(), col, outlier_mask,
                                                             y_train_copy, outlier_treating_methods)
                # print(X_train_masked.shape, y_train_masked.shape)
                for outlier_treating_method, Xy_train in treated_datasets_dict.items():
                    accuracy = fit_model_and_evaluate_accuracy(Xy_train[0], Xy_train[1], X_test, y_test, model,
                                                               IS_REGRESSION)
                    col_results[(detection_method, outlier_treating_method)] = accuracy
                # choose the result of the best treatment (in terms of accuracy)
            from operator import itemgetter
            # find key and value of best result (min for regression and max for classification)
            if len(col_results) == 0:
                continue
            index_of_best_combo, best_accuracy_for_col = min(enumerate(col_results.values()),
                                                             key=itemgetter(1)) if IS_REGRESSION else max(
                enumerate(col_results.values()), key=itemgetter(1))
            best_detection_and_treatment = list(col_results.keys())[index_of_best_combo]
            # best_treatment= ["drop", "median_replaced", "capped_floored"][index_of_best_combo]
            # print(col)
            # print('accuracy : %.3f' % accuracy)
            if (
                    IS_REGRESSION and best_accuracy_for_col < best_accuracy_global and best_accuracy_for_col < best_accuracy_for_round) \
                    or (
                    not IS_REGRESSION and best_accuracy_for_col > best_accuracy_global and best_accuracy_for_col > best_accuracy_for_round):
                best_accuracy_for_round = best_accuracy_for_col
                improved_flag = True
                best_col = col
                best_column_treatment_and_detection = best_detection_and_treatment
                #best_outlier_mask= outlier_mask
                # todo- we should only add at the end, after all detection and  treatments
            # scores_after_treatment.append(
            #     {"detection detection_method": best_detection_and_treatment[0], "number of outliers": number_of_outliers,
            #      "treatment detection_method": best_detection_and_treatment[1],
            #      "score": accuracy})
        if improved_flag:
            improvement = abs(best_accuracy_for_round - best_accuracy_global)
            best_accuracy_global = best_accuracy_for_round
            best_column_detection = best_column_treatment_and_detection[0]
            best_column_treatment = best_column_treatment_and_detection[1]
            outlier_mask = univariate_outlier_mask(X_train_copy[best_col], best_column_detection)
            number_of_outliers = list(outlier_mask).count(False)
            if (best_column_treatment == "drop"):
                X_train_copy, y_train_copy = X_train_copy.iloc[outlier_mask, :], y_train_copy.iloc[
                    outlier_mask]
            elif (best_column_treatment == "median"):
                X_train_copy = replace_outlier_by_median(X_train_copy.copy(), col, outlier_mask)
            else:
                X_train_copy = replace_outlier_by_flooring_and_capping(X_train_copy.copy(), col, outlier_mask)
            filtered_col.append(best_col)
            filtered_columns_dict[best_col] = {"column name": best_col, "number of outliers": number_of_outliers,
                                               "best detection": best_column_detection,
                                               "best treatment": best_column_treatment,
                                               "performance after treating": best_accuracy_global,
                                               "improvement": improvement}
    return best_accuracy_global, filtered_col, filtered_columns_dict


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
This method evaluates the model without 
"""
def evalulate_with_outliers(df, info):
    data_type= info["type"]
    model = REGRESSIOM_MODEL if data_type=="regression" else CLASSIFICATION_MODEL
    # split into feature and label elements
    columns = list(df.columns)
    X, y = df[columns[:-1]], df[columns[-1]]
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    accuracy= fit_model_and_evaluate_accuracy(X_train,y_train,   X_test,y_test, model, data_type=="regression" )
    print (accuracy)
    print (accuracy, file=f)
    return accuracy

def evaluate_with_normailization(df, info):
    data_type= info["type"]
    model = REGRESSIOM_MODEL if data_type=="regression" else CLASSIFICATION_MODEL
    # split into feature and label elements
    columns = list(df.columns)
    X, y = df[columns[:-1]], df[columns[-1]]
    # split into train and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    qt_transformer = QuantileTransformer(output_distribution="uniform",
                                         random_state=115)
    X_train = qt_transformer.fit_transform(X_train)
    X_test = qt_transformer.transform(X_test)

    accuracy= fit_model_and_evaluate_accuracy(X_train,y_train,   X_test,y_test, model, data_type=="regression")
    print (accuracy)
    print (accuracy, file=f)
    return accuracy


def main():
    results_title= ["dataset","baseline accuracy", "multivariate  accuracy"," multivariate method", "univariate accuracy", "univariate method combinations"]
    print(" & ".join(results_title))
    # iterate over all datasets
    for dataset, info in data_sets_info_dict.items():
        print(info)
        print(dataset)
        print(info,file=f)
        print(dataset, file=f)
        #  multivariate outliers
        print("**************************************")
        print("**************************************", file=f)
        print(dataset, file=f2)
        print("**************************************", file=f2)

        df = read_csv(info["path"])
        df = impute(df)  # todo- more imputation methods ?- FWork
        print("accuracy with outliers (baseline)")
        print("accuracy with outliers (baseline)", file = f)
        accuracy_with_outliers= round(evalulate_with_outliers(df, info),3)

        print("accuracy with normalization (baseline)")
        print("accuracy with normalization (baseline)", file = f)
        accuracy_with_normalization= round(evalulate_with_outliers(df, info),3)
        print(accuracy_with_normalization)


        print("accuracy without outliers")
        print("accuracy without outliers", file=f)
        best_results= detect_and_treat_outliers(df, info)
        multivariate_method, multivariate_accuracy = best_results["multi"]["best_method"],   round(best_results["multi"]["best_result"])
        univariate_accuracy = best_results["uni"]["best_result"]
        univariate_combos= [str((x["best detection"], x["best treatment"])) for x in best_results["uni"]["filtered_columns_dict"].values()]
        results_list= [dataset, str(accuracy_with_outliers),str(multivariate_accuracy), multivariate_method, str(univariate_accuracy),str(univariate_combos) ]
        print (" & ".join(results_list))
main()


#univariate outliers
#create a dataframe with outliers flags for each columns
outlier_df= pd.DataFrame()
# for col in df.columns:
#     outlier_df[col] = iqr_anomaly_detector(df[col])
#plot_anomalies(iqr_df)
#print(outlier_df)


f.close()
f2.close()
