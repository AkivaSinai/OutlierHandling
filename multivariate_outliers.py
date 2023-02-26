from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
# Data processing
import numpy as np
import pandas as pd
# Model
from sklearn.mixture import GaussianMixture

# the function receives a train set and returns a dictionary including filtering masks of
# the data according to various multivariate outlier detection methods
def identify_multivariate_outliers(X_train):
    outlier_models= {"isolation forest": IsolationForest(contamination=0.1),"ellipicit_envelope":EllipticEnvelope(contamination=0.01),
                     "local outlier factor": LocalOutlierFactor(), "one class svm": OneClassSVM(nu=0.01)}
    outlier_masks={}
    for model_name in outlier_models:
        yhat = outlier_models[model_name].fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        outlier_masks[model_name]= mask
    outlier_masks["guassian_model"]= guassian_mixtures_based_outlier_detection(X_train)
    return outlier_masks

"""
This function uses the Gussian mixture model in order to find clusters of (multi-dimensional) data. 
The smales that are furthest from the clusters are marked as outliers 
"""
def guassian_mixtures_based_outlier_detection(X_train, threshold_factor=5):
    # create GMM model
    gmm = GaussianMixture(n_components=3, n_init=5, random_state=42)
    # Fit and predict on the data
    gmm.fit_predict(X_train)
    #  Predict Anomalies Using Percentage Threshold
    score = gmm.score_samples(X_train)
    # Get the score threshold_factor for anomaly
    pct_threshold = np.percentile(score, threshold_factor)
    mask = score > pct_threshold
    return mask