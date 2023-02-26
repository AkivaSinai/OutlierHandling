import numpy as np
from Configurations import *

#todo all the constants should be constant variables!

#todo- if have time , experiemnt different threshold_factor valiues. (future work)
def mad_based_outlier(data, thresh=MAD_THRESHOLD):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if len(data.shape) == 1:
            data = data[:, None]
        median = np.median(data, axis=0)
        diff = np.sum((data - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        if(med_abs_deviation==0):
            med_abs_deviation=0.1 #todo- wanted to avoid division by zero, dirty- need to think why this happens
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score < thresh


def percentile_based_outlier(data, threshold=PERCENTILE_THRESHOLD):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return list((data > minval) | (data < maxval))


def univariate_outlier_mask(data, method):
    if(method== "mad"):
        return mad_based_outlier(data)
    elif(method== "percentile"):
        return percentile_based_outlier(data)
    elif(method== "iqr"):
        return iqr_anomaly_detector(data)


def iqr_anomaly_detector(data, threshold=IQR_THRESHOLD):
    def find_anomalies(value, lower_threshold, upper_threshold):
        if value < lower_threshold or value > upper_threshold:
            return False
        else:
            return True
    quartiles = dict(data.quantile([.25, .50, .75]))
    quartile_3, quartile_1 = quartiles[0.75], quartiles[0.25]
    iqr = quartile_3 - quartile_1
    lower_threshold = quartile_1 - (threshold * iqr)
    upper_threshold = quartile_3 + (threshold * iqr)
    #print(f"Lower threshold_factor: {lower_threshold}, \nUpper threshold_factor: {upper_threshold}\n")
    return list(data.apply(find_anomalies, args=(lower_threshold, upper_threshold)))