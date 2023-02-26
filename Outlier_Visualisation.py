import os

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv

from Configurations import data_sets_info_dict


def plot_univariate_distribution(data, feature_name= "feature value", title="feature distribution" , outdir= "\\"):
    sns.displot(data,kde = True)
    #label the axis
    plt.xlabel(feature_name, fontsize = 15)
    plt.ylabel("Count", fontsize = 15)
    plt.title(title, fontsize = 15)
    plt.savefig(outdir + col + "_hist",dpi=300, bbox_inches = "tight")
    plt.show()

def plot_univariate_boxplot(data, feature_name="feature value", title="feature boxplot", outdir= "\\"):
    df = read_csv(info["path"])

    #create the boxplot
    ax = sns.boxplot(x = data)
    #add labels to the plot
    ax.set_xlabel(feature_name, fontsize = 15)
    ax.set_ylabel("Variable", fontsize = 15)
    ax.set_title(title, fontsize =20, pad = 20)
    plt.savefig(outdir + col + "_boxplot",dpi=300, bbox_inches = "tight")
    plt.show()

for dataset_name, info in data_sets_info_dict.items():
    df = read_csv(info["path"])
    outdir = os.getcwd() + '/visualistation/' + dataset_name
    if not os.path.exists(outdir):
        os.umask(0)
        os.makedirs(outdir)
    outdir_box = os.getcwd() + '/visualistation/' + dataset_name+ '/boxplot/'
    if not os.path.exists(outdir_box):
        os.umask(0)
        os.makedirs(outdir_box)
    outdir_hist = os.getcwd() + '/visualistation/' + dataset_name+ '/hist_distribution/'
    if not os.path.exists(outdir_hist):
        os.umask(0)
        os.makedirs(outdir_hist)
    for col in df.columns:
        plot_univariate_distribution(df[col], col, title=col+ " feature distribution ", outdir= outdir_hist)
        plot_univariate_boxplot(df[col], col, title=col+ " feature boxplot ",outdir= outdir_box)

