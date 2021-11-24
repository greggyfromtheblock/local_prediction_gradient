import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import statsmodels.api as stats

def attribution2df(attributions, feature_names, loader):
    """
    Computes the mean absolute attribution from a local attribution, returns a dataframe with feature names and global
    attribution score
    :parameter attributions: local attributions
    :parameter feature_names: feature names
    :parameter loader: dataloader to indicate shape
    """
    attribution_sum = torch.abs(attributions).sum(0) if not isinstance(attributions, torch.Tensor) else torch.abs(attributions).detach().numpy().sum(0)
    attribution_norm_sum = attribution_sum / np.linalg.norm(attribution_sum, ord=1)
    axis_data = np.arange(loader.shape[1])
    data_labels = list(map(lambda idx: feature_names[idx], axis_data))
    df = pd.DataFrame({'feature': data_labels,
                       'importance': attribution_norm_sum})
    sorted_df = df.reindex(df.importance.abs().sort_values(ascending=False).index)
    return sorted_df

def log_global_importance(trainer, dataframe, name='', experiment_name='', n=0, seed=0, log_images=False):
    """
    Logs the global importance plot and csv to neptune
    :param trainer: pytorch lightning trainer instance
    :param name: string to be used as the plot title name, as well as log name on neptune
    :param log_images: option to log images to neptune
    """

    # Sort dataframe
    dataframe = dataframe.reindex(dataframe.importance.abs().sort_values(ascending=False).index)

    if not os.path.exists(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor/{experiment_name}'):
        os.mkdir(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor/{experiment_name}')
    if not os.path.exists(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor/{experiment_name}/global'):
        os.mkdir(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor/{experiment_name}/global')

    dataframe.to_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor/{experiment_name}/global/{name}-{experiment_name}-n{n}-{seed}.csv')

def sample_from_population(n=100, mu=0, sd=1, r=0):
    """
    :param n: number of samples
    :param mu: scale parameter
    :param sd: standard deviation
    :param r: correlation coefficient
    :return:
    """
    # sample r from distribution depending on n and r
    r_sd = np.sqrt(1/n) * (1-r**2)
    sample_r = np.random.normal(loc=r, scale=r_sd)
    sample_r = -1 if sample_r < -1 else sample_r
    sample_r = 1 if sample_r > 1 else sample_r

    # sample mu from distribution depending on n and sd
    mu_sd = sd/np.sqrt(n)
    sample_mu = np.random.normal(loc=mu, scale=mu_sd)

    # sample sd from distribution depending on n and sd
    sd_sd = sd/np.sqrt(2*n)
    sample_sd = np.random.normal(loc=sd, scale=sd_sd)
    return dict(mu=sample_mu, sd=sample_sd, r=sample_r)


def create_correlated_var(x, mu=0.0, sd=1.0, r=0.9, empirical=False):
    """
    Creates a random normally distributed array with the specified correlati`on
    :param x: existing array to correlate to
    :param mu: desired mean of the returned array
    :param sd: desired stdev of the returned vector
    :param r: desired correlation between existing and returned vectors
    :param empirical: if true, mu, sd, and r specify the empirical not the population mean
    """
    n = len(x)
    if not empirical:
        sample_params=sample_from_population(n, mu, sd, r)
        mu = sample_params['mu']
        sd = sample_params['sd']
        r = sample_params['r']

    x = scale(x)
    y = np.random.normal(size=n)
    e = stats.OLS(y, x).fit().resid
    z = r * scale(x) + np.sqrt(1-r**2) * scale(e)
    return mu + sd * z
