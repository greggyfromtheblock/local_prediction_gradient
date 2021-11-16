import os
import torch
import numpy as np
import pandas as pd

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