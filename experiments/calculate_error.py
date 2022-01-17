import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from cga import cga
import pandas as pd
import neptune.new as neptune
import wandb
import pytorch_lightning as pl
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from source.tasks import DeepSurv

from captum.attr import *
from captum.metrics import *
from captum._utils.models.linear_model import SkLearnLinearRegression
from source.wrappers import ForwardWrapper
import os
import seaborn as sns


def load_details(run):
    dir = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities'
    path = f'{dir}/{run.tags[0]}_attribute_details.csv'
    details_df = pd.read_csv(path)
    return details_df, path


def output_diff(model, orig_features, resampled_features):
    orig_output, *_ = model(orig_features)
    resampled_output, *_ = model(resampled_features)
    with torch.no_grad():
        diff = torch.sub(orig_output, resampled_output)
    return diff.detach().numpy()


def load_attributions(run, method):
    attr_x = np.genfromtxt(run.config[f'{method}_x_path'], delimiter=',')
    attr_y = np.genfromtxt(run.config[f'{method}_y_path'], delimiter=',')
    return attr_x, attr_y


def calculate_error(details_x, details_y):
    for detail in [details_x, details_y]:
        outlier_cs = np.percentile(detail.change_slope, [0.5, 99.5])
        clipped_cs = np.clip(detail.change_slope.to_numpy(), *outlier_cs)
        cs_norm = clipped_cs / abs(clipped_cs).max()
        detail.loc[:,'change_slope_norm'] = cs_norm

        outlier_attr = np.percentile(detail.attribution, [0.5, 99.5])
        clipped_attr = np.clip(detail.attribution.to_numpy(), *outlier_attr)
        detail.loc[:,'attribution_norm'] = clipped_attr / abs(clipped_attr).max()
        detail.loc[:, 'error'] = abs(detail.attribution_norm - detail.change_slope_norm)


def view_error(project, id, method):
    api = wandb.Api()
    run = api.run(f"cardiors/{project}/{id}")
    x, y = change_slope_singular(run, method)
    y.plot.scatter('x_orig', 'y_orig', c='error', cmap='PuBu')
    x.plot.scatter('x_orig', 'y_orig', c='error', cmap='PuBu')

# CREATE TUPLES FOR MODEL DIFF


def col2numpy(df, colname):
    """
    convert a dataframe column to a numpy array
    :param df: dataframe
    :param colname: column name to convert to numpy array
    :return:
    """
    vals = df.loc[:, colname].values.tolist()
    vals_numpy = np.array([np.array(eval(row)) for row in vals])
    return vals_numpy


def clip_norm2d(array):
    """
    clips and normalizes the array to the range of [-1, 1]
    :param array: array to be normalized
    :return: normalized array
    """
    array_norm = np.zeros((array.shape))
    for i in range(array.shape[1]):
        column = array[:, i]
        outliers = np.percentile(column, [0.5, 99.5])
        clipped_array = np.clip(column, *outliers)
        col_norm = clipped_array / abs(clipped_array).max()
        array_norm[:, i] = col_norm
    return  array_norm


def clip_norm1d(array):
    """
    clips and normalizes the array to the range of [-1, 1]
    :param array:  array to be normalized
    :return: normalized array
    """
    outliers = np.percentile(array, [0.5, 99.5])
    clipped_array = np.clip(array, *outliers)
    array_norm = clipped_array / abs(clipped_array).max()
    return array_norm


def get_errors(change_slope_array, attr_array):
    """
    calculates squared & absolute error between the attribution value and change_slope
    :param change_slope_array: change slope array (2 dimensional)
    :param attr_array: attribution values array (1 dimensional)
    :return: squared_error, absolute error (both 2 dimensional)
    """
    squared_error = np.zeros((change_slope_array.shape))
    absolute_error = np.zeros((change_slope_array.shape))

    for j in range(change_slope_array.shape[0]):
        for i in range(change_slope_array.shape[1]):
            attr = attr_array[i]
            cs = change_slope_array[j, i]
            squared_error[j, i] = np.square(attr - cs)
            absolute_error[j, i] = np.absolute(attr - cs)

    mae = np.mean(absolute_error, axis=1)
    mse = np.mean(squared_error, axis=1)
    rmse = np.sqrt(np.mean(squared_error, axis=1))
    return squared_error, absolute_error, rmse, mse, mae

def resampling_error(experiment_id, method):
    """
    creates a dataframe with resampling error for each point
    :param experiment_id: experiment identifier
    :param method: feature ablation method to calculate errors on
    :return:
    """

    experiment_id = experiment_id
    api = wandb.Api()
    runs =  api.runs('cardiors/interpretability',
                     filters={"$and": [{'tags': f'{experiment_id}'}, {'tags': 'resample_multiplicities'}, {'state': 'finished'}]})

    # For testing purpose
    runs = [runs[0]]
    for run in runs:
        # load resampling dataframe and model
        xdf = pd.read_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_x.csv', index_col=0)
        ydf = pd.read_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_y.csv', index_col=0)
        model = DeepSurv.load_from_checkpoint(run.config['checkpoint_path'])

        # resampling tensors for xdf
        orig_tensor_x = torch.Tensor(xdf[['x_orig', 'y_orig']].to_numpy(dtype='float64'))
        resamp_vals = col2numpy(xdf, 'x_resampling')
        constant_val = xdf['y_orig'].to_numpy(dtype='float64')
        resamp_tensors_x = []
        for i in range(resamp_vals.shape[1]):
            # modified attribute == noise_x -> resamp_tensor = (resamp, constant)
            resamp_tensor = torch.Tensor(np.concatenate((resamp_vals[:, i].reshape(-1, 1), constant_val.reshape(-1, 1)), axis=1))
            resamp_tensors_x.append(resamp_tensor)

        orig_tensor_y = torch.Tensor(ydf[['x_orig', 'y_orig']].to_numpy(dtype='float64'))
        # resampling tensors for ydf
        resamp_vals = col2numpy(ydf, 'y_resampling')
        constant_val = ydf['x_orig'].to_numpy(dtype='float64')
        resamp_tensors_y = []
        for i in range(resamp_vals.shape[1]):
            # modified attribute == noise_y -> resamp_tensor = (constant, resamp)
            resamp_tensor = torch.Tensor(np.concatenate((constant_val.reshape(-1, 1), resamp_vals[:, i].reshape(-1, 1)), axis=1))
            resamp_tensors_y.append(resamp_tensor)

        # CALCULATE MODEL DIFF + CHANGE SLOPE

        # load attributions
        attr_x, attr_y = load_attributions(run, method=method)
        attr_x = attr_x[:, 0]
        attr_y = attr_y[:, 1]

        orig_output = model(orig_tensor_x)[0].detach().numpy()
        resamp_outputs = None
        for i in range(resamp_vals.shape[1]):
            resamp_output = model(resamp_tensors_x[i])[0].detach().numpy()
            resamp_outputs = resamp_output if resamp_outputs is None else np.concatenate((resamp_outputs, resamp_output), axis=1)

        # calculate change slope
        model_diff = np.subtract(orig_output, resamp_outputs)
        resampling_diff = col2numpy(xdf, 'resampling_diff')
        change_slope = np.true_divide(model_diff, resampling_diff)
        mean_cs = np.mean(change_slope, axis=1)

        xdf.loc[:, 'model_diff'] = model_diff.tolist()
        xdf.loc[:, 'change_slope'] = change_slope.tolist()
        xdf.loc[:, 'mean_cs'] = mean_cs
        xdf.loc[:, 'attribution_x'] = attr_x

        # calculate change slope
        orig_output = model(orig_tensor_y)[0].detach().numpy()
        resamp_outputs = None
        for i in range(resamp_vals.shape[1]):
            resamp_output = model(resamp_tensors_y[i])[0].detach().numpy()
            resamp_outputs = resamp_output if resamp_outputs is None else np.concatenate((resamp_outputs, resamp_output), axis=1)

        model_diff = np.subtract(orig_output, resamp_outputs)
        resampling_diff = col2numpy(ydf, 'resampling_diff')
        change_slope = np.true_divide(model_diff, resampling_diff)
        mean_cs = np.mean(change_slope, axis=1)

        ydf.loc[:, 'model_diff'] = model_diff.tolist()
        ydf.loc[:, 'change_slope'] = change_slope.tolist()
        ydf.loc[:, 'mean_cs'] = mean_cs
        ydf.loc[:, 'attribution_y'] = attr_y

        # Save dataframes
        EVALUATION_DIR = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities/evaluation'
        if not os.path.exists(f'{EVALUATION_DIR}/{experiment_id}'):
            os.mkdir(f'{EVALUATION_DIR}/{experiment_id}')

        if not os.path.exists(f'{EVALUATION_DIR}/{experiment_id}/change_slope'):
            os.mkdir(f'{EVALUATION_DIR}/{experiment_id}/change_slope')

        seed = eval(run.config['_content']['experiment'])['datamodule_kwargs']['seed']
        xdf.to_csv(f'{EVALUATION_DIR}/{experiment_id}/change_slope/x_{method}_{seed}.csv')
        ydf.to_csv(f'{EVALUATION_DIR}/{experiment_id}/change_slope/y_{method}_{seed}.csv')
        return xdf, ydf


if __name__ == '__main__':
    method = sys.argv[1]
    experiment_ids = ['p0.00', 'p0.25', 'p0.5', 'p0.75']
    for experiment_id in experiment_ids:
        resampling_error(experiment_id, method)