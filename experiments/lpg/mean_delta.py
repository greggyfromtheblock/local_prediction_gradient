print('you are using the correct script')
import sys
sys.path.append('/home/ruyogagp/medical_interpretability')
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

RESULTS_DIR = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities'
DATA_DIR = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities'
EVALUATION_DIR = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities/evaluation'

def load_attribute_details(experiment_id):
    dir = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities'
    path = f'{dir}/{experiment_id}_attribute_details.csv'
    details_df = pd.read_csv(path)
    return details_df, path

# TODO: this works if the second tag is the experiment identifier
def load_details(run):
    dir = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities'
    try:
        path = f'{dir}/{run.tags[0]}_attribute_details.csv'
        details_df = pd.read_csv(path)
    except:
        path = f'{dir}/{run.tags[1]}_attribute_details.csv'
        details_df = pd.read_csv(path)
    print(f'reading {path}')
    return details_df, path

def output_diff(model, orig_features, resampled_features):
    orig_output, *_ = model(orig_features)
    resampled_output, *_ = model(resampled_features)
    with torch.no_grad():
        diff = torch.sub(orig_output, resampled_output)
    return diff.detach().numpy()

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

def clip_norm2d(array, tail):
    """
    clips and normalizes the array to the range of [-1, 1]
    :param array: array to be normalized
    :return: normalized array
    """
    array_norm = np.zeros((array.shape))
    c_array = np.zeros((array.shape))
    for i in range(array.shape[0]):
        column = array[i, :]
        outliers = np.percentile(column, [tail, 100-tail])
        clipped_array = np.clip(column, *outliers)
        col_norm = clipped_array / abs(clipped_array).max()
        #TODO: !!!!!FIX THIS!!!!
        array_norm[i, :] = col_norm
        c_array[i, :] = clipped_array
    return  array_norm, c_array

def clip_norm1d(array, tail):
    """
    clips and normalizes the array to the range of [-1, 1]
    :param array:  array to be normalized
    :return: normalized array
    """
    outliers = np.percentile(array, [tail, 100-tail])
    clipped_array = np.clip(array, *outliers)
    array_norm = clipped_array / abs(clipped_array).max()
    return array_norm, clipped_array

def load_attributions(run, method, n_features=5):
    attribution_list = [np.genfromtxt(run.config[f'{method}_feature{idx}_path'], delimiter=',') for idx in range(n_features)]
    return attribution_list

def reject_outliers(data, m = 150.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    max = data[s<m].max()
    min = data[s<m].min()
    out = np.where(s<m, data, np.where(data>0, max, min))
    return out

def clip2d(array):
    clipped_array = np.zeros((array.shape))
    for i in range(array.shape[0]):
        clipped_array[i, :] = reject_outliers(array[i, :])
    return clipped_array

def create_resampling_df(experiment_id, n_resampling=100, n_features=6):
    """
    Modify the original resampling dataframe to create per-feature resampling df with resampling differences
    :param experiment_id: experiment identifier
    :return: None -> saves resampling dataframe
    """
    api = wandb.Api()
    runs =  api.runs('cardiors/interpretability',
                     filters={"$and": [{'tags': f'{experiment_id}'}, {'tags': 'resample_multiplicities'}, {'state': 'finished'}]})
    run = runs[0]
    details_df, path = load_attribute_details(experiment_id)

    modified_attributes = [f'noise{idx}' for idx in range(n_features)]
    for idx in range(n_features):
        df = details_df[details_df.modified_attribute == f'noise{idx}']
        colnames = [f'feature{idx}_intervention{n}' for n in range(n_resampling)]

        # TODO
        rcolnames = [f'feature1_intervention{n}' for n in range(n_resampling)]
        rintervention_df = df.loc[:, rcolnames]
        rintervention_df['intervention'] = rintervention_df.values.tolist()

        intervention_df = df.loc[:, colnames]
        intervention_df['intervention'] = intervention_df.values.tolist()
        resampling_df = pd.DataFrame({f'feature{idx}_resampling': intervention_df.intervention,
                                      'feature1_rresampling': rintervention_df.intervention,
                                      f'feature{idx}_orig': df[f'feature{idx}_orig'],
                                      'time_orig': df.time_orig,
                                      'event_orig': df.event_orig})
        # return resampling_df, df
        for n in range(n_features):
            resampling_df[f'feature{n}_orig'] = df[f'feature{n}_orig']

        orig_vals = resampling_df.loc[:, f'feature{idx}_orig'].values
        resamp_vals = resampling_df.loc[:, f'feature{idx}_resampling'].values.tolist()
        resamp_vals_numpy=np.array([np.array(row) for row in resamp_vals])
        orig_vals_array = np.array([[val] * n_resampling for val in orig_vals])
        resampling_df.loc[:, 'resampling_diff'] = np.subtract(orig_vals_array, resamp_vals_numpy).tolist()

        # exp_id_trunc = experiment_id[:-2]
        resampling_df.to_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_feature{idx}.csv')
        print(f' wrote /data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_feature{idx}.csv')


def calculate_change_slope(experiment_id, n_features=5):
    """
    Calculates change slope for each run
    :param experiment_id: experiment identifier
    :return: None -> saves the change slope per run
    """
    api = wandb.Api()
    runs =  api.runs('cardiors/interpretability',
                     filters={"$and": [{'tags': f'{experiment_id}'}, {'tags': 'resample_multiplicities'}, {'state': 'finished'}]})
    print(len(runs))
    for run in runs:
        for idx in range(n_features):
            resampling_df = pd.read_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_feature{idx}.csv', index_col=0)
            print(f'read /data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_resampling_feature{idx}.csv')
            intervention_df = pd.read_csv(f'/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/resample_multiplicities/{experiment_id}_attribute_details.csv')
            intervention_df = intervention_df[intervention_df['modified_attribute']==f'noise{idx}']
            model = DeepSurv.load_from_checkpoint(run.config['checkpoint_path'])

            # create orig and resampling tensors
            orig_features = [f'feature{idx}_orig' for idx in range(n_features)]
            orig_tensor = torch.Tensor(resampling_df[orig_features].to_numpy(dtype='float64'))
            resamp_tensors = []
            resamp_vals = col2numpy(resampling_df, f'feature{idx}_resampling')
            for i_intervention in range(100):
                resamp_features = [f'feature{n}_intervention{i_intervention}' for n in range(n_features)]
                resamp_tensors.append(torch.Tensor(intervention_df.loc[:, resamp_features].to_numpy(dtype='float64')))

            # calculate model outputs
            orig_output = model(orig_tensor)[0].detach().numpy()
            resamp_outputs = None
            for i in range(resamp_vals.shape[1]):
                resamp_output = model(resamp_tensors[i])[0].detach().numpy()
                resamp_outputs = resamp_output if resamp_outputs is None else np.concatenate((resamp_outputs, resamp_output), axis=1)

            # calculate change_slope
            model_diff = np.subtract(orig_output, resamp_outputs)
            resampling_diff = col2numpy(resampling_df, 'resampling_diff')
            change_slope = np.true_divide(model_diff, resampling_diff)
            clipped_mean_cs = np.mean(clip2d(change_slope), axis=1)
            mean_cs = np.mean(change_slope, axis=1)
            median_cs = np.median(change_slope, axis=1)

            # save values in dataframe
            resampling_df.loc[:, 'orig_output'] = orig_output
            resampling_df.loc[:, 'resamp_output'] = resamp_outputs.tolist()
            resampling_df.loc[:, 'model_diff'] = model_diff.tolist()
            resampling_df.loc[:, 'change_slope'] = change_slope.tolist()
            resampling_df.loc[:, 'mean_cs'] = mean_cs
            resampling_df.loc[:, 'clipped_mean_cs'] = clipped_mean_cs
            resampling_df.loc[:, 'median_cs'] = median_cs
            methods = ['FeaturePermutation',
                       'IntegratedGradients',
                       'InputxGradient',
                       'Lime',
                       'KernelExplainer',
                       'DeepExplainer']


            for method in methods:
                attribution_list = load_attributions(run, method, n_features=n_features)
                resampling_df.loc[:, f'{method}_feature{idx}'] = attribution_list[idx].tolist()

            # Save dataframes
            EVALUATION_DIR = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities/evaluation'
            if not os.path.exists(f'{EVALUATION_DIR}/{experiment_id}'):
                os.mkdir(f'{EVALUATION_DIR}/{experiment_id}')

            if not os.path.exists(f'{EVALUATION_DIR}/{experiment_id}/change_slope'):
                os.mkdir(f'{EVALUATION_DIR}/{experiment_id}/change_slope')

            # very whack
            seed = int(run.config['Lime_feature0_path'].split('_')[-1][:-4])

            resampling_df.to_csv(f'{EVALUATION_DIR}/{experiment_id}/change_slope/feature{idx}_{seed}.csv')
            print(f'Wrote {EVALUATION_DIR}/{experiment_id}/change_slope/feature{idx}_{seed}.csv')

def reject_outliers(data, m = 150.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    max = data[s<m].max()
    min = data[s<m].min()
    out = np.where(s<m, data, np.where(data>0, max, min))
    return out

def calculate_mean_error(experiment_id, n_features=5):
    for idx in range(n_features):
        feature = f'feature{idx}'
        dir = os.listdir(f'{EVALUATION_DIR}/{experiment_id}/change_slope/')
        paths = []
        for filename in dir:
            featurename = filename.split('_')[0]
            if featurename == feature:
                paths.append(f'{EVALUATION_DIR}/{experiment_id}/change_slope/{filename}')

        mean_change_slopes = None
        median_change_slopes = None
        unclipped_mean_change_slopes = None
        for path in paths:
            df = pd.read_csv(path, index_col=0)
            unclipped_mean_change_slope = df.mean_cs.to_numpy().reshape(-1, 1)
            unclipped_mean_change_slopes = unclipped_mean_change_slope if unclipped_mean_change_slopes is None else np.concatenate((unclipped_mean_change_slopes, unclipped_mean_change_slope), axis=1)
            # mean change slope
            mean_change_slope = df.clipped_mean_cs.to_numpy().reshape(-1, 1)
            mean_change_slopes = mean_change_slope if mean_change_slopes is None else np.concatenate((mean_change_slopes, mean_change_slope), axis=1)
            # median change slope
            median_change_slope = df.median_cs.to_numpy().reshape(-1, 1)
            median_change_slopes = median_change_slope if median_change_slopes is None else np.concatenate((median_change_slopes, median_change_slope), axis=1)

        unclipped_mean_cs_norm = unclipped_mean_change_slopes / abs(unclipped_mean_change_slopes).max()
        mean_cs_norm = mean_change_slopes / abs(mean_change_slopes).max()
        median_cs_norm = median_change_slopes / abs(median_change_slopes).max()

        methods = ['FeaturePermutation',
                   'IntegratedGradients',
                   'InputxGradient',
                   'Lime',
                   'KernelExplainer',
                   'DeepExplainer']
        try:
            for method in methods:
                attrs = None
                for path in paths:
                    # normalize per seed & aggregate runs per experiment
                    df = pd.read_csv(path, index_col=0)
                    attr = col2numpy(df, f'{method}_feature{idx}')
                    attr = attr / abs(attr).max()
                    attr = attr[:, idx].reshape(-1, 1)
                    attrs = attr if attrs is None else np.concatenate((attrs, attr), axis=1)
                unclipped_absolute_error = np.absolute(unclipped_mean_cs_norm - attrs)
                absolute_error = np.absolute(mean_cs_norm - attrs)
                median_absolute_error = np.absolute(median_cs_norm - attrs)
                mae = np.mean(absolute_error, axis=1)
                unclipped_mae = np.mean(unclipped_absolute_error, axis=1)
                median_mae = np.mean(median_absolute_error, axis=1)
                orig_features = [f'feature{i}_orig' for i in range(n_features)]
                fulldf = df.loc[:, orig_features]
                fulldf.loc[:, 'absolute_error'] = absolute_error.tolist()
                fulldf.loc[:, 'unclipped_absolute_error'] = unclipped_absolute_error.tolist()
                fulldf.loc[:, 'median_absolute_error'] = median_absolute_error.tolist()
                fulldf.loc[:, 'unclipped_MAE'] = unclipped_mae
                fulldf.loc[:, 'MAE'] = mae
                fulldf.loc[:, 'median_MAE'] = median_mae
                fulldf.loc[:, 'method'] = method
                fulldf.to_csv(f'{EVALUATION_DIR}/{experiment_id}/MeanError_feature{idx}_{method}.csv')
                print(f'wrote {EVALUATION_DIR}/{experiment_id}/MeanError_feature{idx}_{method}.csv')
        except:
            print('hello')
            continue

if __name__ == '__main__':
    print('calculating mean error <3')
    calculate_mean_error('linear_revised_1', n_features=4)
    calculate_mean_error('linear_revised_2', n_features=5)


