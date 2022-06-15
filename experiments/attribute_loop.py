from riskiano.source.datamodules.simulations import *
from riskiano.source.utils.general import attribution2df
from riskiano.source.tasks.survival import DeepSurv
from riskiano.source.evaluation.tabular import *
from riskiano.source.modules.interpretability import *

import pytorch_lightning as pl
import os
import torch
import pandas as pd
import numpy as np

from captum.attr import *
from captum.metrics import *
from captum._utils.models.linear_model import SkLearnLinearRegression

class CsvDataset(Dataset):
    def __init__(self, root, n_inp):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        self.x , self.y = (torch.from_numpy(self.data[:, 0]),
                           torch.from_numpy(self.data[:, 1]))
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx,:]
    def __len__(self):
        return len(self.data)


def feature_attribution(experiment_name, n, seed, baseline_method='zeros', compute_sens=False, comput_inf=False):
    """
    Runs attribution methods at the end of fit
    :param experiment_name: foler name to save the attributions at + name to id the attribution
    :param n: number of extra correlated variable on the training (for simulation experiment)
    :param seed: seed number of the training (for simulation experiment)
    :param baseline_method: baseline to choose from when computing attributions ['zeros', 'ones', 'average']
    :param compute_sens: wether to compute sensitivity max (this takes gazillion years to run)
    """

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    csv_name = experiment_name.split('_')[-1]
    csv_id = f'{csv_name}-n{n}'

    # Get Datamodule
    Datamodule = SyntheticDatamodule
    datamodule = SyntheticDatamodule(csv_id=csv_id,
                                     batch_size=1024,
                                     dirpath='/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/synthetic/')

    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get Module
    module = SingleLayerPerceptron(input_dim=len(datamodule.features),
                                   output_dim=1,
                                   bias=False)
    coefs = datamodule.get_coxph_coeffs()
    module.fill_weights(coefs)

    # Task
    task = DeepSurv(network=module,
                    batch_size=1024,
                    evaluation_time_points=None,
                    evaluation_quantile_bins=[0.9])
    task.eval()
    task.baseline_hazards(dataset=datamodule.train_ds)

    # Get unpacked data
    feature_names = datamodule.features
    train_loader = task.ext_dataloader(datamodule.train_ds,
                                       batch_size=1024,
                                       num_workers=8,
                                       shuffle=True,
                                       drop_last=False)
    valid_loader = task.ext_dataloader(datamodule.valid_ds,
                                 batch_size=1024,
                                 num_workers=8,
                                 shuffle=True,
                                 drop_last=False)
    # Get a batch from dataloader
    valid_batch = next(iter(valid_loader))
    train_batch = next(iter(train_loader))

    # Unpack batch
    valid_data, *_ = task.unpack_batch(valid_batch)
    train_data, *_ = task.unpack_batch(train_batch)

    def wrapped_model(inp, t):
        return task.predict_risk(inp, t)

    # tau for predict_risk
    t = torch.tensor([datamodule.eval_timepoint])
    t_repeat = torch.tensor([datamodule.eval_timepoint]).repeat([task.batch_size, 1])

    # Baseline selection
    if baseline_method == 'zeros':
        baseline = torch.zeros(train_data.shape[0], train_data.shape[1])
    elif baseline_method == 'ones':
        baseline = torch.ones(train_data.shape[0], train_data.shape[1])
    elif baseline_method == 'average':
        average = torch.mean(train_data)
        baseline = torch.empty(train_data.shape[0], train_data.shape[1]).fill_(average)

    # Perturb function for LIME
    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.0001, inputs.shape)).float()
        return noise, inputs - noise

    print('im attributing')
    trainer = 0
    # Compute attributions
    fa = FeatureAblation(wrapped_model)
    fa_attr_valid = fa.attribute(valid_data, additional_forward_args=t, baselines=baseline)
    fa_df = attribution2df(fa_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, fa_df, name='FeatureAblation',
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)

    fp = FeaturePermutation(wrapped_model)
    fp_attr_valid = fp.attribute(valid_data, additional_forward_args=t)
    fp_df = attribution2df(fp_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, fp_df, name='FeaturePermutation',
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)

    ig = IntegratedGradients(wrapped_model)
    ig_attr_valid, ig_delta = ig.attribute(valid_data,
                                           baselines=baseline,
                                           n_steps=100,
                                           additional_forward_args=t,
                                           internal_batch_size=task.batch_size,
                                           return_convergence_delta=True)
    ig_df = attribution2df(ig_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, ig_df, name="IntegratedGradients",
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)


    shap = ShapleyValueSampling(wrapped_model)
    shap_attr_valid = shap.attribute(valid_data, additional_forward_args=t, baselines=baseline)
    shap_df = attribution2df(shap_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, shap_df, name="ShapleyValueSampling",
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)


    ixg = InputXGradient(wrapped_model)
    ixg_attr_valid = ixg.attribute(valid_data, additional_forward_args=t)
    ixg_df = attribution2df(ixg_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, ixg_df, name='InputxGradient',
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)

    s = Saliency(wrapped_model)
    s_attr_valid = s.attribute(valid_data, additional_forward_args=t)
    s_df = attribution2df(s_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, s_df, name='Saliency',
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)

    lime = Lime(wrapped_model,
                interpretable_model=SkLearnLinearRegression())
    lime_attr_valid = lime.attribute(valid_data,
                                     n_samples=100,
                                     additional_forward_args=t)
    lime_df = attribution2df(lime_attr_valid, feature_names, valid_data)
    write_attribution_csv(trainer, lime_df, name='Lime',
                          experiment_name=experiment_name,
                          n=n,
                          seed=seed)

if __name__ == '__main__':
    test = False
    if test:
        feature_attribution(experiment_name='testssss_p0-0.91',
                            n=0,
                            seed=0)

    # Attribution loop
    seed = 0
    for i in range(50):
        seed = seed + 1
        for n in range(6):
            feature_attribution(experiment_name='SingleLayerDSwithCoxWeights_p0-0.0001',
                                seed=seed,
                                n=n)