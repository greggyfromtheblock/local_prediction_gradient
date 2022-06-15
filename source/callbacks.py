import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from pytorch_lightning.callbacks.base import Callback
from captum.attr import *
from captum.metrics import *
from captum._utils.models.linear_model import SkLearnLinearRegression

from source.wrappers import PredictRiskWrapper, ForwardWrapper
from source.tasks import DeepSurv
import shap_fork as shap_fork

import wandb


class FeatureAttribution(Callback):
    def __init__(self,
                 project,
                 baseline_method = 'zeros',
                 experiment_name = '',
                 seed = 0,
                 **kwargs):
        super().__init__()
        """
        Runs attribution methods at the end of fit
        :param project: project for the attribution [correlation, simpsons]
        :param experiment_name: folder name to save the attributions at + name to id the attribution
        :param seed: seed number of the training (for simulation experiment)
        :param baseline_method: baseline to choose from when computing attributions ['zeros', 'ones', 'average']
        """
        self.project = project
        self.baseline_method = baseline_method
        self.experiment_name = experiment_name
        self.seed = seed


    def on_fit_end(self, trainer, pl_module, t=torch.tensor([11.0]), eval_batch_size=None):
        path = trainer.checkpoint_callback.best_model_path

        if isinstance(trainer.logger, list):
            logger = trainer.logger[0]
        else:
            logger = trainer.logger

        task = pl_module.load_from_checkpoint(path, logger=pl_module.logger, strict=True)
        task.eval()

        feature_names = trainer.datamodule.features
        n_features = len(feature_names)

        if n_features >=2:
            f0_loader = task.ext_dataloader(trainer.datamodule.attribute_feature0,
                                            batch_size=len(trainer.datamodule.attribute_feature0),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            f1_loader = task.ext_dataloader(trainer.datamodule.attribute_feature1,
                                            batch_size=len(trainer.datamodule.attribute_feature1),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            attribution_loaders = [f0_loader, f1_loader]
        if n_features >= 4:
            f2_loader = task.ext_dataloader(trainer.datamodule.attribute_feature2,
                                            batch_size=len(trainer.datamodule.attribute_feature2),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            f3_loader = task.ext_dataloader(trainer.datamodule.attribute_feature3,
                                            batch_size=len(trainer.datamodule.attribute_feature3),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            attribution_loaders = [f0_loader, f1_loader, f2_loader, f3_loader]
        if n_features >=5:
            f4_loader = task.ext_dataloader(trainer.datamodule.attribute_feature4,
                                            batch_size=len(trainer.datamodule.attribute_feature4),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            attribution_loaders = [f0_loader, f1_loader, f2_loader, f3_loader, f4_loader]
        if n_features >=6:
            f5_loader = task.ext_dataloader(trainer.datamodule.attribute_feature5,
                                            batch_size=len(trainer.datamodule.attribute_feature5),
                                            num_workers=8,
                                            shuffle=False,
                                            drop_last=False)
            attribution_loaders.append(f5_loader)
            attribution_loaders = [f0_loader, f1_loader, f2_loader, f3_loader, f4_loader, f5_loader]


        # Unpack batch
        unpacked_data = [task.unpack_batch(next(iter(loader)))[0] for loader in attribution_loaders]
        t = torch.tensor([trainer.datamodule.eval_timepoint])

        if self.baseline_method == 'zeros':
            baselines = [torch.zeros(data.shape[0], data.shape[1]) for data in unpacked_data]
        elif self.baseline_method == 'ones':
            baselines = [torch.ones(data.shape[0], data.shape[1]) for data in unpacked_data]

        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0.01, 0.01, inputs.shape)).float()
            return noise, inputs - noise

        if self.project == 'correlation':
            outpath = "/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/correlation_case"
        elif self.project == 'simpsons':
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/simpsons_case"
        else:
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities"


        if not os.path.exists(f'{outpath}/{self.experiment_name}'):
            os.mkdir(f'{outpath}/{self.experiment_name}')

        # KernelExplainer
        wrapped_task = ForwardWrapper(task, method='KernelExplainer')
        baseline = torch.zeros(1, unpacked_data[0].shape[1]) if self.baseline_method == 'zeros' else torch.ones(1, unpacked_data[0].shape[1])
        explainer = shap_fork.KernelExplainer(wrapped_task, baseline)
        ke_explanations = []
        for idx, data in enumerate(unpacked_data):
            ke_explanation = explainer.shap_values(data.numpy())
            ke_explanation = ke_explanation[0] if isinstance(ke_explanation, list) else ke_explanation
            np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_feature{idx}_{self.seed}.csv',
                       ke_explanation, delimiter=',', fmt="%s")
            dicc = {f'KernelExplainer_feature{idx}_path': f'{outpath}/{self.experiment_name}/KernelExplainer_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)

        # DeepExplainer
        wrapped_task = ForwardWrapper(task, method='DeepExplainer')
        baseline = torch.zeros(1, unpacked_data[0].shape[1]) if self.baseline_method == 'zeros' else torch.ones(1, unpacked_data[0].shape[1])
        explainer = shap_fork.DeepExplainer(wrapped_task, baseline)
        for idx, data in enumerate(unpacked_data):
            de_explanation = explainer.shap_values(data)
            np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_feature{idx}_{self.seed}.csv',
                       de_explanation, delimiter=',', fmt="%s")
            dicc = {f'DeepExplainer_feature{idx}_path': f'{outpath}/{self.experiment_name}/DeepExplainer_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)

        # Feature Permutation
        wrapped_model = ForwardWrapper(task, method='FeaturePermutation')
        fp = FeaturePermutation(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = fp.attribute(data, additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'FeaturePermutation_feature{idx}_path': f'{outpath}/{self.experiment_name}/FeaturePermutation_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        wrapped_model = ForwardWrapper(task, method='IntegratedGradients')
        ig = IntegratedGradients(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation, convergence_delta = ig.attribute(data,
                                       baselines=baseline,
                                       n_steps=100,
                                       additional_forward_args=t,
                                       internal_batch_size=task.batch_size,
                                       return_convergence_delta=True)
            np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'IntegratedGradients_feature{idx}_path': f'{outpath}/{self.experiment_name}/IntegratedGradients_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        wrapped_model = ForwardWrapper(task, method='InputXGradient')
        ixg = InputXGradient(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = ixg.attribute(data, additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/InputxGradient_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'InputxGradient_feature{idx}_path': f'{outpath}/{self.experiment_name}/InputxGradient_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        wrapped_model = ForwardWrapper(task, method='Lime')
        lime = Lime(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            explanation = lime.attribute(data,
                                          n_samples=20,
                                          additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/Lime_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'Lime_feature{idx}_path': f'{outpath}/{self.experiment_name}/Lime_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)

class FeatureAttribution2(Callback):
    def __init__(self,
                 project,
                 baseline_method = 'zeros',
                 experiment_name = '',
                 seed = 0,
                 **kwargs):
        super().__init__()
        """
        Runs attribution methods at the end of fit
        :param project: project for the attribution [correlation, simpsons]
        :param experiment_name: folder name to save the attributions at + name to id the attribution
        :param seed: seed number of the training (for simulation experiment)
        :param baseline_method: baseline to choose from when computing attributions ['zeros', 'ones', 'average']
        """
        self.project = project
        self.baseline_method = baseline_method
        self.experiment_name = experiment_name
        self.seed = seed


    def on_fit_end(self, trainer, pl_module, t=torch.tensor([11.0]), eval_batch_size=None):
        path = trainer.checkpoint_callback.best_model_path

        if isinstance(trainer.logger, list):
            logger = trainer.logger[0]
        else:
            logger = trainer.logger

        task = pl_module.load_from_checkpoint(path, logger=pl_module.logger, strict=True)
        task.eval()

        feature_names = trainer.datamodule.features

        dataloaders = []
        f0_loader = task.ext_dataloader(trainer.datamodule.attribute_feature0,
                                        batch_size=len(trainer.datamodule.attribute_feature0),
                                        num_workers=8,
                                        shuffle=False,
                                        drop_last=False)
        f1_loader = task.ext_dataloader(trainer.datamodule.attribute_feature1,
                                        batch_size=len(trainer.datamodule.attribute_feature1),
                                        num_workers=8,
                                        shuffle=False,
                                        drop_last=False)


        # Unpack batch
        attribution_loaders = [f0_loader, f1_loader]
        unpacked_data = [task.unpack_batch(next(iter(loader)))[0] for loader in attribution_loaders]
        t = torch.tensor([trainer.datamodule.eval_timepoint])

        if self.baseline_method == 'zeros':
            baselines = [torch.zeros(data.shape[0], data.shape[1]) for data in unpacked_data]

        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0.001, 0.0001, inputs.shape)).float()
            return noise, inputs - noise

        if self.project == 'correlation':
            outpath = "/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/correlation_case"
        elif self.project == 'simpsons':
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/simpsons_case"
        else:
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities"


        if not os.path.exists(f'{outpath}/{self.experiment_name}'):
            os.mkdir(f'{outpath}/{self.experiment_name}')

        # KernelExplainer
        wrapped_task = ForwardWrapper(task, method='KernelExplainer')
        baseline = torch.zeros(1, unpacked_data[0].shape[1])
        explainer = shap_fork.KernelExplainer(wrapped_task, baseline)
        ke_explanations = []
        for idx, data in enumerate(unpacked_data):
            ke_explanation = explainer.shap_values(data.numpy())
            ke_explanation = ke_explanation[0] if isinstance(ke_explanation, list) else ke_explanation
            np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_feature{idx}_{self.seed}.csv',
                       ke_explanation, delimiter=',', fmt="%s")
            dicc = {f'KernelExplainer_feature{idx}_path': f'{outpath}/{self.experiment_name}/KernelExplainer_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)

        # DeepExplainer
        wrapped_task = ForwardWrapper(task, method='DeepExplainer')
        baseline = torch.zeros(1, unpacked_data[0].shape[1])
        explainer = shap_fork.DeepExplainer(wrapped_task, baseline)
        for idx, data in enumerate(unpacked_data):
            de_explanation = explainer.shap_values(data)
            np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_feature{idx}_{self.seed}.csv',
                       de_explanation, delimiter=',', fmt="%s")
            dicc = {f'DeepExplainer_feature{idx}_path': f'{outpath}/{self.experiment_name}/DeepExplainer_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Ablation
        wrapped_model = ForwardWrapper(task, method='FeatureAblation')
        fa = FeatureAblation(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = fa.attribute(data, additional_forward_args=t, baselines=baseline)
            np.savetxt(f'{outpath}/{self.experiment_name}/FeatureAblation_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'FeatureAblation_feature{idx}_path': f'{outpath}/{self.experiment_name}/FeatureAblation_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)



        # Feature Permutation
        fp = FeaturePermutation(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = fp.attribute(data, additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'FeaturePermutation_feature{idx}_path': f'{outpath}/{self.experiment_name}/FeaturePermutation_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        ig = IntegratedGradients(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation, convergence_delta = ig.attribute(data,
                                                          baselines=baseline,
                                                          n_steps=100,
                                                          additional_forward_args=t,
                                                          internal_batch_size=task.batch_size,
                                                          return_convergence_delta=True)
            np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'IntegratedGradients_feature{idx}_path': f'{outpath}/{self.experiment_name}/IntegratedGradients_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        svs = ShapleyValueSampling(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = svs.attribute(data, additional_forward_args=t, baselines=baseline)
            np.savetxt(f'{outpath}/{self.experiment_name}/ShapleyValueSampling_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'ShapleyValueSampling_feature{idx}_path': f'{outpath}/{self.experiment_name}/ShapleyValueSampling_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        ixg = InputXGradient(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            baseline = baselines[idx]
            explanation = ixg.attribute(data, additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/InputXGradient_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'InputXGradient_feature{idx}_path': f'{outpath}/{self.experiment_name}/InputXGradient_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)


        # Feature Permutation
        lime = Lime(wrapped_model)
        for idx, data in enumerate(unpacked_data):
            explanation = lime.attribute(data,
                                         n_samples=20,
                                         additional_forward_args=t)
            np.savetxt(f'{outpath}/{self.experiment_name}/Lime_feature{idx}_{self.seed}.csv',
                       explanation.detach().numpy(), delimiter=',', fmt="%s")
            dicc = {f'Lime_feature{idx}_path': f'{outpath}/{self.experiment_name}/Lime_feature{idx}_{self.seed}.csv'}
            wandb.config.update(dicc)