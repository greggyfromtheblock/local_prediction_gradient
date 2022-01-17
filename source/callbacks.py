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
        x_loader = task.ext_dataloader(trainer.datamodule.attribute_x,
                                           batch_size=len(trainer.datamodule.attribute_x),
                                           num_workers=8,
                                           shuffle=False,
                                           drop_last=False)
        y_loader = task.ext_dataloader(trainer.datamodule.attribute_y,
                                           batch_size=len(trainer.datamodule.attribute_y),
                                           num_workers=8,
                                           shuffle=False,
                                           drop_last=False)

        # Unpack batch
        x_data, *_ = task.unpack_batch(next(iter(x_loader)))
        y_data, *_ = task.unpack_batch(next(iter(y_loader)))
        t = torch.tensor([trainer.datamodule.eval_timepoint])

        if self.baseline_method == 'zeros':
            baseline_x = torch.zeros(x_data.shape[0], x_data.shape[1])
            baseline_y = torch.zeros(y_data.shape[0], y_data.shape[1])

        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0.001, 0.0001, inputs.shape)).float()
            return noise, inputs - noise

        if self.project == 'correlation':
            outpath = "/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/correlation_case"
        elif self.project == 'simpsons':
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/simpsons_case"
        elif self.project == 'resample_multiplicities':
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/resample_multiplicities"


        if not os.path.exists(f'{outpath}/{self.experiment_name}'):
            os.mkdir(f'{outpath}/{self.experiment_name}')

        # KernelExplainer
        wrapped_task = ForwardWrapper(task, method='KernelExplainer')
        baseline = torch.zeros(1, x_data.shape[1])
        explainer = shap_fork.KernelExplainer(wrapped_task, baseline)
        kernel_explainer_x = explainer.shap_values(x_data.numpy())
        kernel_explainer_x = kernel_explainer_x[0] if isinstance(kernel_explainer_x, list) else kernel_explainer_attr_x
        kernel_explainer_y = explainer.shap_values(y_data.numpy())
        kernel_explainer_y = kernel_explainer_y[0] if isinstance(kernel_explainer_y, list) else kernel_explainer_attr_y
        expected_value_kernel = explainer.expected_value

        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_x_{self.seed}.csv',
                   kernel_explainer_x, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_y_{self.seed}.csv',
                   kernel_explainer_y, delimiter=',', fmt="%s")
        wandb.config.update(dict(KernelExplainer_x_path=f'{outpath}/{self.experiment_name}/KernelExplainer_x_{self.seed}.csv'))
        wandb.config.update(dict(KernelExplainer_y_path=f'{outpath}/{self.experiment_name}/KernelExplainer_y_{self.seed}.csv'))

        # DeepExplainer
        wrapped_task = ForwardWrapper(task, method='DeepExplainer')
        baseline = torch.zeros(1, x_data.shape[1])
        explainer = shap_fork.DeepExplainer(wrapped_task, baseline)
        deep_explainer_x = explainer.shap_values(x_data)
        deep_explainer_y = explainer.shap_values(y_data)

        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_x_{self.seed}.csv',
                   deep_explainer_x, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_y_{self.seed}.csv',
                   deep_explainer_y, delimiter=',', fmt="%s")
        wandb.config.update(dict(DeepExplainer_x_path=f'{outpath}/{self.experiment_name}/DeepExplainer_x_{self.seed}.csv'))
        wandb.config.update(dict(DeepExplainer_y_path=f'{outpath}/{self.experiment_name}/DeepExplainer_y_{self.seed}.csv'))

        # Feature Ablation
        wrapped_model = ForwardWrapper(task, method='FeatureAblation')
        fa = FeatureAblation(wrapped_model)
        fa_attr_x = fa.attribute(x_data, additional_forward_args=t, baselines=baseline)
        fa_attr_y = fa.attribute(y_data, additional_forward_args=t, baselines=baseline)
        np.savetxt(f'{outpath}/{self.experiment_name}/FeatureAblation_x_{self.seed}.csv', fa_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/FeatureAblation_y_{self.seed}.csv', fa_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(FeatureAblation_x_path=f'{outpath}/{self.experiment_name}/FeatureAblation_x_{self.seed}.csv'))
        wandb.config.update(dict(FeatureAblation_y_path=f'{outpath}/{self.experiment_name}/FeatureAblation_y_{self.seed}.csv'))

        # Feature Permutation
        fp = FeaturePermutation(wrapped_model)
        fp_attr_x = fp.attribute(x_data, additional_forward_args=t)
        fp_attr_y = fp.attribute(y_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_x_{self.seed}.csv', fp_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_y_{self.seed}.csv', fp_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(FeaturePermutation_x_path=f'{outpath}/{self.experiment_name}/FeaturePermutation_x_{self.seed}.csv'))
        wandb.config.update(dict(FeaturePermutation_y_path=f'{outpath}/{self.experiment_name}/FeaturePermutation_y_{self.seed}.csv'))

        # Integrated Gradients
        ig = IntegratedGradients(wrapped_model)
        ig_attr_x, ig_delta = ig.attribute(x_data,
                                               baselines=baseline,
                                               n_steps=100,
                                               additional_forward_args=t,
                                               internal_batch_size=task.batch_size,
                                               return_convergence_delta=True)
        ig_attr_y, ig_delta = ig.attribute(y_data,
                                         baselines=baseline,
                                         n_steps=100,
                                         additional_forward_args=t,
                                         internal_batch_size=task.batch_size,
                                         return_convergence_delta=True)
        np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_x_{self.seed}.csv', ig_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_y_{self.seed}.csv', ig_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(IntegratedGradients_x_path=f'{outpath}/{self.experiment_name}/IntegratedGradients_x_{self.seed}.csv'))
        wandb.config.update(dict(IntegratedGradients_y_path=f'{outpath}/{self.experiment_name}/IntegratedGradients_y_{self.seed}.csv'))

        # Shapley Value Sampling
        svs = ShapleyValueSampling(wrapped_model)
        svs_attr_x = svs.attribute(x_data, additional_forward_args=t, baselines=baseline)
        svs_attr_y = svs.attribute(y_data, additional_forward_args=t, baselines=baseline)
        np.savetxt(f'{outpath}/{self.experiment_name}/ShapleyValueSampling_x_{self.seed}.csv', svs_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/ShapleyValueSampling_y_{self.seed}.csv', svs_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(ShapleyValueSampling_x_path=f'{outpath}/{self.experiment_name}/ShapleyValueSampling_x_{self.seed}.csv'))
        wandb.config.update(dict(ShapleyValueSampling_y_path=f'{outpath}/{self.experiment_name}/ShapleyValueSampling_y_{self.seed}.csv'))

        # Input x Gradient
        ixg = InputXGradient(wrapped_model)
        ixg_attr_x = ixg.attribute(x_data, additional_forward_args=t)
        ixg_attr_y = ixg.attribute(y_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/InputxGradient_x_{self.seed}.csv', ixg_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/InputxGradient_y_{self.seed}.csv', ixg_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(InputXGradient_x_path=f'{outpath}/{self.experiment_name}/InputxGradient_x_{self.seed}.csv'))
        wandb.config.update(dict(InputXGradient_y_path=f'{outpath}/{self.experiment_name}/InputxGradient_y_{self.seed}.csv'))

        # Saliency
        s = Saliency(wrapped_model)
        s_attr_x = s.attribute(x_data, additional_forward_args=t)
        s_attr_y = s.attribute(y_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Saliency_x_{self.seed}.csv', s_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/Saliency_y_{self.seed}.csv', s_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(Saliency_x_path=f'{outpath}/{self.experiment_name}/Saliency_x_{self.seed}.csv'))
        wandb.config.update(dict(Saliency_y_path=f'{outpath}/{self.experiment_name}/Saliency_y_{self.seed}.csv'))

        # Lime
        lime = Lime(wrapped_model,
                    interpretable_model=SkLearnLinearRegression())
        lime_attr_x = lime.attribute(x_data,
                                         n_samples=20,
                                         additional_forward_args=t)
        lime_attr_y = lime.attribute(y_data,
                                   n_samples=20,
                                   additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Lime_x_{self.seed}.csv', lime_attr_x.detach().numpy(), delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/Lime_y_{self.seed}.csv', lime_attr_y.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(Lime_x_path=f'{outpath}/{self.experiment_name}/Lime_x_{self.seed}.csv'))
        wandb.config.update(dict(Lime_y_path=f'{outpath}/{self.experiment_name}/Lime_y_{self.seed}.csv'))

