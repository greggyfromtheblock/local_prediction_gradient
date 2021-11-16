import os
import torch
import shap_fork as shapley
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from pytorch_lightning.callbacks.base import Callback
from captum.attr import *
from captum.metrics import *
from captum._utils.models.linear_model import SkLearnLinearRegression

from medical_interpretability.source.wrappers import PredictRiskWrapper, ForwardWrapper

from riskiano.source.utils.general import attribution2df
from riskiano.source.evaluation.tabular import *
from riskiano.source.tasks.survival import DeepSurvivalMachine, DeepSurv



class FeatureAttribution(Callback):
    def __init__(self,
                 baseline_method='zeros',
                 compute_sens=False,
                 compute_inf=False,
                 experiment_name='',
                 n=0,
                 seed=0,
                 **kwargs):
        super().__init__()
        """
        Runs attribution methods at the end of fit
        :param experiment_name: foler name to save the attributions at + name to id the attribution
        :param n: number of extra correlated variable on the training (for simulation experiment)
        :param seed: seed number of the training (for simulation experiment)
        :param baseline_method: baseline to choose from when computing attributions ['zeros', 'ones', 'average']
        :pram compute_sens: wether to compute sensitivity max (this takes gazillion years to run)
        """
        self.baseline_method = baseline_method
        self.compute_sens = compute_sens
        self.experiment_name = experiment_name
        self.n = n
        self.seed = seed
        self.compute_inf = compute_inf



    def on_fit_end(self, trainer, pl_module, device='cuda:0', t=torch.tensor([11.0]), eval_batch_size=None):
        path = trainer.checkpoint_callback.best_model_path

        if isinstance(trainer.logger, list):
            logger = trainer.logger[0]
        else:
            logger = trainer.logger

        task = pl_module.load_from_checkpoint(path, logger=pl_module.logger, strict=True)
        task.eval()

        feature_names = trainer.datamodule.features
        eval_batch_size = len(trainer.datamodule.valid_ds)
        train_loader = task.ext_dataloader(trainer.datamodule.train_ds,
                                             batch_size=eval_batch_size,
                                             num_workers=8,
                                             shuffle=True,
                                             drop_last=False)
        valid_loader = task.ext_dataloader(trainer.datamodule.valid_ds,
                                             batch_size=eval_batch_size,
                                             num_workers=8,
                                             shuffle=True,
                                             drop_last=False)

        # Get a batch from dataloader
        valid_batch = next(iter(valid_loader))
        train_batch = next(iter(train_loader))

        # Unpack batch
        valid_data, *_ = task.unpack_batch(valid_batch)
        train_data, *_ = task.unpack_batch(train_batch)

        # Wrapper for the model that returns F(t) from the forward pass
        if isinstance(task, DeepSurvivalMachine):
            def wrapped_model(inp, t):
                return task(inp, t)[1].squeeze(0)
        elif isinstance(task, DeepSurv):
            def wrapped_model(inp, t):
                return task.predict_risk(inp, t)
        else:
            raise NotImplementedError('FeatureAttribution currently only supports DeepSurvivalMachine and DeepSurv')

        t = torch.tensor([trainer.datamodule.eval_timepoint])
        t_repeat = torch.tensor([trainer.datamodule.eval_timepoint]).repeat([task.batch_size, 1])

        if self.baseline_method == 'zeros':
            baseline = torch.zeros(valid_data.shape[0], valid_data.shape[1])
        elif self.baseline_method == 'ones':
            baseline = torch.ones(valid_data.shape[0], valid_data.shape[1])
        elif self.baseline_method == 'average':
            average = torch.mean(valid_data)
            baseline = torch.empty(valid_data.shape[0], valid_data.shape[1]).fill_(average)

        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0.001, 0.0001, inputs.shape)).float()
            return noise, inputs - noise

        outpath = "/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/cor"

        if not os.path.exists(f'{outpath}/{self.experiment_name}'):
            os.mkdir(f'{outpath}/{self.experiment_name}')

        # KernelExplainer
        wrapped_task = ForwardShapWrapper(task, method='KernelExplainer')
        baseline = torch.zeros(1, valid_data.shape[1])
        explainer = shapley.KernelExplainer(wrapped_task, baseline)
        kernel_explainer_attr = explainer.shap_values(valid_data.numpy())
        kernel_explainer_attr = kernel_explainer_attr[0] if isinstance(kernel_explainer_attr, list) else kernel_explainer_attr
        expected_value_kernel = explainer.expected_value
        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_{self.seed}_n{self.n}.csv',
                   kernel_explainer_attr, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_{self.seed}_expected-value_n{self.n}.csv',
                   expected_value_kernel, delimiter=',', fmt="%s")
        kernel_df = attribution2df(kernel_explainer_attr, feature_names, valid_data)
        log_global_importance(trainer, kernel_df, name="KernelSHAP",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)
        print(f'saved {outpath}/{self.experiment_name}/KernelExplainer_{self.seed}_n{self.n}.csv')

        # DeepExplainer
        wrapped_task = ForwardShapWrapper(task, method='DeepExplainer')
        baseline = torch.zeros(1, valid_data.shape[1])
        explainer = shapley.DeepExplainer(wrapped_task, baseline)
        deep_explainer_attr = explainer.shap_values(valid_data)
        expected_value_deep = explainer.expected_value
        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_{self.seed}_n{self.n}.csv',
                   deep_explainer_attr, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_{self.seed}_expected-value.csv',
                   expected_value_deep, delimiter=',', fmt="%s")
        print(f'saved {outpath}/{self.experiment_name}/DeepExplainer_{self.seed}_n{self.n}.csv')
        deep_explainer_df = attribution2df(deep_explainer_attr, feature_names, valid_data)
        log_global_importance(trainer, deep_explainer_df, name="DeepSHAP",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Feature Ablation
        wrapped_model = ForwardShapWrapper(task, method='FeatureAblation')
        fa = FeatureAblation(wrapped_model)
        fa_attr = fa.attribute(valid_data, additional_forward_args=t, baselines=baseline)
        print(f'saved {outpath}/{self.experiment_name}/FeatureAblation_{self.seed}.csv')
        fa_df = attribution2df(fa_attr, feature_names, valid_data)
        log_global_importance(trainer, fa_df, name="FeatureAblation",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Feature Permutation
        fp = FeaturePermutation(wrapped_model)
        fp_attr = fp.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_{self.seed}_n{self.n}.csv', fp_attr.detach().numpy(), delimiter=',', fmt="%s")
        fp_df = attribution2df(fp_attr, feature_names, valid_data)
        log_global_importancse(trainer, fp_df, name="FeaturePermutation",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Integrated Gradients
        ig = IntegratedGradients(wrapped_model)
        ig_attr, ig_delta = ig.attribute(valid_data,
                                               baselines=baseline,
                                               n_steps=100,
                                               additional_forward_args=t,
                                               internal_batch_size=task.batch_size,
                                               return_convergence_delta=True)
        np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_{self.seed}_n{self.n}.csv', ig_attr.detach().numpy(), delimiter=',', fmt="%s")
        ig_df = attribution2df(ig_attr, feature_names, valid_data)
        log_global_importance(trainer, ig_df, name="IntegratedGradients",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Shapley Value Sampling
        shap = ShapleyValueSampling(wrapped_model)
        shap_attr = shap.attribute(valid_data, additional_forward_args=t, baselines=baseline)
        np.savetxt(f'{outpath}/{self.experiment_name}/ShapleyValueSampling_{self.seed}_n{self.n}.csv', shap_attr.detach().numpy(), delimiter=',', fmt="%s")
        shap_df = attribution2df(shap_attr, feature_names, valid_data)
        log_global_importance(trainer, shap_df, name="ShapleyValueSampling",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Input x Gradient
        ixg = InputXGradient(wrapped_model)
        ixg_attr = ixg.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/InputxGradient_{self.seed}_n{self.n}.csv', ixg_attr.detach().numpy(), delimiter=',', fmt="%s")
        ixg_df = attribution2df(ixg_attr, feature_names, valid_data)
        log_global_importance(trainer, ixg_df, name="InputXGradient",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Saliency
        s = Saliency(wrapped_model)
        s_attr = s.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Saliency_{self.seed}_n{self.n}.csv', s_attr.detach().numpy(), delimiter=',', fmt="%s")
        s_df = attribution2df(s_attr, feature_names, valid_data)
        log_global_importance(trainer, s_df, name="Saliency",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)

        # Lime
        lime = Lime(wrapped_model,
                    interpretable_model=SkLearnLinearRegression())
        lime_attr = lime.attribute(valid_data,
                                         n_samples=20,
                                         additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Lime{self.seed}_n{self.n}.csv', lime_attr.detach().numpy(), delimiter=',', fmt="%s")
        lime_df = attribution2df(lime_attr, feature_names, valid_data)
        log_global_importance(trainer, lime_df, name="Lime",
                              experiment_name=self.experiment_name,
                              n=self.n,
                              seed=self.seed)