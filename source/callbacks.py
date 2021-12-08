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

class CSVDataset(Dataset):
    def __init__(self, name, project):
        """
        Dataset to read the attribution resampling csvs for the attribution experiments
        :param name: csv id -> must end with _attribute
        :param project: selection of correlation or simpson
        """
        self.path = '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/correlation' if project == 'correlation' else '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/simpson'
        self.name = name
        self.csv_path = f'{self.path}/{self.name}_attribute.csv'
        self._read()

        # cut columns
        self.durations = self.data[['time']]
        self.durations.columns = ['duration']
        self.durations['duration'] = self.durations['duration'] + 1
        self.events = self.data[['event']]
        self.events.columns = ['event']
        self.data.drop(['time', 'event'], axis=1, inplace=True)

    def _read(self):
        self.data = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        data = torch.Tensor(self.data.values[idx, :])
        duration = torch.Tensor(self,durations.values[idx])
        event = torch.Tensor(self.events.values[idx])
        return data, (duration, event)

    def __len__(self):
        return len(self.data)


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
        eval_batch_size = len(trainer.datamodule.valid_ds)

        csv_dataset = CSVDataset(self.experiment_name, project=self.project)
        data_loader = DataLoader(csv_dataset, batch_size=len(csv_dataset))

        # Unpack batch
        valid_data, *_ = task.unpack_batch(next(iter(data_loader)))
        t = torch.tensor([trainer.datamodule.eval_timepoint])

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

        if self.project == 'correlation':
            outpath = "/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/correlation_case"
        else:
            outpath ="/data/analysis/ag-reils/ag-reils-shared/cardioRS/results/interpretability/simpsons_case"

        if not os.path.exists(f'{outpath}/{self.experiment_name}'):
            os.mkdir(f'{outpath}/{self.experiment_name}')

        # KernelExplainer
        wrapped_task = ForwardWrapper(task, method='KernelExplainer')
        baseline = torch.zeros(1, valid_data.shape[1])
        explainer = shap_fork.KernelExplainer(wrapped_task, baseline)
        kernel_explainer_attr = explainer.shap_values(valid_data.numpy())
        kernel_explainer_attr = kernel_explainer_attr[0] if isinstance(kernel_explainer_attr, list) else kernel_explainer_attr
        expected_value_kernel = explainer.expected_value

        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_{self.seed}.csv',
                   kernel_explainer_attr, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/KernelExplainer_{self.seed}_expected-value.csv',
                   expected_value_kernel, delimiter=',', fmt="%s")
        wandb.config.update(dict(KernelExplainer_path=f'{outpath}/{self.experiment_name}/KernelExplainer_{self.seed}.csv'))

        # DeepExplainer
        wrapped_task = ForwardWrapper(task, method='DeepExplainer')
        baseline = torch.zeros(1, valid_data.shape[1])
        explainer = shap_fork.DeepExplainer(wrapped_task, baseline)
        deep_explainer_attr = explainer.shap_values(valid_data)
        expected_value_deep = explainer.expected_value

        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_{self.seed}.csv',
                   deep_explainer_attr, delimiter=',', fmt="%s")
        np.savetxt(f'{outpath}/{self.experiment_name}/DeepExplainer_{self.seed}_expected-value.csv',
                   expected_value_deep, delimiter=',', fmt="%s")
        wandb.config.update(dict(DeepExplainer_path=f'{outpath}/{self.experiment_name}/DeepExplainer_{self.seed}.csv'))

        # Feature Ablation
        wrapped_model = ForwardWrapper(task, method='FeatureAblation')
        fa = FeatureAblation(wrapped_model)
        fa_attr = fa.attribute(valid_data, additional_forward_args=t, baselines=baseline)
        np.savetxt(f'{outpath}/{self.experiment_name}/FeatureAblation_{self.seed}.csv', fa_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(FeatureAblation_path=f'{outpath}/{self.experiment_name}/FeatureAblation_{self.seed}.csv'))

        # Feature Permutation
        fp = FeaturePermutation(wrapped_model)
        fp_attr = fp.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/FeaturePermutation_{self.seed}.csv', fp_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(FeaturePermutation_path=f'{outpath}/{self.experiment_name}/FeaturePermutation_{self.seed}.csv'))

        # Integrated Gradients
        ig = IntegratedGradients(wrapped_model)
        ig_attr, ig_delta = ig.attribute(valid_data,
                                               baselines=baseline,
                                               n_steps=100,
                                               additional_forward_args=t,
                                               internal_batch_size=task.batch_size,
                                               return_convergence_delta=True)
        np.savetxt(f'{outpath}/{self.experiment_name}/IntegratedGradients_{self.seed}.csv', ig_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(IntegratedGradients_path=f'{outpath}/{self.experiment_name}/IntegratedGradients_{self.seed}.csv'))

        # Shapley Value Sampling
        svs = ShapleyValueSampling(wrapped_model)
        svs_attr = svs.attribute(valid_data, additional_forward_args=t, baselines=baseline)
        np.savetxt(f'{outpath}/{self.experiment_name}/ShapleyValueSampling_{self.seed}.csv', svs_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(ShapleyValueSampling_path=f'{outpath}/{self.experiment_name}/ShapleyValueSampling_{self.seed}.csv'))

        # Input x Gradient
        ixg = InputXGradient(wrapped_model)
        ixg_attr = ixg.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/InputxGradient_{self.seed}.csv', ixg_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(InputXGradient_path=f'{outpath}/{self.experiment_name}/InputxGradient_{self.seed}.csv'))

        # Saliency
        s = Saliency(wrapped_model)
        s_attr = s.attribute(valid_data, additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Saliency_{self.seed}.csv', s_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(Saliency_path=f'{outpath}/{self.experiment_name}/Saliency_{self.seed}.csv'))

        # Lime
        lime = Lime(wrapped_model,
                    interpretable_model=SkLearnLinearRegression())
        lime_attr = lime.attribute(valid_data,
                                         n_samples=20,
                                         additional_forward_args=t)
        np.savetxt(f'{outpath}/{self.experiment_name}/Lime_{self.seed}.csv', lime_attr.detach().numpy(), delimiter=',', fmt="%s")
        wandb.config.update(dict(Lime_path=f'{outpath}/{self.experiment_name}/Lime_{self.seed}.csv'))

