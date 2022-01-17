print('helloooo')

import os
import hydra
import torch
import torch.nn as nn
from torch.nn import Sigmoid, SELU, ReLU
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from source.datamodule import SyntheticDatamodule
# from riskiano.datamodules.simulations import SyntheticDatamodule
from riskiano.modules.survival import *
from riskiano.modules.general import *
from riskiano.utils.general import *
from riskiano.callbacks.survival import WriteCheckpointLogs, CalculateSurvivalMetrics, EstimateBaselineHazards

from source.tasks import DeepSurv
from source.callbacks import FeatureAttribution
import wandb

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# globals:
import pandas as pd
pd.options.mode.use_inf_as_na = True
import pytorch_lightning as pl


assert os.environ['NEPTUNE_API_TOKEN'], 'No Neptune API Token found. Please do `export NEPTUNE_API_TOKEN=<token>`.'
config_path = "/home/ruyogagp/medical_interpretability/source/config"

def train(FLAGS, seed):
    print('!!!!!!!!!!!', FLAGS.experiment.datamodule_kwargs.csv_id, '!!!!!!!!!!')
    pl.seed_everything(seed)
    Task = eval(FLAGS.experiment.task)
    Module = eval(FLAGS.experiment.module)
    DataModule = eval(FLAGS.experiment.datamodule)

    ### initialize datamodule
    datamodule = DataModule(**FLAGS.experiment.datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup("fit")

    module = Module(input_dim=len(datamodule.features),
                    **FLAGS.experiment.module_kwargs)

    # initialize Task
    task = Task(network=module,
                optimizer_kwargs=dict(weight_decay=0.01),
                **FLAGS.experiment.task_kwargs)

    FLAGS.experiment.datamodule_kwargs.seed = seed
    attribution_callback = FeatureAttribution(project='resample_multiplicities',
                                              baseline_method='zeros',
                                              experiment_name=FLAGS.experiment.datamodule_kwargs.csv_id,
                                              seed=FLAGS.experiment.datamodule_kwargs.seed)


    # initialize trainer
    callbacks = get_default_callbacks(monitor=FLAGS.experiment.monitor) + [WriteCheckpointLogs(), attribution_callback]
    wandb_logger = WandbLogger(project="cardiors/interpretability", tags=['correlation_case', FLAGS.experiment.datamodule_kwargs.csv_id])

    trainer = pl.Trainer(**FLAGS.trainer,
                         callbacks=callbacks,
                         logger=wandb_logger)

    wandb.init(project='test', group='correlation_case', config=FLAGS, tags=['resample_multiplicities', FLAGS.experiment.datamodule_kwargs.csv_id])

    # run
    trainer.fit(task, datamodule)
    trainer.logger.experiment.finish()


@hydra.main(config_path, config_name="interpretability")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.config_path = config_path
    seed = 150
    for i in range(50):
        seed += 1
        train(FLAGS, seed=seed)

if __name__ == '__main__':
    main()