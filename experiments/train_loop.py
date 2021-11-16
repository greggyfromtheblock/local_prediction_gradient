print('beep beep bloo')

import sys
import os
import hydra
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from omegaconf import DictConfig, OmegaConf

from riskiano.source.callbacks.survival import *
from riskiano.source.callbacks.attribution import *
from riskiano.source.datamodules.simulations import *
from riskiano.source.utils.general import *
from riskiano.source.modules.general import *
from riskiano.source.tasks.survival import *
import torchmetrics
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)
import pandas as pd
pd.options.mode.use_inf_as_na = True
# Specify config path
config_path = "/home/ruyogagp/code/riskiano/riskiano/source/config"

def train(FLAGS, experiment_name='p4-0.9', seed=100):
    pl.seed_everything(seed)

    print(OmegaConf.to_yaml(FLAGS))
    print(FLAGS)

    FLAGS.experiment.experiment_kwargs.experiment_name = experiment_name
    FLAGS.experiment.experiment_kwargs.seed = seed

    csv_name = experiment_name.split('_')[-1]
    FLAGS.experiment.datamodule_kwargs.csv_id = f'{experiment_name}'

    Task = eval(FLAGS.experiment.task)
    Module = StandardMLP
    DataModule = eval(FLAGS.experiment.datamodule)
    print(FLAGS.experiment.datamodule_kwargs.dirpath)

    ### initialize datamodule
    datamodule = DataModule(**FLAGS.experiment.datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup("fit")

    module = MLP(input_dim=len(datamodule.features),
                 output_dim=1,
                 hidden_dims=[256, 256, 256, 256, 256],
                 norm_layer=[0],
                 activation_fn=nn.SiLU,
                 final_activation = nn.SiLU,
                 dropout=0)

    # initialize Task
    task = Task(network=module,
                optimizer=torch.optim.Adam,
                optimizer_kwargs=dict(weight_decay=0.0005),
                schedule=torch.optim.lr_scheduler.MultiStepLR,
                schedule_kwargs=dict(milestones=[10, 15, 20], gamma=0.5),
                **FLAGS.experiment.task_kwargs)

    # initialize trainer
    callbacks = get_default_callbacks(monitor=FLAGS.experiment.monitor) + [WriteCheckpointLogs()]

    trainer = pl.Trainer(**FLAGS.trainer,
                         callbacks=callbacks,
                         logger=set_up_neptune(FLAGS))

    FLAGS["parameters/callbacks"] = [c.__class__.__name__ for c in callbacks]
    trainer.logger.run["FLAGS"] = FLAGS

    if FLAGS.trainer.auto_lr_find:
        trainer.tune(model=task, datamodule=datamodule)

    # run
    trainer.fit(task, datamodule)
    trainer.logger.run.stop()

    print("DONE.")


# This thing is to train with loopz
@hydra.main(config_path, config_name="interpretability")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.config_path = config_path
    for i in range(10):
        seed = FLAGS.start_seed + i
        train(FLAGS, experiment_name= 'simpsons_linear', seed=seed)

if __name__ == '__main__':
    main()