import random
import torch
import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from riskiano.losses.survival import cox_ph_loss
from riskiano.datamodules.datasets import BatchedDS, DeepHitBatchedDS

from sksurv.metrics import concordance_index_ipcw
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

class AbstractSurvivalTask(pl.LightningModule):
    """
    Defines a Task (in the sense of Lightning) to train a CoxPH-Model.
    """

    def __init__(self, network,
                 transforms=None,
                 batch_size=128,
                 num_workers=8,
                 learning_rate=1e-3,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={}, **kwargs):
        """
        Defines a Task (in the sense of Lightning) to train a CoxPH-Model.
        :param network: `nn.Module or pl.LightningModule`,  the network that should be used.
        :param transforms:  `nn.Module or pl.LightningModule`, optional contains the Transforms applied to input.
        :param batch_size:  `int`, batchsize
        :param num_workers: `int`, num_workers for the DataLoaders
        :param optimizer:   `torch.optim`, class, is instantiated w/ the passed optimizer args by trainer.
        :param optimizer_kwargs:    `dict`, optimizer args.
        :param schedule:    `scheudle calss` to use
        :param schedule_kwargs:  `dict`, schedule kwargs, like: {'patience':10, 'threshold':0.0001, 'min_lr':1e-6}
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.net = network
        self.transforms = transforms

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs
        self.evaluation_quantile_bins = eval(evaluation_quantile_bins) if not isinstance(evaluation_quantile_bins, (type(None), list, ListConfig)) else evaluation_quantile_bins
        self.evaluation_time_points = eval(evaluation_time_points) if not isinstance(evaluation_time_points, (type(None), list, ListConfig)) else evaluation_time_points

        self.params = []
        self.networks = [self.net]
        for n in self.networks:
            if n is not None:
                self.params.extend(list(n.parameters()))

        # save the params.
        self.save_hyperparameters()

    def unpack_batch(self, batch):
        data, (durations, events) = batch
        return data, durations, events

    def configure_optimizers(self):
        if isinstance(self.optimizer, str): self.optimizer = eval(self.optimizer)
        if isinstance(self.schedule, str): self.schedule = eval(self.schedule)
        self.optimizer_kwargs["lr"] = self.learning_rate

        optimizer = self.optimizer(self.params, **self.optimizer_kwargs)
        print(f'Using Optimizer {str(optimizer)}.')
        if self.schedule is not None:
            print(f'Using Scheduler {str(self.schedule)}.')
            schedule = self.schedule(optimizer, **self.schedule_kwargs)
            if isinstance(self.schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'scheduler': schedule,
                    'monitor': 'Ctd_0.9',
                }
            else:
                return [optimizer], [schedule]
        else:
            print('No Scheduler specified.')
            return optimizer

    def ext_dataloader(self, ds, batch_size=None, shuffle=False, num_workers=None,
                       drop_last=False):  ### Already transformed datamodules? -> Or pass transformers?
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, drop_last=drop_last)

    def training_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data, events)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        loss_dict = dict([(f'val_{k}', v) for k, v in loss_dict.items()])
        for k, v in loss_dict.items():
            self.log(k, v, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        return loss_dict

    def validation_epoch_end(self, outputs):
        metrics = {}
        # aggregate the per-batch-metrics:
        for metric_name in ["val_loss"]:
            # for metric_name in [k for k in outputs[0].keys() if k.startswith("val")]:
            metrics[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        # calculate the survival metrics
        valid_ds = self.val_dataloader().dataset if not \
            isinstance(self.val_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.train_dataloader().dataset.dataset
        metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=valid_ds,
                                                           time_points=self.evaluation_time_points,
                                                           quantile_bins = self.evaluation_quantile_bins)
        # train metrics:
        if self.hparams.report_train_metrics:
            train_metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=train_ds,
                                                                     time_points = self.evaluation_time_points,
                                                                     quantile_bins = self.evaluation_quantile_bins)
            for key, value in train_metrics_survival.items():
                metrics[f'train_{key}'] = value

        for key, value in metrics_survival.items():
            metrics[f'valid_{key}'] = value

        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[5, 10], quantile_bins=None):
        """
        Calculate epoch level survival metrics.
        :param train_ds:
        :param valid_ds:
        :param time_points: times at which to evaluate.
        :param quantile_bins: ALTERNATIVELY (to time_points) -> pass quantiles of the time axis.
        :return:
        """
        metrics = {}
        ctds = []
        cs = []

        assert None in [time_points, quantile_bins], 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

        # move to structured arrays:
        struc_surv_train = np.array([(e, d) for e, d in zip(surv_train[0], surv_train[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])
        struc_surv_valid = np.array([(e, d) for e, d in zip(surv_valid[0], surv_valid[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=32, num_workers=0, shuffle=False, drop_last=False)

        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        for i, tau in enumerate(taus):
            risks = []
            tau_ = torch.Tensor([tau])
            with torch.no_grad():
                for batch in loader:
                    data, durations, events = self.unpack_batch(batch)
                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
                    risk = self.predict_risk(data, t=tau_)  # returns RISK (e.g. F(t))
                    risks.append(risk.detach().cpu().numpy())
            try:
                risks = np.concatenate(risks, axis=0)
            except ValueError:
                risks = np.asarray(risks)
            risks = risks.ravel()
            risks[pd.isna(risks)] = np.nanmax(risks)
            Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                         risks,
                                         tau=tau, tied_tol=1e-8)
            C = concordance_index(event_times=surv_valid[1],
                                  predicted_scores=-risks,
                                  event_observed=surv_valid[0])
            ctds.append(Ctd[0])
            cs.append(C)

        self.train()

        for k, v in zip(annot, ctds):
            metrics[f'Ctd_{k}'] = v
        for k, v in zip(annot, cs):
            metrics[f'C_{k}'] = v

        return metrics

    def shared_step(self, data, duration, events):
        """
        shared step between training and validation. should return a tuple that fits in loss.
        :param data:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract method")
        return durations, events, some_args

    def loss(self, predictions, durations, events):
        """
        Calculate Loss.
        :param predictions:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract Class")
        loss1 = None
        loss2 = None
        loss = None
        return {'loss': loss,
                'loss1': loss1,
                'loss2': loss2,}

    def predict_dataset(self, ds, times):
        """
        Predict the survival function for each sample in the dataset at all durations in the dataset.
        Returns a pandas DataFrame where the rows are timepoints and the columns are the samples. Values are S(t|X)
        :param ds:
        :param times: a np.Array holding the times for which to calculate the risk.
        :return:
        """
        raise NotImplementedError("Abstract method")

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples: n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)
        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()
        for t in times:
            if hasattr(self, 'isoreg'):
                if f"1_{t}_Ft" in self.isoreg:
                    pass
                else:
                    for i, array in enumerate([pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values]):
                        if len(list(np.argwhere(np.isnan(array))))>0:
                            print(i)
                            print(np.argwhere(np.isnan(array)))
                    F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                    self.isoreg[f"1_{t}_Ft"] = IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"1_{t}_Ft"].values, F_t_obs)
            else:
                F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                self.isoreg = {f"1_{t}_Ft": IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"1_{t}_Ft"].values, F_t_obs)}

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"1_{t}_Ft__calibrated"] = self.isoreg[f"1_{t}_Ft"].predict(pred_df[f"1_{t}_Ft"])
        return pred_df

    def forward(self, X, t=None):
        """
        Predict a sample
        :return: f_t, F_t, S_t
        """
        raise NotImplementedError("Abstract method")

    def predict_risk(self, X, t=None):
        """
        Predict risk for X. Risk and nothing else.
        :param X:
        :param t:
        :return:
        """
        raise NotImplementedError("Abstract method")

class DeepSurv(AbstractSurvivalTask):
    """
    Train a DeepSurv-Model
    """
    def __init__(self, network,
                 transforms=None,
                 batch_size=128,
                 num_workers=8,
                 learning_rate=1e-3,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 return_risk=True,
                 **kwargs
                 ):
        """
        CoxPH Training in pl
        :param network:         `nn.Module` or `pl.LightningModule`, the network
        :param transforms:      `nn.Module` holding the transformations to apply to datamodules if necessary
        :param batch_size:      `int`, batchsize
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
        """
        super().__init__(
            network=network,
            transforms=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
        )
        # save the params.
        self.save_hyperparameters()
        self.n_events = 1
        self.bhs = None
        self.return_risk = return_risk

    @property
    def __name__(self):
        return 'CoxPH'

    def shared_step(self, data, durations, events):
        logh = self.net(data)
        return durations, events, logh

    def loss(self, durations, events, logh):
        nll = cox_ph_loss(logh, durations, events)
        return {'loss': nll}

    def on_fit_start(self):
        if self.bhs is None and self.return_risk:
            self.bhs = self.compute_bhs()

    def baseline_hazards(self, dataset):
        if self.return_risk:
            self.bhs = self.compute_bhs(dataset=dataset)

    def on_save_checkpoint(self, checkpoint):
        if self.return_risk:
            checkpoint["bhs"] = self.bhs
        else:
            checkpoint["bhs"] = None

    def on_load_checkpoint(self, checkpoint):
        if self.return_risk:
            self.bhs = checkpoint["bhs"]

    def forward(self, X, t=None):
        """
        Predict a sample
        :return:
        """
        tens = torch.Tensor(2, 3)
        log_ph = self.net(X.to(self.device))
        return log_ph, log_ph, log_ph # to fit the general interface of this method

    def predict_risk(self, X, t=None):
        """
        Predict RISK for a sample.
        :parameter X: Input Data
        :parameter t: Timepoint to predict risk at
        :return: `torch.tensor` Risk
        """
        t = t.item()
        log_ph, *_ = self.forward(X)

        if self.return_risk:
            try:
                # Check if t is in self.bhs
                cum_bhs = self.bhs.at[t, 'cum_bhs']
            except KeyError:
                # If t is not in dataframe, interpolate and insert t to the dataframe
                t_row = pd.Series({'bhs': float('nan'), 'cum_bhs': float('nan')}, name=t)
                self.bhs = self.bhs.append(t_row).sort_index().interpolate(method='linear', axis=0)
                cum_bhs = self.bhs.at[t, 'cum_bhs']

            S_t = torch.exp(-torch.exp(log_ph) * cum_bhs)

            return 1 - S_t
        else:
            return log_ph

    def compute_bhs(self, dataset=None):
        """
        Fits CoxPH on the entirety of train-set, and extracts the baseline hazard from the fitter
        :parameter dataset: dataset to fit CoxPH to
        :return `pd.Dataframe` baseline hazards dataframe with timepoints as index
        """
        bhs = self._compute_bhs_lifelines(dataset)
        return bhs

    def _compute_bhs_native(self):
        raise NotImplementedError("native baseline hazard computation is not yet functional")

    def _compute_bhs_lifelines(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataloader().dataset if not \
                isinstance(self.train_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
                else self.train_dataloader().dataset.dataset

        # collect durations, events, and data
        loader = self.ext_dataloader(dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=False)
        durations = []
        events = []
        data = []

        with torch.no_grad():
            for batch in loader:
                feats, d, e = self.unpack_batch(batch)
                data.append(feats.cpu().detach())
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
            del loader

        data = torch.cat(data, dim=0).numpy()
        durations = torch.cat(durations, dim=0).numpy()
        events = torch.cat(events, dim=0).numpy()

        # create full dataframe
        feature_df = pd.DataFrame.from_records(data)
        label_dict = dict(durations=durations.ravel(), events=events.ravel())
        label_df = pd.DataFrame.from_records(label_dict)
        full_df = pd.concat([feature_df, label_df], axis=1)

        # fit CoxPH
        cph = CoxPHFitter()
        cph.fit(full_df,
                duration_col="durations",
                event_col="events",
                show_progress=False,
                step_size=0.5)

        # interpolate on time
        bhs = pd.concat([cph.baseline_hazard_, cph.baseline_cumulative_hazard_], axis=1)
        bhs.columns = ['bhs', 'cum_bhs']
        bhs.index.name = 'time'
        bhs.interpolate()
        bhs.reindex(np.linspace(0, np.max(durations))).interpolate(method='linear',
                                                                   inplace=True,
                                                                   limit_direction='forward')
        bhs = bhs.dropna()
        del durations
        del events
        del data

        return bhs

    def predict_dataset(self, dataset: object, times: list):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        assert self.bhs is not None, 'Model has to have baseline hazards calulated on train set'

        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                log_hs.append(self.forward(data)[0].cpu().detach())
                del data
            del loader

        log_hs = torch.cat(log_hs, dim=0).numpy()

        pred_df = pd.DataFrame(log_hs.ravel(), columns=['loghs'])

        pred_df['hs'] = np.exp(log_hs).ravel()
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()

        if self.return_risk:
            H_all = pd.DataFrame(np.matmul(np.expand_dims(self.bhs['cum_bhs'].values, -1),
                                           np.exp(log_hs).T),
                                 index=[i for i in self.bhs.index.values])  # [n_durations x n_samples]
            S_all = np.exp(-H_all)

            S_tX = []
            for t in times:
                S_tX.append(S_all.iloc[[S_all.index.get_loc(t, method="nearest")]].values)
            S_tX = np.concatenate(S_tX, axis=0)  # [times, n_samples]

            F_tX = 1 - S_tX

            for t_i, t in enumerate(times):
                pd.concat([pred_df, pd.DataFrame.from_dict({
                    f"0_{t}_Ft": F_tX[t_i, :].ravel(),
                    f"0_{t}_St": F_tX[t_i, :].ravel()})], axis=1)

        return pred_df

    def extract_latent(self, dataset: object):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        # bhs = self.bhs

        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                log_hs.append(self.forward(data)[0].detach())
                del data
            del loader

        log_hs = torch.cat(log_hs, dim=0).numpy()
        hs = torch.exp(torch.cat(log_hs), dim=0).numpy()

        pred_df = pd.DataFrame(log_hs.ravel(), columns=['loghs'])
        pred_df['hs'] = hs.ravel()
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df