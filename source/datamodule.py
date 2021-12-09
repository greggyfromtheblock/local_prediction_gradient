import pytorch_lightning as pl
from lifelines import CoxPHFitter

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from riskiano.datamodules.datasets import *


class SyntheticDatamodule(pl.LightningDataModule):
    def __init__(self, csv_id, batch_size, num_workers = 8,
                 dirpath= '/data/analysis/ag-reils/ag-reils-shared/cardioRS/data/interpretability/',
                 **kwargs):
        """
        :param csv_id: name of the csv to be used for training
        :param batch_size: batch size to be used for tha loaders
        :param num_workers: number of workers xd
        :param dirpath: path of directory where the csv can be found,
                        this path has to end with /
        """
        super().__init__()
        # Parameters
        self.dirpath = dirpath
        self.batch_size = batch_size
        self.datatransform = StandardScaler(copy=True)
        self.num_workers = num_workers
        self.csv_id = csv_id


        # Class attributes
        self.surv_df, *_ = self.create_df_from_csv()
        self.train_ds = None
        self.valid_ds = None
        self.eval_timepoint = self.surv_df['time'].quantile(0.9)
        self.features = [data for data in self.surv_df.columns][:-2]
        self.n_features = len(self.features)


    def setup(self, stage=None):
        self.train_ds = SyntheticDataset(name=self.csv_id, path=self.dirpath, mode='train', labeltransform=None,
                                         datatransform=self.datatransform)
        self.valid_ds = SyntheticDataset(name=self.csv_id, path=self.dirpath, mode='valid', labeltransform=None,
                                         datatransform=self.train_ds.datatransform)
        self.attribute_x = SyntheticDataset(name=self.csv_id, path=self.dirpath, mode='attribute_x', labeltransform=None,
                                             datatransform=self.train_ds.datatransform)
        self.attribute_y = SyntheticDataset(name=self.csv_id, path=self.dirpath, mode='attribute_y', labeltransform=None,
                                            datatransform=self.train_ds.datatransform)

    def train_dataloader(self):
        return DataLoader(BatchedDS(self.train_ds, batch_size=self.batch_size),
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=BatchedDS.default_collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(BatchedDS(self.valid_ds, batch_size=self.batch_size),
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=BatchedDS.default_collate, shuffle=False)

    def create_df_from_csv(self):
        train_df = pd.read_csv(f'{self.dirpath}{self.csv_id}_train.csv')
        valid_df = pd.read_csv(f'{self.dirpath}{self.csv_id}_valid.csv')
        return train_df.append(valid_df, ignore_index=True), train_df, valid_df

    def get_coxph_coeffs(self):
        cph = CoxPHFitter()
        cph = cph.fit(self.surv_df, 'time', 'event')
        coefs = cph.params_.tolist()
        return coefs