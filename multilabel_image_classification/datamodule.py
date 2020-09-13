from os.path import join, dirname, realpath
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from multilabel_image_classification.dataset import MultiLabelImageClassificationDataset
import pandas as pd

class MultiLabelImageClassificationDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size, train_transform, val_transform):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

        train_df_path = join(dirname(realpath(__file__)), "dataframe.csv")
        self.train_df = pd.read_csv(train_df_path)

        val_df_path = join(dirname(realpath(__file__)), "dataframe.csv")
        self.val_df = pd.read_csv(val_df_path)

    def setup(self, stage=None):
        self.train_set = MultiLabelImageClassificationDataset(
            df=self.train_df,
            transform=self.train_transform,
        )

        self.val_set = MultiLabelImageClassificationDataset(
            df=self.val_df,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)