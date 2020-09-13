from icevision.all import *
import pandas as pd

from object_detection.parser import ArtifactParser


class ArtifactsDetectionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()

        self.batch_size = batch_size

        df = pd.read_csv("atrifacts_dataset_with_paths.csv")
        data_splitter = RandomSplitter([.8, .2], seed=42)

        self.parser = ArtifactParser(df)
        self.train_records, self.valid_records = self.parser.parse(data_splitter)

    def setup(self, stage=None):
        presize = 512
        size = 384

        crop_fn = partial(tfms.A.RandomSizedCrop, min_max_height=(size // 2, size // 2), p=.3)
        train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize, crop_fn=crop_fn), tfms.A.Normalize()])
        # train_tfms = tfms.A.Adapter([tfms.A.Normalize()])
        valid_tfms = tfms.A.Adapter([tfms.A.Normalize()])

        self.trainset = Dataset(self.train_records, train_tfms)
        self.valset = Dataset(self.valid_records, valid_tfms)

    def train_dataloader(self):
        return faster_rcnn.train_dl(self.trainset, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return faster_rcnn.train_dl(self.valset, batch_size=self.batch_size, num_workers=2, shuffle=False)
