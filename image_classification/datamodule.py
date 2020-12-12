import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .dataset import ImageClassificationDataset


class ImageClassificationDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size, train_transform, val_transform):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        self.train_set = ImageClassificationDataset(
            file_paths=[
                "./images/image_1.jpg",
                "./images/image_2.jpg",
                "./images/image_3.jpg",
            ],
            labels=[1, 2, 3],
            transform=self.train_transform,
        )

        self.val_set = ImageClassificationDataset(
            file_paths=[
                "./images/image_1.jpg",
                "./images/image_2.jpg",
                "./images/image_3.jpg",
            ],
            labels=[1, 2, 3],
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    train_transform = A.Compose(
        [
            A.Resize(400, 400),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75
            ),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    dm = ImageClassificationDatamodule(16, train_transform, val_transform)
