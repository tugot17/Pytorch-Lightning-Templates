import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from os.path import join, dirname, realpath

from image_segmentation.dataset import ImageSegmentationDataset


class ImageSegmentationDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size, train_transform, val_transform):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        train_images_dir = join(dirname(realpath(__file__)), "data", "train", "images")
        train_masks_dir = join(dirname(realpath(__file__)), "data", "train", "images")

        val_images_dir = join(dirname(realpath(__file__)), "data", "validation", "images")
        val_masks_dir = join(dirname(realpath(__file__)), "data", "validation", "images")

        self.train_set = ImageSegmentationDataset(train_images_dir, train_masks_dir,
                                                  ["pet"], self.train_transform)

        self.val_set = ImageSegmentationDataset(val_images_dir, val_masks_dir, ["pet"],
                                                self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == '__main__':
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )

    dm = ImageSegmentationDatamodule(16, train_transform, val_transform)
