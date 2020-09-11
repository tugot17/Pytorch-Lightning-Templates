import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import albumentations as A
from albumentations.pytorch import ToTensorV2


from image_segmentation.datamodule import ImageSegmentationDatamodule
from image_segmentation.lightning_module import ImageSegmentator

pl.seed_everything(42)

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

    model = ImageSegmentator()

    #Fast run first
    trainer = Trainer(gpus=1, fast_dev_run=True, checkpoint_callback=False)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    TENSORBOARD_DIRECTORY = "logs/"
    EXPERIMENT_NAME = "Unet_resnet50"
    logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name=EXPERIMENT_NAME)

    #And then actual training
    trainer = Trainer(max_epochs=40,
                      logger=logger,
                      gpus=1,
                      # precision=16,
                      accumulate_grad_batches=4,
                      deterministic=True,
                      early_stop_callback=True,
                      checkpoint_callback=checkpoint_callback,
                      # resume_from_checkpoint = 'my_checkpoint.ckpt'
                      )

    trainer.fit(model, dm)