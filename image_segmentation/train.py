import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import albumentations as A
from albumentations.pytorch import ToTensorV2


from .datamodule import ImageSegmentationDatamodule
from .lightning_module import ImageSegmentator

if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5
            ),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dm = ImageSegmentationDatamodule(16, train_transform, val_transform)

    model = ImageSegmentator()

    # Fast run first
    trainer = Trainer(
        gpus=1, fast_dev_run=True, checkpoint_callback=False, logger=False
    )


    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=2,
        verbose=True,
        monitor="val/loss",
        mode="min",
    )

    experiment_name = ...
    PROJECT_NAME = ...

    logger = WandbLogger(name=experiment_name, project=PROJECT_NAME)

    # And then actual training
    pl.seed_everything(42)
    trainer = Trainer(
        max_epochs=40,
        logger=logger,
        gpus=1,
        # precision=16,
        deterministic=True,
        accumulate_grad_batches=2,
        callbacks=[EarlyStopping(monitor="val/loss")],
        # resume_from_checkpoint = 'my_checkpoint.ckpt'
    )

    trainer.fit(model, dm)
