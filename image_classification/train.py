import os

from albumentations.pytorch import ToTensorV2
import albumentations as A

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .datamodule import ImageClassificationDatamodule
from .lightning_module import ImageClassifier


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

    model = ImageClassifier()

    # Fast run first
    trainer = Trainer(
        gpus=1, fast_dev_run=True, checkpoint_callback=False, logger=False
    )
    trainer.fit(model, dm)

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
