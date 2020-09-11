import os

from albumentations.pytorch import ToTensorV2
import albumentations as A

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from image_classification.datamodule import ImageClassificationDatamodule
from image_classification.lightning_module import ImageClassifier

pl.seed_everything(42)

if __name__ == '__main__':
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_transform = A.Compose([
        A.Resize(400, 400),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    dm = ImageClassificationDatamodule(16, train_transform, val_transform)

    model = ImageClassifier()

    #Fast run first
    trainer = Trainer(gpus=1, fast_dev_run=True, checkpoint_callback=False)
    trainer.fit(model, dm)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    TENSORBOARD_DIRECTORY = "logs/"
    EXPERIMENT_NAME = "resnet_50"
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