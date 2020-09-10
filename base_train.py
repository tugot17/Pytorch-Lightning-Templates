import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42)

if __name__ == '__main__':
    dm = ...

    model = ...

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
    EXPERIMENT_NAME = "experient"
    logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name=EXPERIMENT_NAME)

    #And then actual training
    trainer = Trainer(max_epochs=40,
                      logger=logger,
                      gpus=1,
                      # precision=16,
                      accumulate_grad_batches=4,
                      deterministic=True,
                      early_stop_callback=True,
                      # resume_from_checkpoint = 'my_checkpoint.ckpt'
                      )

    trainer.fit(model, dm)