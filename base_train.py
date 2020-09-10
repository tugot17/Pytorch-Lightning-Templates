import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

TENSORBOARD_DIRECTORY = "logs/"
EXPERIMENT_NAME = "experient"
pl.seed_everything(42)

if __name__ == '__main__':
    dm = ...

    model = ...

    #Fast run first
    trainer = Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model, dm)

    checkpoint_callback = ModelCheckpoint(
        filepath='model/',
        save_top_k=2,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name=EXPERIMENT_NAME)

    #And then actual training
    trainer = Trainer(max_epochs=40,
                      logger=logger,
                      gpus=1,
                      # precision=16,
                      accumulate_grad_batches=4,
                      deterministic=True,
                      early_stop_callback=True,
                      # resume_from_checkpoint = 'model/my_checkpoint.ckpt'
                      )

    trainer.fit(model, dm)