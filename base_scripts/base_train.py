import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    dm = ...

    model = ...

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
