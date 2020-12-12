from icevision.all import *
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, lr_scheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .datamodule import ArtifactsDetectionDataModule

pl.seed_everything(42)


class LightModel(faster_rcnn.lightning.ModelAdapter):
    lr = 1e-3

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    batch_size = 16
    dm = ArtifactsDetectionDataModule(batch_size)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    backbone_model = faster_rcnn.model(num_classes=len(dm.parser.class_map))

    light_model = LightModel(backbone_model, metrics=metrics)

    # Fast run first
    trainer = Trainer(
        gpus=1, fast_dev_run=True, checkpoint_callback=False, logger=False
    )
    trainer.fit(light_model, dm)

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

    trainer.fit(light_model, dm)

    torch.save(light_model.model.state_dict(), "faster_rcnn_224.pth")
