from icevision.all import *
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam

from object_detection.datamodule import ArtifactsDetectionDataModule

pl.seed_everything(42)


class LightModel(faster_rcnn.lightning.ModelAdapter):
    lr = 1e-3

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

if __name__ == '__main__':
    batch_size = 16
    dm = ArtifactsDetectionDataModule(batch_size)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    backbone_model = faster_rcnn.model(num_classes=len(dm.parser.class_map))

    light_model = LightModel(backbone_model, metrics=metrics)


    #Fast run first
    trainer = Trainer(gpus=1, fast_dev_run=True, checkpoint_callback=False, logger=False)
    trainer.fit(light_model, dm)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    TENSORBOARD_DIRECTORY = "logs/"
    EXPERIMENT_NAME = "faster_rcnn_resnet_50"
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

    trainer.fit(light_model, dm)

    torch.save(light_model.model, "artifacts_detector.pth")
