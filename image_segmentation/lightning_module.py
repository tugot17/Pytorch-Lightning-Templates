import torch
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class ImageSegmentator(pl.LightningModule):
    lr = 1e-3

    def __init__(self):
        super().__init__()

        self.criterion = smp.utils.losses.DiceLoss()
        self.metrics = {
            "IoU": smp.utils.metrics.IoU(threshold=0.5),
            "FScore": smp.utils.metrics.Fscore(),
        }

        self.model = smp.Unet(
            "resnet50", encoder_weights="imagenet", in_channels=3, classes=1
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)

        metrics_dict = {
            f"val/{name}": metric.to(self.device)(logits, y)
            for name, metric in self.metrics.items()
        }

        self.log_dict({**{"val/loss": loss}, **metrics_dict})

        return loss