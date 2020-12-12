import torch
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from optimizers.over9000 import RangerLars


class BaseModel(pl.LightningModule):
    lr = 1e-3

    def __init__(self):
        super().__init__()

        self.criterion = ...
        self.metrics = {}

        self.model = ...

    def forward(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = RangerLars(self.parameters(), self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)

        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)

        metrics_dict = {
            f"val_{name}": metric(logits, y) for name, metric in self.metrics.items()
        }

        return {**{"val_loss": loss}, **metrics_dict}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {
            name: torch.stack([x[f"val_{name}"] for x in outputs]).mean()
            for name, metric in self.metrics.items()
        }

        tensorboard_logs["val_loss"] = avg_loss

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}
