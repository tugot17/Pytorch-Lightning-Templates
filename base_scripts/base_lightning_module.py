from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl


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
