import torch
from pytorch_lightning.metrics import Recall, Accuracy
from torch.optim import lr_scheduler
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl

from optimizers.over9000 import RangerLars


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-8, max=1 - 1e-8)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * (
            (1 - target) * torch.log(1 - output)
        )
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


class MultiLabelImageClassifier(pl.LightningModule):
    lr = 1e-3

    def __init__(self, number_of_classes, pos_weight=None):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.metrics = {}

        self.model = models.resnet50(pretrained=True)

        ## Only the last layer is trained
        # for p in self.model.parameters():
        #     p.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.number_of_classes = number_of_classes
        self.model.fc = nn.Linear(self.num_ftrs, self.number_of_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = RangerLars(self.parameters(), self.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer]

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
