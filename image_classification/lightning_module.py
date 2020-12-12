from pytorch_lightning.metrics import Accuracy, Precision, Recall
from torch.optim import Adam, lr_scheduler
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):
    lr = 1e-3

    def __init__(self, num_classes):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = {
            "accuracy": Accuracy(),
            "recall_macro": Recall(num_classes=num_classes, average="macro"),
            "precision_macro": Precision(num_classes=num_classes, average="macro"),
        }

        self.model = models.resnet50(pretrained=True)

        ## Only the last layer is trained
        # for p in self.model.parameters():
        #     p.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.num_classes = num_classes
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)

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
