from typing import Any, List, Union
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
from torch.optim import Adam
import torch 
class CIFAR10Classifier(LightningModule):
    def __init__(self, model=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if model is None:
            raise NotImplementedError("You need to pass a model to the classifier")
        
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        return opt
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        return {'loss': loss, 'acc': acc}
    
    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, sync_dist=True)
        self.log("train_acc", avg_acc, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        return {'loss': loss, 'acc': acc}
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_acc", avg_acc, sync_dist=True)
    