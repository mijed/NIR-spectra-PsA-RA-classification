import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from config.consts import LR

#
class ConvNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=198464, out_features=3)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return self.classifier(x)


class LitConvNet(pl.LightningModule):
    def __init__(self, convnet):
        super().__init__()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=3)
        self.model = convnet

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x,y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat.argmax(dim=1), y)
        self.log('train_acc_step', self.accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer
