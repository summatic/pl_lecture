from re import S
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class TorchModel(nn.Module):
    def __init__(self) -> None:
        super(TorchModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.net(x)
        loss = F.log_softmax(x, dim=1)
        return loss

class PlModel1(pl.LightningModule):
    def __init__(self) -> None:
        super(PlModel1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.net(x)
        loss = F.log_softmax(x, dim=1)
        return loss

class PLModel(pl.LightningModule):
    def __init__(self):
        super(PLModel, self).__init__()
        self.net = TorchModel()

        self.encoder
        self.decoder
    
    def forward(self, x):
        x = self.net(x)
        loss = F.log_softmax(x, dim=1)

        self.encoder(x)

        return loss
