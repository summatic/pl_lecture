import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DVICE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger


class Model(pl.LightningModule):
    def __init__(self, hidden_dim, dropout_rate, activation):
        super(Model, self).__init__()

        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        else:
            activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            activation,
            nn.Conv2d(32, 64, 3, 1),
            activation,
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, hidden_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        x = self.net(x)
        loss = F.log_softmax(x, dim=1)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, target = train_batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch
        output = self.forward(data)
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        self.log("val_loss", loss)
        self.log("val_acc", 100 * correct / len(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", default=50)
    parser.add_argument("--dropout_rate", default=0.1)
    parser.add_argument("--activation", default="relu")
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Model(
        hidden_dim=int(args.hidden_dim)*100,
        dropout_rate=float(args.dropout_rate)*0.1,
        activation=args.activation
    )
    logger = WandbLogger(project="pl_best", name="test")

    trainer = pl.Trainer(default_root_dir=os.getcwd(), logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()

