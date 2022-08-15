
for epoch in range(1, 16):  # epoch
    losses = []
    for batch in dataloader:  # step
        loss = training_step()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss)

        training_step_end()
    training_epoch_end()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
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
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.net(x)
        loss = F.log_softmax(x, dim=1)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss, _ = self.get_loss(batch=train_batch, reduction="none")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, output = self.get_loss(batch=val_batch, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        self.log("val_loss", loss)
        self.log("val_acc", 100 * correct / len(target))

    def test_step(self, test_batch, batch_idx):
        loss, output = self.get_loss(batch=test_batch, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        self.log("test_loss", loss)
        self.log("test_acc", 100 * correct / len(target))

    def get_loss(self, batch, reduction):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target, reduction=reduction)
        return loss, output
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

