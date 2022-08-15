"""
1. Donwload / tokenize, process
2. Clean and save to disk
3. Dataset
4. Transform
5. Dataloader 
"""

from netrc import netrc
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

save_path = "../data"
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_data = MNIST(root=save_path, train=True, download=True, transform=transforms)
test_data = MNIST(root=save_path, train=False, download=False, transform=transforms)
predict_data = MNIST(root=save_path, train=False, download=False, transform=transforms)

train_data, val_data = random_split(dataset=train_data, lengths=[55000, 5000])

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
predict_loader = DataLoader(predict_data, batch_size=32)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        MNIST(root=self.data_dir, download=True)

    def setup(self, stage):
        """
        stage = ["fit", "test", "predict"]
        """

        if stage == "fit":
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(mnist, [55000, 5000])

        if stage == "test":
            mnist = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_set = mnist

        if stage == "predict":
            mnist = MNIST(self.data_dir, train=False, transform=self.transform)
            self.predict_set = mnist

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=32)


datamodule = MNISTDataModule()
trainer = pl.Trainer()

trainer.fit(
    model=net,
    train_dataloaders=datamodule.train_dataloader(),
    val_dataloaders=datamodule.val_dataloader(),
)

trainer.fit(
    model=net,
    datamodule=datamodule
)
trainer.validate(
    model=net,
    datamodule=datamodule
)
trainer.test(
    model=net,
    datamodule=datamodule
)
trainer.predict(
    model=net,
    datamodule=datamodule
)

if args.mode == "train":
    func = trainer.fit
elif args.mode == "validation":
    func = trainer.validate

func(model=net, datamodule=datamodule)