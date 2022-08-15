# ModelCheckpoint

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

ckpt_callback = ModelCheckpoint(
    dirpath="my/paty",
    monitor="val_acc",
    mode="max",
    save_top_k=3,
    filename="ckpt_name-{epoch}-{val_acc:.2f}",
)

trainer = Trainer(callbacks=[ckpt_callback])

# EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping
es_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.1, patience=10
)
es_callback = EarlyStopping(
    monitor="val_acc", min_delta=0.1, patience=10, mode="max"
)
trainer = Trainer(callbacks=[es_callback, ckpt_callback])

# ModelSummary
from pytorch_lightning.callbacks import ModelSummary
summary_callback = ModelSummary(max_depth=1)
trainer = Trainer(callbacks=[summary_callback])
