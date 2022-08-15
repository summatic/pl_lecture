import pytorch_lightning as pl

trainer = pl.Trainer()


# Accelerator
trainer = pl.Trainer(accelerator="cpu")

trainer = pl.Trainer(accelerator="gpu", device=2)

trainer = pl.Trainer(accelerator="tpu", device=2)


# Device
trainer = pl.Trainer(accelerator="gpu", device=3) # 0, 1, 2
trainer = pl.Trainer(accelerator="gpu", device=[0, 1, 2])
trainer = pl.Trainer(accelerator="gpu", device="3")

trainer = pl.Trainer(accelerator="gpu", device=-1)
trainer = pl.Trainer(accelerator="gpu", device="-1")

trainer = pl.Trainer(accelerator="gpu", device=[7])
trainer = pl.Trainer(accelerator="gpu", device="0, 1")


# Strategy
trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=4)
trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu", devices=4)

trainer = pl.Trainer(strategy="ddp_shared")  # fairscale
trainer = pl.Trainer(strategy="deepspeed")  # deepspeed
trainer = pl.Trainer(strategy="bagua")  # bagua

trainer = pl.Trainer(strategy="horovod")  # horovod


# max_epochs / max_steps
trainer = pl.Trainer(max_epochs=100)
trainer = pl.Trainer(max_epochs=-1)

trainer = pl.Trainer(max_steps=100)
trainer = pl.Trainer(max_steps=-1)  # disable

# check_val_every_n_epochs
trainer = pl.Trainer(check_val_every_n_epoch=5)

# log_every_n_steps
trainer = pl.Trainer(log_every_n_steps=5)

# logger
# csv_logs, tensorboard
# comet, mlflow, nepture, wandb
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger(save_dir="curr/path", name="test")
trainer = pl.Trainer(logger=logger)


# Debugging
# fast_dev_run
trainer = pl.Trainer(fast_dev_run=7) 

# overfit_batches
trainer = pl.Trainer(overfit_batches=1.0)

trainer = pl.Trainer(overfit_batches=0.25)

trainer = pl.Trainer(overfit_batches=10)

# num_sanity_val_steps
trainer = pl.Trainer(num_sanity_val_steps=2)
trainer = pl.Trainer(num_sanity_val_steps=0)
trainer = pl.Trainer(num_sanity_val_steps=-1)



# gradient_clip_val
trainer = pl.Trainer(gradient_clip_val=4, gradient_clip_algorithm="value")


# resume_from_checkpoint
trainer = pl.Trainer(resume_from_checkpoint="some/path/checkpoint.ckpt")