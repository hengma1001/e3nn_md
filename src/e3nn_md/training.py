import os
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


class SaveModelCallback(Callback):
    def __init__(self, dirpath: Optional[Path] = None, every_n_epochs: int = 1):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs
    ):
        self.dirpath = os.path.join(trainer.default_root_dir, "models")
        if trainer.current_epoch % self.every_n_epochs == 0:
            model_savepath = Path(self.dirpath) / f"epoch={trainer.current_epoch}.ckpt"
            trainer.save_checkpoint(model_savepath, weights_only=True)


# class SampleTimeCallback(Callback):
#     def on_validation_batch_end(
#         self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs
#     ):
#         trainer.current_step
#         trainer.logger.experiment.log()
#         self.dirpath = os.path.join(trainer.default_root_dir, "models")
#         if trainer.current_epoch % self.every_n_epochs == 0:
#             model_savepath = Path(self.dirpath) / f"epoch={trainer.current_epoch}.ckpt"
#             trainer.save_checkpoint(model_savepath, weights_only=True)


def train_diffu(
    model,
    default_root_dir,
    train_loader,
    val_loader,
    test_loader,
    n_gpus=1,
    max_epochs=1,
    every_n_epochs: Optional[int] = 1,
    **kwargs,
):
    trainer = L.Trainer(
        default_root_dir=default_root_dir,
        accelerator="auto",
        devices=n_gpus,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(default_root_dir, "checkpoints"),
                every_n_epochs=every_n_epochs,
                save_weights_only=True,
            ),
            LearningRateMonitor("epoch"),
            SaveModelCallback(every_n_epochs=every_n_epochs),
        ],
        **kwargs,
    )

    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
