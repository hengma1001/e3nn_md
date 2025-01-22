import os
from pathlib import Path
from typing import Optional

import torch
import wandb

from .conv_model import e3_diffusion


def training(
    model: e3_diffusion,
    default_root_dir,
    train_loader,
    val_loader,
    test_loader,
    n_gpus=1,
    max_epochs=1,
    every_n_epochs: Optional[int] = 1,
    **kwargs,
):
    os.makedirs(default_root_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(max_epochs):
        wandb.log({"epoch": epoch})
        for step, data in enumerate(train_loader):
            loss = model._get_loss(data)
            wandb.log({"training loss": loss})
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            val_loss = 0.0
            for step, data in enumerate(val_loader):
                val_loss += model._get_loss(data)

            wandb.log({"val loss": val_loss / len(val_loader)})

    with torch.no_grad():
        test_loss = 0.0
        for step, data in enumerate(test_loader):
            test_loss += model._get_loss(data)

        wandb.log({"test loss": test_loss / len(test_loader)})
        # if epoch % every_n_epochs == 0:
        #     model.
