from __future__ import annotations

import torch
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
from lightning import LightningModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src import mask
from src.dataset import DATASET_MODES, SpikesDataset

class Flip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        X, Y = batch
        if torch.rand(1) < .5:
            return (X.flip(dims=[2]), Y.flip(dims=[2]))
        else:
            return (X, Y)

class UnetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        data_source: str,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.net = net
        self.data_source = data_source

    def setup(self, stage: str):
        self.criterion = nn.PoissonNLLLoss(
            log_input=True,
            full=False,
        )
        self.criterion_base = nn.PoissonNLLLoss(
            log_input=False,
            full=False,
        )
        self.masker = mask.Masker(mask.MaskParams(MASK_MODE = mask.MaskMode.neuron), self.device)
        self.train_loss = torchmetrics.MeanMetric()
        self.train_dataset = SpikesDataset(self.data_source)
        self.val_r2 = torchmetrics.MeanMetric()
        self.val_dataset = SpikesDataset(self.data_source, DATASET_MODES.val)
        self.transform = torch.nn.Sequential(
            Flip(),
        )


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False)

    def model_step(self, batch, masking: bool = True):
        # Use a masked langueage model and predict the missing values
        X, rate, _, _ = batch
        if masking:
            X_masked, labels = self.masker.mask_batch(X)
        else:
            X_masked = X.clone()
            labels = torch.ones_like(X) * mask.UNMASKED_LABEL

        masked = labels != mask.UNMASKED_LABEL
        X_smoothed = self.net((X_masked).to(torch.float32))
        loss = self.criterion(X_smoothed[masked], X[masked])
        loss_base = self.criterion_base(X[masked], X[masked])
        return loss - loss_base, X_smoothed, X, masked, rate

    def on_train_start(self):
        self.train_loss.reset()

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def training_step(self, batch, batch_idx: int):
        X, rate, _, _ = batch
        X, rate = self.transform((X, rate))
        batch = (X, rate, _, _)
        self.last_step = self.model_step(batch)

        loss, preds, targets, mask, rate = self.last_step

        assert targets.min() >= 0, "Negative targets"

        preds_exp = torch.exp(preds)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/mean_preds", preds_exp.mean())
        self.log("train/std_preds", preds_exp.std())
        self.log("train/mean_targets", targets.to(torch.float32).mean())
        self.log("train/std_targets", targets.to(torch.float32).std())
        self.log("train/mean_mask", mask.to(torch.float32).mean())

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, preds, _, _, rate = self.model_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if rate is not None:
            # Figure out whether we can predict the rate well
            _, preds, targets, mask, rate = self.model_step(batch, masking=False)
            r2 = torch.corrcoef(torch.stack([preds.ravel(), rate.ravel()]))[0, 1] ** 2
            self.val_r2(r2)
            self.log("val/r2", self.val_r2, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # Show the last step
        loss, preds, targets, mask, rate = self.last_step
        tensorboard = self.get_tb()
        if tensorboard is not None:
            tensorboard.add_image(
                "debug/preds", torch.exp(preds[0]), self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/targets", targets[0], self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/mask", mask[0], self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/target_rates",
                torch.fmax(
                    torch.fmin(torch.exp(rate[0]), torch.Tensor([1.0]).to(device=self.device)),
                    torch.Tensor([0.0]).to(device=self.device),
                ),
                self.current_epoch,
                dataformats="HW",
            )
            plt.figure(figsize=(8, 4))
            plt.imshow(torch.exp(rate[0]).detach().cpu().numpy() - torch.exp(preds[0]).detach().cpu().numpy())
            plt.colorbar()
            tensorboard.add_figure("debug/deltas", plt.gcf(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), self.hparams.lr, amsgrad=True  # type: ignore
        )
        return {
            "optimizer": optimizer,
        }

    def get_tb(self) -> SummaryWriter | None:
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                return logger.experiment
