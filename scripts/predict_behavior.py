#!/usr/bin/env python
"""
Train a linear decoder on top of a pretrained transformer autoencoder to decode behavior.

The script supports three training modes:
1. Frozen: Train only the decoder while keeping the pretrained model frozen
2. End-to-end: Two-phase training - first decoder only, then fine-tune everything
3. Random: Train from random initialization (no pretraining)
4. Passthrough: Train decoder directly on input spikes (no transformer)

Example
-------
# Mode 1: Frozen pretrained model
python train_decoder.py --checkpoint checkpoint.pt --mode frozen --dataset lorenz

# Mode 2: End-to-end fine-tuning (two-phase)
python train_decoder.py --checkpoint checkpoint.pt --mode end2end --dataset lorenz --decoder-only-epochs 20

# Mode 3: Random initialization
python train_decoder.py --mode random --dataset lorenz

# Mode 4: Passthrough (no transformer)
python train_decoder.py --mode passthrough --dataset lorenz

"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ndt_reimplementation import instantiate_autoencoder
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import dataset loading functions and model from the training scripts
from train_autoencoder import load_dataset


class BehaviorDecoder(nn.Module):
    """Linear decoder that maps from neural representations to behavior."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Neural representations (B, T, D)
        Returns:
            Decoded behavior (B, T, output_dim)
        """
        # print(x.mean().item(), x.std().item())
        return self.decoder(x)


def gaussian_smooth_1d(x, sigma=5):
    """
    Apply Gaussian smoothing along the time dimension.

    Args:
        x: Input tensor of shape (batch, time, neurons)
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Smoothed tensor of same shape as input
    """
    batch, time, neurons = x.shape

    # Create Gaussian kernel
    # Kernel size should be odd and large enough to capture the Gaussian
    kernel_size = int(6 * sigma + 1)  # 6 sigma captures 99.7% of distribution
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size

    # Create 1D Gaussian kernel
    kernel = torch.arange(kernel_size, dtype=torch.float32)
    kernel = kernel - kernel_size // 2  # Center around 0
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize

    # Move kernel to same device as input
    kernel = kernel.to(x.device)

    # Reshape for conv1d: (batch, time, neurons) -> (batch * neurons, 1, time)
    x_reshaped = x.permute(0, 2, 1).reshape(batch * neurons, 1, time)

    # Reshape kernel for conv1d: needs shape (out_channels, in_channels, kernel_size)
    kernel = kernel.view(1, 1, -1)

    # Apply convolution with padding to maintain time dimension
    padding = kernel_size // 2
    x_smoothed = F.conv1d(x_reshaped, kernel, padding=padding)

    # Reshape back: (batch * neurons, 1, time) -> (batch, time, neurons)
    x_smoothed = x_smoothed.view(batch, neurons, time).permute(0, 2, 1)

    return x_smoothed


class TransformerWithDecoder(nn.Module):
    """Combines pretrained PM Transformer with behavior decoder."""

    def __init__(
        self,
        transformer: nn.Module,
        behavior_dim: int,
        freeze_transformer: bool = True,
        passthrough: bool = False,
        target_layer: int = -1,
        input_dim: int = None,
    ):
        super().__init__()
        self.passthrough = passthrough
        self.transformer = transformer
        self.target_layer = target_layer

        if self.passthrough:
            # Decoder takes raw spike inputs
            assert input_dim is not None, (
                "input_dim must be specified for passthrough mode"
            )
            self.decoder = BehaviorDecoder(input_dim=input_dim, output_dim=behavior_dim)
        else:
            hidden_dim = transformer.hidden_dim

            self.decoder = BehaviorDecoder(
                input_dim=hidden_dim, output_dim=behavior_dim
            )

        self.set_freeze_transformer(freeze_transformer)

    def set_freeze_transformer(self, freeze: bool):
        """Freeze or unfreeze transformer parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = not freeze

        # DO NOT FREEZE THE INPUT LAYER
        self.transformer.input_projection.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input spikes (B, T, neurons)
        Returns:
            Tuple of (reconstructed spikes, decoded behavior)
        """
        # Get internal representations
        if self.passthrough:
            h = gaussian_smooth_1d(x, sigma=5)  # Smooth spikes if passthrough
        else:
            h = self.transformer.forward_until_layer(x, self.target_layer)

        # Decode behavior from representations
        behavior = self.decoder(h)

        # We don't need spike reconstruction for behavior decoding
        spikes = None

        return spikes, behavior


def compute_r2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Compute R² score."""
    # Flatten batch and time dimensions
    pred_flat = pred.reshape(-1, pred.shape[-1])
    true_flat = true.reshape(-1, true.shape[-1])

    # Compute R² for each behavior dimension
    r2_per_dim = []
    for i in range(pred.shape[-1]):
        # Compute correlation coefficient
        if pred_flat[:, i].std() > 0 and true_flat[:, i].std() > 0:
            corr = torch.corrcoef(torch.stack([pred_flat[:, i], true_flat[:, i]]))[0, 1]
            r2_per_dim.append(corr**2)
        else:
            r2_per_dim.append(torch.tensor(0.0))

    return torch.stack(r2_per_dim).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    epoch_r2s = []

    criterion_mse = nn.MSELoss()

    for spikes, _, behavior in loader:
        spikes = spikes.to(device).float()
        behavior = behavior.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        _, pred_behavior = model(spikes)

        # Compute loss
        loss = criterion_mse(pred_behavior, behavior)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_losses.append(loss.item())
        with torch.no_grad():
            r2 = compute_r2(pred_behavior, behavior)
            epoch_r2s.append(r2.item())

    return float(np.mean(epoch_losses)), float(np.mean(epoch_r2s))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    losses = []
    r2s = []

    criterion_mse = nn.MSELoss()

    with torch.no_grad():
        for spikes, _, behavior in loader:
            spikes = spikes.to(device).float()
            behavior = behavior.to(device).float()

            # Forward pass
            _, pred_behavior = model(spikes)

            # Compute metrics
            loss = criterion_mse(pred_behavior, behavior)
            r2 = compute_r2(pred_behavior, behavior)

            losses.append(loss.item())
            r2s.append(r2.item())

    return float(np.mean(losses)), float(np.mean(r2s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train behavior decoder on spike data.")

    # Dataset arguments
    p.add_argument(
        "--dataset",
        default="lorenz",
        choices=[
            "lorenz",
            "mc_maze",
            "mc_maze_small",
            "mc_maze_medium",
            "mc_maze_large",
        ],
        help="Dataset name.",
    )
    p.add_argument(
        "--data-root", default="data/h5", help="Path containing dataset files."
    )

    # Model arguments
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (for frozen/end2end modes).",
    )
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["frozen", "end2end", "random", "passthrough"],
        help="Training mode: frozen, end2end, random, or passthrough.",
    )

    # Architecture arguments (used for random mode or if loading fails)
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden dimension for Transformer layers.",
    )
    p.add_argument(
        "--num-layers", type=int, default=4, help="Number of Transformer layers."
    )
    p.add_argument(
        "--num-heads", type=int, default=1, help="Number of attention heads."
    )
    p.add_argument(
        "--ffn-dim", type=int, default=128, help="Feedforward dimension in Transformer."
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for Transformer layers.",
    )
    p.add_argument("--pos-encoding", default="sin", help="Positional encoding type.")
    p.add_argument(
        "--projection", default="linear", help="Projection type for transformer."
    )
    p.add_argument(
        "--context-forward",
        type=int,
        default=10,
        help="Context length for transformer.",
    )
    p.add_argument(
        "--context-backward",
        type=int,
        default=10,
        help="Context length for transformer.",
    )

    # Training arguments
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--decoder-only-epochs",
        type=int,
        default=20,
        help="Number of epochs to train decoder only in end2end mode before unfreezing transformer.",
    )

    # Logging arguments
    p.add_argument(
        "--logdir", default="runs_decoder", help="TensorBoard log directory."
    )
    p.add_argument(
        "--save-checkpoint",
        default="decoder_checkpoint.pt",
        help="File to save best model (by val R²).",
    )
    p.add_argument(
        "--target-layer",
        type=int,
        default=-1,
        help="Which layer to target for decoding behavior (-1 for last layer).",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=1,
        help="Which subset to use (1 for full datset, X for 1/X of the data)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.data_root)

    train_data = torch.from_numpy(data["train_data"][:: args.subset]).int()
    val_data = torch.from_numpy(data["val_data"]).int()

    # Load behavior data
    try:
        train_behavior = torch.from_numpy(
            data["train_behavior"][:: args.subset]
        ).float()
        val_behavior = torch.from_numpy(data["val_behavior"]).float()
        behavior_dim = train_behavior.shape[-1]
        print(f"Behavior dimensionality: {behavior_dim}")
    except KeyError:
        raise ValueError(f"Dataset {args.dataset} does not contain behavior data!")

    # Dummy ground truth for compatibility with DataLoader
    train_truth = train_data.clone()
    val_truth = val_data.clone()

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_data, train_truth, train_behavior),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_truth, val_behavior),
        batch_size=args.batch_size,
        shuffle=False,
    )

    n_neurons = train_data.shape[2]
    trial_len = train_data.shape[1]

    # Initialize transformer
    if args.mode in ["frozen", "end2end"]:
        if args.checkpoint is None:
            raise ValueError(f"Mode '{args.mode}' requires --checkpoint argument!")

        print(f"Loading pretrained model from {args.checkpoint}")
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Extract args from checkpoint if available
        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            # Use checkpoint args for model instantiation
            ckpt_args.dropout = args.dropout
            ckpt_args.context_forward = 0
            ckpt_args.context_backward = -1

            transformer = instantiate_autoencoder(ckpt_args, n_neurons, trial_len)
        else:
            raise ValueError("Checkpoint does not contain model args!")

        # Load pretrained weights
        if (
            n_neurons
            != checkpoint["model_state_dict"]["input_projection.weight"].shape[1]
        ):
            print(
                f"Warning: Checkpoint has {checkpoint['model_state_dict']['input_projection.weight'].shape[1]} neurons, but current data has {n_neurons}; initializing at random."
            )

            # Override the checkpoint
            ckpt = checkpoint["model_state_dict"]
            if "input_projection.weight" in ckpt:
                ckpt["input_projection.weight"] = (
                    transformer.input_projection.weight / 10.0
                )
            if "output_projection.weight" in ckpt:
                ckpt["output_projection.weight"] = transformer.output_projection.weight
            if "output_projection.bias" in ckpt:
                ckpt["output_projection.bias"] = transformer.output_projection.bias
            if "output_bias" in ckpt:
                ckpt["output_bias"] = transformer.output_bias
            for key, value in ckpt.items():
                print(f"{key}: {value.shape}")

        transformer.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    elif args.mode == "random":
        print("Initializing transformer with random weights")
        transformer = instantiate_autoencoder(args, n_neurons, trial_len)

    elif args.mode == "passthrough":
        print("Passthrough mode - no transformer will be used")
        # Create a dummy transformer that won't be used
        transformer = instantiate_autoencoder(args, n_neurons, trial_len)

    # Handle target layer
    if args.target_layer == -1:
        target_layer = args.num_layers - 1  # Last layer
    else:
        target_layer = args.target_layer

    # Create combined model
    # For end2end mode, start with frozen transformer
    initial_freeze = args.mode in ["frozen", "end2end"]
    model = TransformerWithDecoder(
        transformer=transformer,
        behavior_dim=behavior_dim,
        freeze_transformer=initial_freeze,
        passthrough=(args.mode == "passthrough"),
        target_layer=target_layer,
        input_dim=n_neurons,
    ).to(device)

    print(f"Training mode: {args.mode}")
    if args.mode == "end2end":
        print(
            f"Two-phase training: decoder-only for {args.decoder_only_epochs} epochs, then end-to-end"
        )
    print(f"Target layer: {target_layer}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Setup optimizer - will be recreated when we switch phases
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=50
    )

    # Setup logging
    writer = SummaryWriter(log_dir=f"{args.logdir}/{args.dataset}_{args.mode}")
    best_val_r2 = -float("inf")

    # Training loop
    print("Starting training...")
    phase_switched = False

    for epoch in range(1, args.epochs + 1):
        # Switch to end-to-end training after decoder-only epochs
        if (
            args.mode == "end2end"
            and epoch == args.decoder_only_epochs + 1
            and not phase_switched
        ):
            print(
                f"\n=== Phase 2: Switching to end-to-end training at epoch {epoch} ==="
            )
            model.set_freeze_transformer(False)

            # Create new optimizer with all parameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,  # Use lower learning rate for fine-tuning
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )

            print(
                f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )
            phase_switched = True

        train_loss, train_r2 = train_one_epoch(model, train_loader, device, optimizer)
        val_loss, val_r2 = evaluate(model, val_loader, device)

        # Log metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/r2", train_r2, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/r2", val_r2, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Log phase
        if args.mode == "end2end":
            phase = (
                "decoder_only" if epoch <= args.decoder_only_epochs else "end_to_end"
            )
            writer.add_scalar("phase", 0 if phase == "decoder_only" else 1, epoch)

        # Update learning rate
        # scheduler.step(-val_loss)

        # Print progress
        phase_str = ""
        if args.mode == "end2end":
            phase_str = f" [{('decoder-only' if epoch <= args.decoder_only_epochs else 'end-to-end')}]"

        tqdm.write(
            f"Epoch {epoch:03d}{phase_str} | "
            f"train loss {train_loss:.4f} | train R² {train_r2:.4f} | "
            f"val loss {val_loss:.4f} | val R² {val_r2:.4f}"
        )

        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_r2": val_r2,
                    "args": args,
                    "phase": "decoder_only"
                    if args.mode == "end2end" and epoch <= args.decoder_only_epochs
                    else "full",
                },
                args.save_checkpoint,
            )
            tqdm.write(f"  → Saved new best model (R² = {val_r2:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val R² = {best_val_r2:.4f}")
    print(f"Model saved to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
