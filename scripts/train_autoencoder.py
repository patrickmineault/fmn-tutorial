#!/usr/bin/env python
"""

Minimal command‑line script to train a masked Transformer autoencoder on neural
spike data.  The script:
  • loads one of the mc_maze datasets or the Lorenz datasets
  • trains a Transformer masking a fraction of timesteps;
  • logs epoch‑wise negative log‑likelihood (Poisson) and R² to TensorBoard and W&B;
  • saves the best checkpoint (lowest validation NLL).

Example
-------
python train_tutorial.py --dataset lorenz --data-root ./data/h5 \
                            --logdir ./runs/lorenz_ae --wandb-project neural-autoencoder

Requirements
------------
    pip install torch torchvision torchaudio tensorboard tqdm h5py nlb-tools wandb matplotlib
"""
from __future__ import annotations

import argparse
import h5py
import math
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_dataset(name: str, data_root: str) -> Dict[str, np.ndarray]:
    root = Path(data_root)
    with open(root / f"{name}_data.pkl", "rb") as f:
        return pickle.load(f)
    
def do_masking(batch: torch.Tensor, mask_ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask *mask_ratio* timesteps per trial (span width = 1)."""
    batch_size, num_timesteps = batch.shape[:2]

    width = torch.randint(1, 6, (1, )).item()  # random span length for each sample
    width = 1
    mask = torch.rand(batch_size, num_timesteps) < (mask_ratio / width)

    if width > 1:
        # Convolve to widen the mask
        kernel = torch.ones((width, ), dtype=torch.float32)
        mask = torch.nn.functional.conv1d(
            mask.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0),
            padding=width // 2
        ).squeeze(1) > 0.5
        if width % 2 == 0:
            mask = mask[:, :-1]

    # Replace some masked tokens with 0 (80%) or random spikes (20%)
    mask_token_ratio = 0.8
    random_token_ratio = 0.25

    replace_zero = (torch.rand_like(mask, dtype=float) < mask_token_ratio) & mask
    replace_rand = (torch.rand_like(mask, dtype=float) < random_token_ratio) & mask & ~replace_zero

    batch_mean = batch.to(float).mean().item()
    batch = batch.clone()  # avoid in‑place modification
    batch[replace_zero] = 0
    if replace_rand.any():
        rand_values = (torch.rand_like(batch, dtype=float) < batch_mean).to(torch.int)
        batch[replace_rand] = rand_values[replace_rand]

    return batch, mask


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    mask_ratio: float,
) -> float:
    net.train()
    epoch_losses = []
    i = 0
    for spikes, _, _ in loader:  # dataset returns a single tensor
        spikes = spikes.to(device)
        optimizer.zero_grad()
        spikes_masked, mask = do_masking(spikes, mask_ratio)
        preds = net(spikes_masked.float())
        loss = criterion(preds[mask], spikes[mask]).mean()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        i += 1
    return float(np.mean(epoch_losses))


def create_rate_comparison_plot(spikes, preds, ground_truth, num_neurons=5, num_trials=3):
    """Create a plot comparing predicted vs true firing rates for a subset of neurons."""
    fig, axes = plt.subplots(num_neurons, num_trials, figsize=(12, 2*num_neurons))
    if num_neurons == 1:
        axes = axes.reshape(1, -1)
    if num_trials == 1:
        axes = axes.reshape(-1, 1)
    
    for n in range(num_neurons):
        for t in range(num_trials):
            ax = axes[n, t]
            
            # Plot ground truth
            ax.plot(ground_truth[t, :, n].cpu().numpy(), 'b-', alpha=0.7, label='True' if t == 0 else '')
            ax.plot(preds[t, :, n].cpu().numpy(), 'r--', alpha=0.7, label='Pred' if t == 0 else '')
            ax.plot(spikes[t, :, n].cpu().numpy(), 'k--', alpha=0.7, label='Spikes' if t == 0 else '')
            
            ax.set_ylim(bottom=0)
            if n == 0:
                ax.set_title(f'Trial {t+1}')
            if t == 0:
                ax.set_ylabel(f'Neuron {n+1}\nRate')
            if n == num_neurons - 1:
                ax.set_xlabel('Time')
            
            if n == 0 and t == 0:
                ax.legend()
    
    plt.tight_layout()
    return fig


def evaluate(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    mask_ratio: float = 0.0,
    writer: SummaryWriter = None,
    global_step: int = 0,
    use_wandb: bool = False,
    has_ground_truth: bool = False
) -> Tuple[float, float]:
    net.eval()
    losses, r2s = [], []
    i = 0
    with torch.no_grad():
        for spikes, ground_truth, _ in loader:
            # Measure performance on the pretext task
            spikes = spikes.to(device)
            masked_spikes, mask = do_masking(spikes, mask_ratio) if mask_ratio > 0 else (spikes, torch.ones_like(spikes, dtype=torch.bool))
            preds = net(masked_spikes.float())
            loss = criterion(preds[mask], spikes[mask]).mean()
            losses.append(loss.item())

            # Measure performance on inferring latents (when that makes sense)
            preds = net(spikes.float())
            preds_rates = torch.exp(preds)  # Convert from log rates to rates
            
            # Calculate R² if we have ground truth that's different from spikes
            if has_ground_truth:
                corr_mat = torch.corrcoef(torch.concat([preds_rates.reshape(-1, spikes.shape[2]), ground_truth.to(device).reshape(-1, spikes.shape[2])], dim=1).T)
                corrs = torch.diag(corr_mat[:corr_mat.shape[0] // 2, corr_mat.shape[0] // 2:])
                r2 = (corrs ** 2).mean().item()
                r2s.append(r2)
            else:
                r2s.append(0.0)

            if i == 0 and writer is not None and global_step % 50 == 0 and has_ground_truth:
                # TensorBoard logging (keep existing)
                preds_show = preds_rates[5::16, :, 10::20].permute(0, 2, 1)
                ground_truth_show = ground_truth[5::16, :, 10::20].permute(0, 2, 1)
                preds_show = preds_show.reshape(-1, preds_show.shape[2]).detach().cpu()
                ground_truth_show = ground_truth_show.reshape(-1, ground_truth_show.shape[2]).detach().cpu()

                writer.add_image(f"predictions/epoch_{global_step}", preds_show, global_step=global_step, dataformats="HW")
                writer.add_image(f"ground_truth/epoch_{global_step}", ground_truth_show, global_step=global_step, dataformats="HW")
                
                # W&B logging with proper plots
                if use_wandb and global_step % 10 == 0:
                    # Select a few neurons and trials for visualization
                    num_neurons_to_plot = min(5, preds_rates.shape[2])
                    num_trials_to_plot = min(3, preds_rates.shape[0])
                    
                    # Create comparison plot
                    fig = create_rate_comparison_plot(
                        100 * spikes[:num_trials_to_plot, :, :num_neurons_to_plot].to(device),
                        100 * preds_rates[:num_trials_to_plot, :, :num_neurons_to_plot],
                        100 * ground_truth[:num_trials_to_plot, :, :num_neurons_to_plot].to(device),
                        num_neurons=num_neurons_to_plot,
                        num_trials=num_trials_to_plot
                    )
                    
                    wandb.log({
                        "rate_comparison": wandb.Image(fig),
                        "epoch": global_step
                    })
                    plt.close(fig)
                    
                    # Log correlation heatmap
                    # Sample some neurons for correlation heatmap
                    n_sample = min(20, preds_rates.shape[2])
                    pred_sample = preds_rates[:, :, :n_sample].reshape(-1, n_sample)
                    true_sample = ground_truth[:, :, :n_sample].to(device).reshape(-1, n_sample)
                    
                    # Compute correlations between each predicted and true neuron
                    corr_matrix = np.zeros((n_sample, n_sample))
                    for i in range(n_sample):
                        for j in range(n_sample):
                            corr = torch.corrcoef(torch.stack([pred_sample[:, i], true_sample[:, j]]))[0, 1]
                            corr_matrix[i, j] = corr.cpu().numpy()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_xlabel('True Neuron Index')
                    ax.set_ylabel('Predicted Neuron Index')
                    ax.set_title('Correlation between Predicted and True Rates')
                    plt.colorbar(im, ax=ax)
                    
                    wandb.log({
                        "correlation_heatmap": wandb.Image(fig),
                        "epoch": global_step
                    })
                    plt.close(fig)

            i += 1
            
    return float(np.mean(losses)), float(np.mean(r2s))


###############################################################################
#  Main
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train masked Transformer autoencoder on spike data.")
    p.add_argument("--dataset", default="lorenz", choices=["lorenz", *_DATASET_MAP.keys()], help="Dataset name.")
    p.add_argument("--data-root", default="data/h5", help="Path containing dataset files.")
    p.add_argument("--pos-encoding", default="sin", help="Positional encoding.")
    p.add_argument("--hidden-dim", type=int, default=32, help="Inner channel size for Transformer layers.")
    p.add_argument("--ffn-dim", type=int, default=128, help="Dim of feedforward layer of Transformer layers.")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for Transformer layers.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask-ratio", type=float, default=0.25, help="Fraction of timesteps to mask during training.")
    p.add_argument("--num-layers", type=int, default=4, help="Number of Transformer layers.")
    p.add_argument("--num-heads", type=int, default=1, help="Number of Transformer heads.")
    p.add_argument("--projection", default="linear", choices=["linear", "identity", "tied", "input_only", "output_only"])
    p.add_argument("--context-forward", type=int, default=10, help="Forward context length for Transformer (-1 = full context, 0 = causal, 1+ = partial).")
    p.add_argument("--context-backward", type=int, default=10, help="Backward context length for Transformer (-1 = full context, 1+ = partial).")
    p.add_argument("--logdir", default="runs", help="TensorBoard log directory.")
    p.add_argument("--checkpoint", default="checkpoint.pt", help="File to save best model (by val NLL).")
    p.add_argument("--model-type", default="pm", choices=["pm", "ndt", "unet"], help="Model type to use.")
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="neural-autoencoder", help="W&B project name.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team name.")
    p.add_argument("--wandb-name", default=None, help="W&B run name.")
    p.add_argument('--delta', type=int, default=1, help="Maximum shift for vectorized circshift (default: 1).")
    return p.parse_args()

def circshift_collate_fn(batch, delta=1):
    def fun(batch):
        """Vectorized version for better performance"""
        data_list, truth_list, behavior_list = zip(*batch)
        
        data = torch.stack(data_list)
        truth = torch.stack(truth_list)
        behavior = torch.stack(behavior_list)
        
        batch_size = data.size(0)
        seq_len = data.size(1)
        
        # Generate random shifts for each sample
        shifts = torch.randint(-delta, delta + 1, (batch_size,))
        
        # Apply shifts using advanced indexing
        indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        shifted_indices = (indices - shifts.unsqueeze(1)) % seq_len
        
        # Apply the shifts
        data = data.gather(1, shifted_indices.unsqueeze(-1).expand(-1, -1, data.size(-1)))
        return data, truth, behavior

    return fun


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(args.dataset, args.data_root)

    train_data = torch.from_numpy(data["train_data"]).int()
    val_data = torch.from_numpy(data["val_data"]).int()

    has_ground_truth = False
    try:
        val_truth = torch.from_numpy(data["val_truth"])
        has_ground_truth = True
        print(f"Found ground truth for val data (different from input: {has_ground_truth})")
    except KeyError:
        # No ground truth available, use the same as input
        val_truth = val_data.clone()
        print("No ground truth for val data available")
    train_truth = torch.from_numpy(data.get("train_truth", train_data.numpy()))

    try:
        train_behavior = torch.from_numpy(data["train_behavior"]).float()
        val_behavior = torch.from_numpy(data["val_behavior"]).float()
    except KeyError:
        train_behavior = torch.zeros_like(train_data, dtype=torch.float32)
        val_behavior = torch.zeros_like(val_data, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_data, train_truth, train_behavior), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, collate_fn=circshift_collate_fn(args.delta))
    val_loader = DataLoader(TensorDataset(val_data, val_truth, val_behavior), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    n_neurons = train_data.shape[2]
    trial_len = train_data.shape[1]

    if args.model_type == "rendt":
        from ndt_reimplementation import instantiate_autoencoder
        net = instantiate_autoencoder(args, n_neurons, trial_len).to(device)
    elif args.model_type == "unet":
        from src import unet
        net = unet.UNet1D(
            nlayers=args.num_layers,
            dim=n_neurons,
            latent_dim=args.hidden_dim,
            upsample=unet.UpsampleMethod.LINEAR
        ).to(device)
    elif args.model_type == "ndt":
        from ndt_transformer import instantiate_autoencoder
        net = instantiate_autoencoder(args, n_neurons, trial_len, device).to(device)
    

    # Calculate model parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    # Prepare hyperparameters dictionary
    hyperparams = {
        # Model architecture
        "model/input_dim": n_neurons,
        "model/hidden_dim": args.hidden_dim,
        "model/num_layers": args.num_layers,
        "model/num_heads": args.num_heads,
        "model/ffn_dim": 256,  # default from TransformerAutoencoder
        "model/dropout": args.dropout,
        "model/pos_encoding": args.pos_encoding,
        "model/projection": args.projection,
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        
        # Dataset info
        "data/dataset": args.dataset,
        "data/n_neurons": n_neurons,
        "data/trial_length": trial_len,
        "data/n_train_trials": len(train_data),
        "data/n_val_trials": len(val_data),
        "data/has_ground_truth": has_ground_truth,
        
        # Training hyperparameters
        "train/batch_size": args.batch_size,
        "train/epochs": args.epochs,
        "train/learning_rate": args.lr,
        "train/mask_ratio": args.mask_ratio,
        "train/optimizer": "Adam",
        "train/scheduler": "StepLR",
        "train/scheduler_step_size": args.epochs // 2,
        "train/scheduler_gamma": 0.3,
        
        # Other
        "device": str(device),
    }
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=hyperparams
        )
        wandb.watch(net, log="all", log_freq=100)

    criterion = nn.PoissonNLLLoss(reduction="none", log_input=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.3)
    from optim import WarmupCosineSchedule

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=min(40, int(args.epochs * .1)),
        t_total=args.epochs,
    )

    writer = SummaryWriter(log_dir=args.logdir)
    
    # Log hyperparameters to TensorBoard
    writer.add_text("hyperparameters", 
                    "\n".join([f"{k}: {v}" for k, v in hyperparams.items()]), 
                    global_step=0)
    
    # Log as scalars for easier filtering in TensorBoard
    for key, value in hyperparams.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"hyperparams/{key}", value, 0)
    
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        
        train_loss = train_one_epoch(
            net, train_loader, device, criterion, optimizer, mask_ratio=args.mask_ratio
        )
        # TensorBoard logging
        writer.add_scalar("train/nll", train_loss, epoch)
        log_data = {}
        if epoch % 10 == 0:
            val_loss, val_r2 = evaluate(
                net, val_loader, device, criterion, 
                mask_ratio=args.mask_ratio, 
                writer=writer, 
                global_step=epoch,
                use_wandb=args.use_wandb,
                has_ground_truth=has_ground_truth
            )

            tqdm.write(
                f"Epoch {epoch:03d} | train NLL {train_loss:.4f} | val NLL {val_loss:.4f} | val R² {val_r2:.4f}"
            )

            writer.add_scalar("val/nll", val_loss, epoch)
            writer.add_scalar("val/r2", val_r2, epoch)

            log_data = {"val/nll": val_loss,
                    "val/r2": val_r2}

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch, 
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "args": args
                }
                torch.save(checkpoint, args.checkpoint)
                
                # Save to W&B
                if args.use_wandb:
                    wandb.save(args.checkpoint)
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_r2"] = val_r2
                    wandb.run.summary["best_epoch"] = epoch

        # W&B logging
        if args.use_wandb:
            wandb.log({
                "train/nll": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch, **log_data
            })

                
        scheduler.step()

    writer.close()
    
    if args.use_wandb:
        wandb.finish()
        
    print(f"Training complete. Best val NLL {best_val_loss:.4f}. Model saved to {args.checkpoint}.")
    print(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")


if __name__ == "__main__":
    main()