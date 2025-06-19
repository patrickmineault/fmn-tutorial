# %% [markdown]
# # Tutorial 1: Masked Autoencoding for Neural Data
# 
# This tutorial walks through the core pieces of the repository and shows how to
# train a small masked autoencoder on neural spike data. We work with the Lorenz
# dataset provided with this repository. The same code can be adapted to larger
# datasets such as `mc_maze`.
#
# We will:
#
# 1. Load spike data using the `SpikesDataset` class.
# 2. Visualize example spike trains.
# 3. Build a small UNet-based autoencoder.
# 4. Train the model using masked autoencoding.
# 5. Inspect the learned representation.
#
# Running this notebook requires the dependencies listed in `requirements.txt`.
# Make sure the toy H5 files have been downloaded into `data/h5` as explained in
# the repository `README.md`.
# %%

# %% [markdown]
# ## Imports
# First we import PyTorch and the helper utilities from the `src` package.
# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import mask, unet
from src.dataset import DATASET_MODES, SpikesDataset

# %% [markdown]
# ## Loading the dataset
# The configuration files inside `data/config` describe where the H5 files live
# and contain a few training parameters. We'll use the Lorenz dataset which
# contains simulated spike trains as well as ground-truth firing rates.
# %%

data_cfg = Path("data/config/lorenz.yaml")
train_ds = SpikesDataset(data_cfg)
val_ds = SpikesDataset(data_cfg, mode=DATASET_MODES.val)

print("Train samples:", len(train_ds))
print("Validation samples:", len(val_ds))
print("Input shape:", train_ds[0][0].shape)

# %% [markdown]
# ### Visualize a raster plot
# Let's look at one example trial from the dataset. We plot spikes for each
# neuron over time.
# %%

spikes, rates, *_ = train_ds[0]
plt.figure(figsize=(8, 3))
plt.imshow(spikes.numpy(), aspect="auto", origin="lower")
plt.xlabel("time")
plt.ylabel("neuron")
plt.title("Example spike raster")
plt.colorbar(label="spike count")
plt.show()

# %% [markdown]
# ## Building a UNet autoencoder
# The repository provides a 1D UNet architecture in `src/unet.py`. We wrap this
# model with a small helper that applies a `Softplus` non‑linearity so the output
# is positive (spike rates must be non‑negative).
# %%

class UNetAutoencoder(nn.Module):
    def __init__(self, dim, latent_dim=16, nlayers=2):
        super().__init__()
        self.unet = unet.UNet1D(nlayers=nlayers, dim=dim, latent_dim=latent_dim)
        self.nonlinear = nn.Softplus()

    def forward(self, x):
        return self.nonlinear(self.unet(x))

# %% [markdown]
# ## Masking strategy
# Masked autoencoding randomly hides a fraction of the input tokens. The provided
# `Masker` module supports masking entire timesteps or entire neurons. Here we
# mask 20% of timesteps.
# %%

masker = mask.Masker(mask.MaskMode.timestep, mask_ratio=0.2)

# %% [markdown]
# ## Training utilities
# We define a small helper function that performs one optimization step. This is
# similar to `model_step` in `scripts/train.py` but simplified for clarity.
# %%

def train_step(net, batch, device, criterion):
    spikes, _, *_ = batch
    spikes = spikes.to(device)
    the_mask = masker(spikes)
    masked_spikes = spikes * (1 - the_mask)

    preds = net(masked_spikes.float())

    loss = criterion(preds[the_mask], spikes[the_mask].float())
    return loss, preds, the_mask

# %% [markdown]
# ## Running the training loop
# We iterate over the dataset with a PyTorch `DataLoader` and train for a few
# epochs. Each epoch processes all batches from the training set while showing a
# progress bar using `tqdm`.
# %%

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UNetAutoencoder(dim=spikes.shape[0], latent_dim=16, nlayers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

num_epochs = 5  # keep small for the tutorial

for epoch in range(num_epochs):
    net.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
        optimizer.zero_grad()
        loss, preds, _ = train_step(net, batch, device, criterion)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    net.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val epoch {epoch+1}"):
            loss, _, _ = train_step(net, batch, device, criterion)
            val_losses.append(loss.item())

    print(
        f"Epoch {epoch+1}: train loss {sum(train_losses)/len(train_losses):.4f}, "
        f"val loss {sum(val_losses)/len(val_losses):.4f}"
    )

# %% [markdown]
# ## Inspecting predictions
# After training we can visualize how well the model reconstructs held‑out
# timesteps. Below we compare the original spikes and the model predictions for a
# short segment.
# %%

net.eval()
example_batch = next(iter(val_loader))
with torch.no_grad():
    loss, preds, the_mask = train_step(net, example_batch, device, criterion)

spikes = example_batch[0]
masked_spikes = spikes * (1 - the_mask.cpu())

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(spikes[0].numpy(), aspect="auto", origin="lower")
plt.title("Target spikes")
plt.subplot(1, 3, 2)
plt.imshow(masked_spikes[0].numpy(), aspect="auto", origin="lower")
plt.title("Masked input")
plt.subplot(1, 3, 3)
plt.imshow(preds[0].cpu().numpy(), aspect="auto", origin="lower")
plt.title("Model prediction")
plt.tight_layout()
plt.show()

# %% [markdown]
# This concludes the minimal working example of training a masked autoencoder on
# spike data using the utilities provided in this repository. From here you could
# experiment with deeper networks, different masking ratios, or additional
# datasets such as `mc_maze_small.yaml`.
# %%
